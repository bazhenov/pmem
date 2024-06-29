//! # Page Management
//!
//! This module provides a system for managing and manipulating snapshots of a memory. It is designed
//! to facilitate operations on persistent memory (pmem), allowing for efficient snapshots, modifications, and commits
//! of data changes. The core functionality revolves around the `PagePool` structure, which manages a pool of pages,
//! and the `Snapshot` structure, which represents a modifiable snapshot of the page pool's state at a given point
//! in time.
//!
//! ## Concepts
//!
//! - **Page Pool**: A collection of pages that can be snapshot, modified, and committed. It acts as the primary
//! interface for interacting with the page memory.
//!
//! - **Snapshot**: A snapshot represents the state of the page pool at a specific moment.
//!   It can be modified independently of the pool, and later committed back to the pool to update its state.
//! - **Commit**: The act of applying the changes made in a snapshot back to the page pool, updating the pool's state
//!   to reflect those changes.
//! - **Patch**: A modification recorded in a snapshot. It consists of the address where the modification starts and
//!   the bytes that were written.
//!
//! ## Usage
//!
//! The module is designed to be used as follows:
//!
//! 1. **Initialization**: Create a `PagePool` with a specified number of pages.
//! 2. **Snapshotting**: Create a snapshot of the current state of the `PagePool`.
//! 3. **Modification**: Use the snapshot to perform modifications. Each modification is recorded as a patch.
//! 4. **Commit**: Commit the snapshot back to the `PagePool`, applying all the patches and updating the pool's state.
//!
//! ## Example
//!
//! ```rust
//! use pmem::page::PagePool;
//!
//! let mut pool = PagePool::new(5); // Initialize a pool with 5 pages
//! let mut snapshot = pool.snapshot(); // Create a snapshot of the current state
//! snapshot.write(0, &[1, 2, 3, 4]); // Write 4 bytes at offset 0.
//! pool.commit(snapshot); // Commit the changes back to the pool
//! ```
//!
//! ## Safety and Correctness
//!
//! The module ensures safety and correctness through the following mechanisms:
//!
//! - **Immutability of Committed Snapshots**: Once a snapshot is committed, it becomes immutable, ensuring that
//! any reference to its data remains valid and unchanged until corresponding `Rc` reference is held.
//! - **Linear Snapshot History**: The module enforces a linear history of snapshots, preventing branches in the
//! snapshot history and ensuring consistency of changes proposed in snapshots.
//!
//! ## Performance Considerations
//!
//! Since snapshots do not require duplicating the entire state of the page pool, they can be created with minimal
//! overhead, making it perfectly valid and cost-effective to create a snapshot even when the intention is only to
//! read data without any modifications.

use crate::ensure;
use std::{borrow::Cow, iter, ops::Range, rc::Rc, usize};

pub const PAGE_SIZE: usize = 1 << 24; // 16Mb
pub type Addr = u32;
pub type PageOffset = u32;
pub type PageNo = u32;
pub type LSN = u64;

type Result<T> = std::result::Result<T, Error>;

/// Represents a modification recorded in a snapshot.
///
/// A `Patch` can either write a range of bytes starting at a specified offset
/// or recalim a range of bytes starting at a specified offset.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq, Debug))]
enum Patch {
    // Write given data at a specified offset
    Write(PageOffset, Vec<u8>),

    /// Reclaim given amount of bytes at a specified address
    Reclaim(PageOffset, usize),
}

impl Patch {
    pub fn addr(&self) -> u32 {
        match self {
            Patch::Write(offset, _) => *offset,
            Patch::Reclaim(offset, _) => *offset,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Patch::Write(_, bytes) => bytes.len(),
            Patch::Reclaim(_, len) => *len,
        }
    }

    pub fn normalize(self, other: Patch) -> NormalizedPatches {
        if other.fully_covers(&self) {
            NormalizedPatches::Merged(other)
        } else if self.fully_covers(&other) {
            self.normalize_covered(other)
        } else if self.adjacent(&other) {
            self.normalize_adjacency(other)
        } else if self.overlaps(&other) {
            self.normalize_overlap(other)
        } else {
            // Not connected patches
            let (a, b) = self.reorder(other);
            NormalizedPatches::Reordered(a, b)
        }
    }

    fn end(&self) -> u32 {
        self.addr() + self.len() as u32
    }

    fn to_range(&self) -> Range<u32> {
        self.addr()..self.addr() + self.len() as u32
    }

    fn overlaps(&self, other: &Patch) -> bool {
        let a = self.to_range();
        let b = other.to_range();

        a.contains(&b.start) || b.contains(&a.start)
    }

    fn adjacent(&self, other: &Patch) -> bool {
        let a = self.to_range();
        let b = other.to_range();

        a.start == b.end || b.start == a.end
    }

    fn fully_covers(&self, other: &Patch) -> bool {
        self.addr() <= other.addr() && other.end() <= self.end()
    }

    fn trim_after(&mut self, at: u32) {
        let end = self.end();
        assert!(self.addr() < at && at < end, "Out-of-bounds");
        match self {
            Patch::Write(offset, data) => data.truncate((at - *offset) as usize),
            Patch::Reclaim(_, len) => {
                *len -= (end - at) as usize;
            }
        }
    }

    fn trim_before(&mut self, at: u32) {
        assert!(self.addr() < at && at < self.end(), "Out-of-bounds");
        match self {
            Patch::Write(offset, data) => {
                data.truncate((at - *offset) as usize);
                *offset = at;
            }
            Patch::Reclaim(offset, len) => {
                *len -= (at - *offset) as usize;
                *offset = at;
            }
        }
    }

    /// Normalize 2 patches when they are overlapping
    fn normalize_overlap(self, other: Patch) -> NormalizedPatches {
        use {NormalizedPatches::*, Patch::*};
        assert!(self.overlaps(&other));

        match (self, other) {
            (Write(a_offset, a_data), Write(b_offset, b_data)) => {
                let a_end = a_offset as usize + a_data.len();
                let b_end = b_offset as usize + b_data.len();
                let start = a_offset.min(b_offset) as usize;
                let end = a_end.max(b_end);

                let mut result_data = vec![0; end - start];

                let range = {
                    let start = a_offset as usize - start;
                    let end = start + a_data.len();
                    start..end
                };
                result_data[range].copy_from_slice(&a_data[..]);

                let range = {
                    let start = b_offset as usize - start;
                    let end = start + b_data.len();
                    start..end
                };
                result_data[range].copy_from_slice(&b_data[..]);

                Merged(Write(start as u32, result_data))
            }
            (a @ Reclaim(_, _), b @ Reclaim(_, _)) => {
                let offset = a.addr().min(b.addr());
                let end = a.end().max(b.end()) as usize;
                let len = end - offset as usize;
                Merged(Reclaim(offset, len))
            }
            (mut a, b) => {
                if a.addr() < b.addr() {
                    a.trim_after(b.addr());
                } else {
                    a.trim_before(b.end());
                }
                let (a, b) = a.reorder(b);
                Reordered(a, b)
            }
        }
    }

    /// Normalize 2 patches if the are adjacent
    ///
    /// Only adjacent patches of the same type can be merged
    fn normalize_adjacency(self, other: Patch) -> NormalizedPatches {
        use {NormalizedPatches::*, Patch::*};
        assert!(self.adjacent(&other));

        match self.reorder(other) {
            (Write(addr, mut a), Write(_, b)) => {
                a.extend(b);
                Merged(Write(addr, a))
            }
            (Reclaim(addr, a), Reclaim(_, b)) => Merged(Reclaim(addr, a + b)),
            (a, b) => Reordered(a, b),
        }
    }

    fn normalize_covered(self, other: Patch) -> NormalizedPatches {
        use {NormalizedPatches::*, Patch::*};
        assert!(self.fully_covers(&other));

        match (self, other) {
            (Write(a_addr, mut bytes), b @ Reclaim(_, _)) => {
                let split_idx = (b.addr() - a_addr) as usize + b.len();
                let right = Write(b.end(), bytes.split_off(split_idx));

                bytes.truncate(bytes.len() - b.len());
                let left = Write(a_addr, bytes);

                assert!(left.len() != 0 || right.len() != 0);
                if left.len() == 0 {
                    Reordered(b, right)
                } else if right.len() == 0 {
                    Reordered(left, b)
                } else {
                    Splitted(left, b, right)
                }
            }
            (a @ Reclaim(_, _), b @ Write(_, _)) => {
                let left = Reclaim(a.addr(), (b.addr() - a.addr()) as usize);
                let right = Reclaim(b.end(), (a.end() - b.end()) as usize);

                assert!(left.len() != 0 || right.len() != 0);
                if left.len() == 0 {
                    Reordered(b, right)
                } else if right.len() == 0 {
                    Reordered(left, b)
                } else {
                    Splitted(left, b, right)
                }
            }
            (Write(a_addr, mut a_data), Write(b_addr, b_data)) => {
                let start = (b_addr - a_addr) as usize;
                let end = start + b_data.len();
                a_data[start..end].copy_from_slice(&b_data);

                Merged(Write(a_addr, a_data))
            }
            (a @ Reclaim(_, _), Reclaim(_, _)) => Merged(a),
        }
    }

    fn reorder(self, other: Patch) -> (Patch, Patch) {
        if self.addr() < other.addr() {
            (self, other)
        } else {
            (other, self)
        }
    }

    #[cfg(test)]
    fn set_addr(&mut self, value: u32) {
        match self {
            Patch::Write(addr, _) => *addr = value,
            Patch::Reclaim(addr, _) => *addr = value,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum NormalizedPatches {
    Merged(Patch),
    Reordered(Patch, Patch),
    Splitted(Patch, Patch, Patch),
}

impl NormalizedPatches {
    #[cfg(test)]
    fn len(self) -> usize {
        match self {
            NormalizedPatches::Merged(p1) => p1.len(),
            NormalizedPatches::Reordered(p1, p2) => p1.len() + p2.len(),
            NormalizedPatches::Splitted(p1, p2, p3) => p1.len() + p2.len() + p3.len(),
        }
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &Patch> {
        let slice = match self {
            NormalizedPatches::Merged(a) => [Some(a), None, None],
            NormalizedPatches::Reordered(a, b) => [Some(a), Some(b), None],
            NormalizedPatches::Splitted(a, b, c) => [Some(a), Some(b), Some(c)],
        };
        slice.into_iter().filter_map(|i| i)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("Read of the page out-of-bounds")]
    OutOfBounds,
}

/// A pool of pages that can capture and commit snapshots of data changes.
///
/// The `PagePool` struct allows for the creation of snapshots representing the state
/// of a set of pages at a particular point in time. These snapshots can then be modified
/// and eventually committed back to the pool, updating the pool's state to reflect the changes
/// made in the snapshot.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// # use pmem::page::PagePool;
/// let mut pool = PagePool::new(5);    // Initialize a pool with 5 pages
/// let mut snapshot = pool.snapshot(); // Create a snapshot of the current state
/// snapshot.write(0, &[0, 1, 2, 3]);   // Modify the snapshot
/// pool.commit(snapshot);              // Commit the changes back to the pool
/// ```
///
/// This structure is particularly useful for systems that require consistent views of data
/// at different points in time, or for implementing undo/redo functionality where each snapshot
/// can represent a state in the history of changes.
#[derive(Default)]
pub struct PagePool {
    latest: Rc<CommitedSnapshot>,
}

impl PagePool {
    /// Constructs a new `PagePool` with a specified number of pages.
    ///
    /// This function initializes a `PagePool` instance with an empty set of patches and
    /// a given number of pages.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages the pool should initially contain. This determines
    /// the range of valid addresses that can be written to in snapshots derived from this pool.
    pub fn new(pages: usize) -> Self {
        let snapshot = CommitedSnapshot {
            patches: vec![],
            base: None,
            pages: pages as u32,
            lsn: 1,
        };
        Self {
            latest: Rc::new(snapshot),
        }
    }

    /// Takes a snapshot of the current state of the `PagePool`.
    ///
    /// This method creates a `Snapshot` instance representing the current state of the page pool.
    /// The snapshot can be used to perform modifications independently of the pool. These modifications
    /// are not reflected in the pool until the snapshot is committed using the [`commit`] method.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use pmem::page::PagePool;
    /// # let mut pool = PagePool::new(5);
    /// let snapshot = pool.snapshot();
    /// // The snapshot can now be modified, and those modifications
    /// // won't affect the original pool until committed.
    /// ```
    ///
    /// # Returns
    ///
    /// A `Snapshot` instance representing the current state of the page pool. It can be used both for reading and
    /// changing state of the page pool.
    ///
    /// [`commit`]: Self::commit
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            patches: vec![],
            base: Rc::clone(&self.latest),
            pages: self.latest.pages,
        }
    }

    /// Commits the changes made in a snapshot back to the page pool.
    ///
    /// This method updates the state of the page pool to reflect the modifications
    /// recorded in the provided snapshot. Once committed, the snapshot becomes part
    /// of the page pool's history, and its changes are visible in subsequent snapshots.
    ///
    /// Each snapshot is linked to the pool state it was created from. If the page poll was changed
    /// since the moment when snapshot was created, attempt to commit such a snapshot will return an error,
    /// because such changes might not be consistent anymore.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - A snapshot containing modifications to commit to the page pool.
    pub fn commit(&mut self, snapshot: Snapshot) {
        assert!(
            Rc::ptr_eq(&self.latest, &snapshot.base),
            "Proposed snaphot is not linear"
        );
        let lsn = self.latest.lsn + 1;
        self.latest = Rc::new(CommitedSnapshot {
            patches: snapshot.patches,
            base: Some(Rc::clone(&self.latest)),
            pages: snapshot.pages,
            lsn,
        });
    }

    #[cfg(test)]
    fn read(&self, addr: PageOffset, len: usize) -> Result<Cow<[u8]>> {
        self.latest.read(addr, len)
    }
}

/// Represents a committed snapshot of a page pool.
///
/// A `CommitedSnapshot` captures the state of a page pool at a specific point in time,
/// including any patches (modifications) that have been applied up to that point. It serves
/// as a read-only view into the historical state of the pool, allowing for consistent reads
/// of pages as they existed at the time of the snapshot.
///
/// Each `CommitedSnapshot` can optionally reference a base snapshot, forming a chain
/// that represents the full history of modifications leading up to the current state.
/// This chain is traversed backwards when reading from a snapshot to reconstruct the state
/// of a page by applying patches in reverse chronological order.
pub struct CommitedSnapshot {
    /// A patches that have been applied in this snapshot.
    patches: Vec<Patch>,

    /// A reference to the base snapshot from which this snapshot was derived.
    /// If present, the base snapshot represents the state of the page pool
    /// immediately before the current snapshot's patches were applied.
    base: Option<Rc<CommitedSnapshot>>,

    /// The total number of pages represented by this snapshot. This is used to
    /// validate read requests and ensure they do not exceed the bounds of the snapshot.
    pages: PageNo,

    /// A log sequence number (LSN) that uniquely identifies this snapshot. The LSN
    /// is used internally to ensure that snapshots are applied in a linear and consistent order.
    lsn: LSN,
}

impl Default for CommitedSnapshot {
    fn default() -> Self {
        Self {
            patches: vec![],
            base: None,
            pages: 1,
            lsn: 1,
        }
    }
}

impl CommitedSnapshot {
    pub fn read(&self, addr: PageOffset, len: usize) -> Result<Cow<'_, [u8]>> {
        assert!(len <= PAGE_SIZE, "Out of bounds read");

        let (page_no, _) = split_ptr(addr);
        ensure!(page_no < self.pages, Error::OutOfBounds);

        let mut buffer = vec![0; len];
        self.read_to_buf(addr, &mut buffer)?;
        Ok(Cow::Owned(buffer))
    }

    pub fn read_to_buf(&self, addr: PageOffset, buf: &mut [u8]) -> Result<()> {
        assert!(buf.len() <= PAGE_SIZE, "Out of bounds read");

        let (page_no, _) = split_ptr(addr);
        ensure!(page_no < self.pages, Error::OutOfBounds);

        let range = (addr as usize)..addr as usize + buf.len();

        let mut patch_list = vec![self.patches.as_slice()];
        #[allow(clippy::useless_asref)]
        let snapshots = self
            .base
            .as_ref()
            .map(Rc::clone)
            .map(collect_snapshots)
            .unwrap_or_default();
        patch_list.extend(snapshots.iter().map(|s| s.patches.as_slice()));

        apply_patches(&patch_list, buf, &range);
        Ok(())
    }
}

#[derive(Clone)]
pub struct Snapshot {
    patches: Vec<Patch>,
    base: Rc<CommitedSnapshot>,
    pages: PageNo,
}

impl Snapshot {
    pub fn write(&mut self, addr: Addr, bytes: &[u8]) {
        assert!(bytes.len() <= PAGE_SIZE, "Buffer too large");
        self.patches.push(Patch::Write(addr, bytes.to_vec()))
    }

    pub fn read(&self, addr: PageOffset, len: usize) -> Result<Cow<'_, [u8]>> {
        let (page_no, _) = split_ptr(addr);
        ensure!(page_no < self.pages, Error::OutOfBounds);

        let mut buf = vec![0; len];
        self.base.read_to_buf(addr, &mut buf)?;

        let addr = addr as usize;
        let range = addr..addr + len;
        apply_patches(&[self.patches.as_slice()], &mut buf, &range);
        Ok(Cow::Owned(buf))
    }

    /// Frees the given segment of memory.
    ///
    /// Reclaim is guaranteed to follow zeroing semantics. The read operation from the corresponding segment
    /// after reclaiming will return zeros.
    ///
    /// The main purpose of reclaim is to mark memory as not used, so it can be freed from persistent storage
    /// after snapshot compaction.
    pub fn reclaim(&mut self, addr: PageOffset, len: usize) -> Result<()> {
        let (page_no, _) = split_ptr(addr);
        ensure!(page_no < self.pages && len < PAGE_SIZE, Error::OutOfBounds);
        self.patches.push(Patch::Reclaim(addr, len));
        Ok(())
    }

    #[cfg(test)]
    fn resize(&mut self, pages: usize) {
        self.pages = pages as PageNo;
    }
}

fn collect_snapshots(snapshot: Rc<CommitedSnapshot>) -> Vec<Rc<CommitedSnapshot>> {
    let mut snapshots = vec![];
    #[allow(clippy::useless_asref)]
    let mut snapshot = Some(snapshot);
    #[allow(clippy::useless_asref)]
    while let Some(s) = snapshot {
        snapshots.push(Rc::clone(&s));
        snapshot = s.base.as_ref().map(Rc::clone);
    }
    snapshots
}

/// Applies a list of patches to a given buffer within a specified range.
///
/// This function iterates over the provided snapshots and applies each patch from each snapshot
/// that intersects with the specified range to the given buffer. Snapshots must be provided in reversed order
/// (recent first).
///
/// # Arguments
///
/// * `snapshots` - A slice of slices of `Patch` instances. Each inner slice represents
///                 a set of patches to be applied, and the outer slice represents a chain
///                 of snapshots, with the most recent snapshot first.
/// * `buffer` - A mutable reference to the buffer where the patches will be applied.
/// * `range` - A range specifying the portion of the buffer to which the patches should be applied.
fn apply_patches(snapshots: &[&[Patch]], buffer: &mut [u8], range: &Range<usize>) {
    for patches in snapshots.iter().rev() {
        for patch in patches.iter().filter(|p| intersects(p, range)) {
            // Calculating intersection of the path and input interval
            let offset = patch.addr() as usize;
            let start = range.start.max(offset);
            let end = range.end.min(offset + patch.len());
            let len = end - start;

            let slice_range = {
                let from = start.saturating_sub(range.start);
                from..from + len
            };

            let patch_range = {
                let from = start.saturating_sub(offset);
                from..from + len
            };

            match patch {
                Patch::Write(_, bytes) => buffer[slice_range].copy_from_slice(&bytes[patch_range]),
                Patch::Reclaim(_, _) => buffer[slice_range].fill(0),
            }
        }
    }
}

fn merge_patches(mut input: Vec<Patch>) -> Vec<Patch> {
    input.sort_unstable_by_key(Patch::addr);

    if input.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(input.capacity());
    result.push(input.remove(0));

    for next in input {
        // always Some(_) because we start with non empty result
        let last = result.last().unwrap();
        if next.overlaps(last) {
            let last = result.pop().unwrap();
        }
    }

    result
}

fn split_ptr(addr: Addr) -> (PageNo, PageOffset) {
    const PAGE_SIZE_BITS: u32 = PAGE_SIZE.trailing_zeros();
    let page_no = addr >> PAGE_SIZE_BITS;
    let offset = addr & ((PAGE_SIZE - 1) as u32);
    (page_no, offset)
}

/// Returns true of given patch intersects given range of bytes
fn intersects(patch: &Patch, range: &Range<usize>) -> bool {
    let start = patch.addr() as usize;
    let end = start + patch.len();
    start < range.end && end > range.start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_new_page() -> Result<()> {
        let snapshot = CommitedSnapshot::from("foo".as_bytes());
        assert_str_eq(snapshot.read(0, 3)?, "foo");
        assert_str_eq(snapshot.read(3, 1)?, [0]);
        Ok(())
    }

    #[test]
    fn commited_changes_should_be_visible_on_a_page() -> Result<()> {
        let mut mem = PagePool::from("Jekyll");

        let mut snapshot = mem.snapshot();
        snapshot.write(0, b"Hide");
        mem.commit(snapshot);

        assert_str_eq(mem.read(0, 4)?, b"Hide");
        Ok(())
    }

    #[test]
    fn uncommited_changes_should_be_visible_only_on_the_snapshot() -> Result<()> {
        let mem = PagePool::from("Jekyll");

        let mut snapshot = mem.snapshot();
        snapshot.write(0, b"Hide");

        assert_str_eq(snapshot.read(0, 4)?, "Hide");
        assert_str_eq(mem.read(0, 6)?, "Jekyll");
        Ok(())
    }

    #[test]
    fn patch_page() -> Result<()> {
        let mut mem = PagePool::from("Hello panic!");

        let mut snapshot = mem.snapshot();
        snapshot.write(6, b"world");
        mem.commit(snapshot);

        assert_str_eq(mem.read(0, 12)?, "Hello world!");
        assert_str_eq(mem.read(0, 8)?, "Hello wo");
        assert_str_eq(mem.read(3, 9)?, "lo world!");
        assert_str_eq(mem.read(6, 5)?, "world");
        assert_str_eq(mem.read(8, 4)?, "rld!");
        assert_str_eq(mem.read(7, 3)?, "orl");
        Ok(())
    }

    #[test]
    fn data_across_multiple_pages_can_be_written() -> Result<()> {
        let mut mem = PagePool::new(3);

        let alice = b"Alice";
        let bob = b"Bob";
        let charlie = b"Charlie";

        // Addresses on 3 consequent pages
        let page_a = 0;
        let page_b = 1 * PAGE_SIZE as Addr;
        let page_c = 2 * PAGE_SIZE as Addr;

        let mut snapshot = mem.snapshot();
        snapshot.write(page_a, alice);
        snapshot.write(page_b, bob);
        snapshot.write(page_c, charlie);

        // Checking that data is visible in snapshot
        assert_str_eq(snapshot.read(page_a, alice.len())?, alice);
        assert_str_eq(snapshot.read(page_b, bob.len())?, bob);
        assert_str_eq(snapshot.read(page_c, charlie.len())?, charlie);

        mem.commit(snapshot);

        // Checking that data is visible after commit to page pool
        assert_str_eq(mem.read(page_a, alice.len())?, alice);
        assert_str_eq(mem.read(page_b, bob.len())?, bob);
        assert_str_eq(mem.read(page_c, charlie.len())?, charlie);

        Ok(())
    }

    #[test]
    fn data_can_be_removed_on_snapshot() -> Result<()> {
        let data = [1, 2, 3, 4, 5];
        let mem = PagePool::from(data.as_slice());

        let mut snapshot = mem.snapshot();
        snapshot.reclaim(1, 3)?;

        assert_eq!(snapshot.read(0, 5)?, [1u8, 0, 0, 0, 5].as_slice());
        Ok(())
    }

    #[test]
    fn page_pool_should_return_error_of_ptr_out_of_bounds() -> Result<()> {
        let mem = PagePool::default();

        let Err(Error::OutOfBounds) = mem.read(PAGE_SIZE as u32, 1) else {
            panic!("OutOfBounds should be geberated");
        };
        Ok(())
    }

    #[test]
    fn change_number_of_pages_on_commit() -> Result<()> {
        let mut mem = PagePool::new(2); // Initially 2 pages

        let page_a = 0;
        let page_b = 1 * PAGE_SIZE as Addr;

        let alice = b"Alice";
        let bob = b"Bob";

        let mut snapshot = mem.snapshot();
        snapshot.write(page_a, alice);
        snapshot.write(page_b, bob);
        mem.commit(snapshot);

        let mut snapshot = mem.snapshot();
        snapshot.resize(1); // removing second page
        let Err(Error::OutOfBounds) = snapshot.read(page_b, bob.len()) else {
            // changes should be visible immediatley on snapshot
            panic!("{} should be generated", Error::OutOfBounds);
        };
        mem.commit(snapshot);

        // changes also should be visible after commit
        let Err(Error::OutOfBounds) = mem.read(page_b, bob.len()) else {
            panic!("{} should be generated", Error::OutOfBounds);
        };

        Ok(())
    }

    mod patch {
        use super::*;
        use Patch::*;

        #[test]
        fn adjacent_patches_of_the_same_type() {
            assert_merged(
                Write(0, vec![0, 1, 2]),
                Write(3, vec![3, 4, 5]),
                Write(0, vec![0, 1, 2, 3, 4, 5]),
            );

            // TODO replace with proptests
            assert_merged(
                Write(3, vec![3, 4, 5]),
                Write(0, vec![0, 1, 2]),
                Write(0, vec![0, 1, 2, 3, 4, 5]),
            );
        }

        #[test]
        fn adjacent_patches_of_different_types() {
            assert_not_merged(Write(0, vec![0, 1, 2]), Reclaim(3, 3))
        }

        #[test]
        fn island_patches_of_the_same_type() {
            assert_merged(
                Write(0, vec![0, 1, 2]),
                Write(1, vec![10]),
                Write(0, vec![0, 10, 2]),
            );

            assert_merged(
                Write(1, vec![10]),
                Write(0, vec![0, 1, 2]),
                Write(0, vec![0, 1, 2]),
            );
        }

        #[test]
        fn island_patches_of_different_types() {
            assert_splitted(
                Write(0, vec![0, 1, 2]),
                Reclaim(1, 1),
                (Write(0, vec![0]), Reclaim(1, 1), Write(2, vec![2])),
            );

            assert_splitted(
                Reclaim(0, 3),
                Write(1, vec![1]),
                (Reclaim(0, 1), Write(1, vec![1]), Reclaim(2, 1)),
            );
        }

        #[test]
        fn overlapping_patches_of_different_types() {
            assert_reordered(
                Write(0, vec![0, 1, 2]),
                Reclaim(1, 15),
                (Write(0, vec![0]), Reclaim(1, 15)),
            );

            assert_reordered(
                Reclaim(1, 15),
                Write(0, vec![0, 1, 2]),
                (Write(0, vec![0, 1, 2]), Reclaim(3, 13)),
            );
        }

        #[test]
        fn overlapping_patches_of_the_same_type() {
            assert_merged(
                Write(0, vec![0, 1, 2]),
                Write(2, vec![20, 30, 40]),
                Write(0, vec![0, 1, 20, 30, 40]),
            );

            assert_merged(
                Write(2, vec![20, 30, 40]),
                Write(0, vec![0, 1, 2]),
                Write(0, vec![0, 1, 2, 30, 40]),
            );

            assert_merged(Reclaim(2, 3), Reclaim(4, 6), Reclaim(2, 8));
        }

        #[test]
        fn identical_patches_of_the_same_type() {
            assert_merged(
                Write(0, vec![0, 1, 2]),
                Write(0, vec![0, 1, 2]),
                Write(0, vec![0, 1, 2]),
            );
        }

        #[test]
        fn regression() {
            assert_reordered(
                Write(0, vec![0, 0, 0]),
                Reclaim(0, 1),
                (Reclaim(0, 1), Write(1, vec![0, 0])),
            );
            assert_reordered(
                Reclaim(0, 3),
                Write(0, vec![0]),
                (Write(0, vec![0]), Reclaim(1, 2)),
            );
        }

        #[test]
        fn identical_patches_of_different_types() {
            assert_merged(Write(0, vec![0, 1, 2]), Reclaim(0, 3), Reclaim(0, 3));
        }

        #[test]
        fn covered_patches() {
            assert_merged(Write(0, vec![0, 1, 2]), Reclaim(0, 4), Reclaim(0, 4));
        }

        fn assert_merged(a: Patch, b: Patch, expected: Patch) {
            match a.normalize(b) {
                NormalizedPatches::Merged(patch) => assert_eq!(patch, expected),
                result @ _ => panic!("Patch should be merged: {:?}", result),
            }
        }

        fn assert_reordered(a: Patch, b: Patch, expected: (Patch, Patch)) {
            match a.normalize(b) {
                NormalizedPatches::Reordered(p1, p2) => assert_eq!((p1, p2), expected),
                result @ _ => panic!("Patch should be merged: {:?}", result),
            }
        }

        fn assert_splitted(a: Patch, b: Patch, expected: (Patch, Patch, Patch)) {
            match a.normalize(b) {
                NormalizedPatches::Splitted(p1, p2, p3) => assert_eq!((p1, p2, p3), expected),
                result @ _ => panic!("Patch should be splitted: {:?}", result),
            }
        }

        fn assert_not_merged(a: Patch, b: Patch) {
            match a.clone().normalize(b.clone()) {
                NormalizedPatches::Reordered(p1, p2) => {
                    assert_eq!(p1, a);
                    assert_eq!(p2, b);
                }
                result @ _ => panic!("Patch should not be merged: {:?}", result),
            }
        }
    }

    mod ptr {
        use super::*;

        #[test]
        fn split_ptr_generic() {
            let addr = 0x0A_F01234;
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0x0A, "Unexpected page number");
            assert_eq!(offset, 0xF01234, "Unexpected offset");
        }

        #[test]
        fn split_ptr_at_boundary() {
            let addr = PAGE_SIZE as u32 - 1; // Last address of the first page
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(
                page_no, 0,
                "Page number should be 0 at the last address of the first page"
            );
            assert_eq!(
                offset,
                PAGE_SIZE as u32 - 1,
                "Offset should be at the page boundary"
            );
        }

        #[test]
        fn test_split_ptr_zero() {
            let addr = 0x00000000;
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0);
            assert_eq!(offset, 0);
        }

        #[test]
        fn test_split_ptr_max_addr() {
            let addr = Addr::MAX; // Maximum possible address
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0xFF, "Unexpected page number for maximum address");
            assert_eq!(offset, 0xFFFFFF, "Unexpected offset for maximum address");
        }
    }

    // #[cfg(not(miri))]
    mod proptests {
        use super::*;
        use proptest::{collection::vec, prelude::*};

        /// The size of database for testing
        const DB_SIZE: usize = 1024;

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 1000,
                ..ProptestConfig::default()
            })]

            /// This test uses "shadow writes" to check if snapshot writing and reading
            /// algorithms are consistent with sequential consistency. We do it by
            /// mirroring all patches to a shadow buffer sequentially. In the end,
            /// the final snapshot state should be equal to the shadow buffer.
            #[test]
            fn shadow_write(snapshots in vec(any_snapshot(), 0..10)) {
                let mut shadow_buffer = vec![0; DB_SIZE];
                let mut mem = PagePool::default();

                for patches in snapshots {
                    let mut snapshot = mem.snapshot();
                    for patch in patches {
                        let offset = patch.addr() as usize;
                        let range = offset..offset + patch.len();
                        match patch {
                            Patch::Write(offset, bytes) => {
                                snapshot.write(offset, &bytes);
                                shadow_buffer[range].copy_from_slice(bytes.as_slice());
                            },
                            Patch::Reclaim(offset, len) => {
                                snapshot.reclaim(offset, len)?;
                                shadow_buffer[range].fill(0);

                            }
                        }
                    }
                    mem.commit(snapshot);
                }

                assert_buffers_eq(&*mem.read(0, DB_SIZE).unwrap(), shadow_buffer.as_slice());
            }

            #[test]
            fn any_patches_length_should_be_positive((a, b) in patches::any_pair()) {
                match a.normalize(b) {
                    NormalizedPatches::Merged(a) => prop_assert!(a.len() > 0, "{:?}", a),
                    NormalizedPatches::Reordered(a, b) => prop_assert!(a.len() > 0 && b.len() > 0, "{:?}, {:?}", a, b),
                    NormalizedPatches::Splitted(a, b, c) => prop_assert!(a.len() > 0 && b.len() > 0 && c.len() > 0, "{:?}, {:?}, {:?}", a, b, c),
                }
            }

            #[test]
            fn patches_commutativity((a, b) in prop_oneof![patches::adjacent(), patches::disconnected()]) {
                let a_plus_b = a.clone().normalize(b.clone());
                let b_plus_a = b.normalize(a);
                prop_assert_eq!(a_plus_b, b_plus_a);
            }

            #[test]
            fn adjacent_patches_length((a, b) in patches::adjacent()) {
                let sum = a.len() + b.len();
                let result = a.normalize(b);
                prop_assert_eq!(result.len(), sum);
            }

            /// For any two covered patches the result patch length is a maximum of input patches length
            #[test]
            fn covered_patches_length((a, b) in patches::covered()) {
                let expected_len = a.len().max(b.len());
                prop_assert_eq!(a.clone().normalize(b.clone()).len(), expected_len);
                prop_assert_eq!(b.clone().normalize(a.clone()).len(), expected_len);
            }

            /// If B is fully covered by A, A should fully rewrite B
            #[test]
            fn covered_patches_rewrite((covering, covered) in patches::covered()) {
                let NormalizedPatches::Merged(result) = covered.normalize(covering.clone()) else {
                    panic!("Merged() expected");
                };
                prop_assert_eq!(covering, result);
            }

            #[test]
            fn covered_patches_ajacency((covering, covered) in patches::adjacent()) {
                match covering.normalize(covered.clone()) {
                    NormalizedPatches::Reordered(p1, p2) => prop_assert!(p1.addr() == p2.end() || p2.addr() == p1.end()),
                    NormalizedPatches::Merged(_) => {}
                    NormalizedPatches::Splitted(p1, p2, p3) => prop_assert!(false, "Merged()/Reordered() expected. Got: {:?}, {:?}, {:?}", p1, p2, p3)
                }
            }

            #[test]
            fn connected_patches_are_adjacent_after_normalization((a, b) in patches::any_connected_pair()) {
                match a.normalize(b) {
                    NormalizedPatches::Merged(_) => {},
                    NormalizedPatches::Reordered(a, b) => {
                        prop_assert!(a.end() == b.addr(), "Patches are not adjacent: {:?}, {:?}", a, b);
                    },
                    NormalizedPatches::Splitted(a, b, c) => {
                        prop_assert!(a.end() == b.addr(), "Patches are not adjacent: {:?}, {:?}", a, b);
                        prop_assert!(b.end() == c.addr(), "Patches are not adjacent: {:?}, {:?}", b, c);
                    },
                }
            }

            #[test]
            fn patches_are_ordered_after_normalization((a, b) in patches::any_pair()) {
                match a.normalize(b) {
                    NormalizedPatches::Merged(_) => {},
                    NormalizedPatches::Reordered(a, b) => {
                        prop_assert!(a.end() <= b.addr(), "Patches are not adjacent: {:?}, {:?}", a, b);
                    },
                    NormalizedPatches::Splitted(a, b, c) => {
                        prop_assert!(a.end() <= b.addr(), "Patches are not adjacent: {:?}, {:?}", a, b);
                        prop_assert!(b.end() <= c.addr(), "Patches are not adjacent: {:?}, {:?}", b, c);
                    },
                }
            }

            #[test]
            fn patches_have_positive_length((a, b) in patches::any_pair()) {
                for patch in a.normalize(b).iter() {
                    prop_assert!(patch.len() > 0);
                }
            }

            #[test]
            fn patches_do_not_overlaps((a, b) in patches::any_pair()) {
                match a.normalize(b) {
                    NormalizedPatches::Merged(_) => {},
                    NormalizedPatches::Reordered(a, b) => {
                        prop_assert!(!a.overlaps(&b), "Patches are overlapping: {:?}, {:?}", a, b);
                    },
                    NormalizedPatches::Splitted(ref a, ref b, ref c) => {
                        for (i, j) in [(a, b), (b, c), (c, a)] {
                            prop_assert!(!i.overlaps(j), "Patches are overlapping: {:?}, {:?}", i, j);
                        }
                    },
                }
            }

            // #[test]
            // fn overlapped((a, b) in patches::overlapped()) {
            //     let NormalizedPatches::Reordered(a, b) = a.normalize(b) else {
            //         panic!("Reordered() expected: ")
            //     };
            // }
        }

        /// A set of strategies to generate different patches for property testing
        mod patches {
            use prop::test_runner::TestRng;

            use super::*;
            use std::mem;

            pub(super) fn any_pair() -> impl Strategy<Value = (Patch, Patch)> {
                prop_oneof![
                    disconnected(),
                    adjacent(),
                    overlapped(),
                    covered().prop_perturb(randomize_order)
                ]
            }

            pub(super) fn any_connected_pair() -> impl Strategy<Value = (Patch, Patch)> {
                prop_oneof![
                    adjacent(),
                    overlapped(),
                    covered().prop_perturb(randomize_order)
                ]
            }

            pub(super) fn adjacent() -> impl Strategy<Value = (Patch, Patch)> {
                (any_patch(), any_patch()).prop_map(|(mut a, b)| {
                    a.set_addr(b.end());
                    (a, b)
                })
            }

            pub(super) fn disconnected() -> impl Strategy<Value = (Patch, Patch)> {
                (any_patch(), any_patch())
                    .prop_filter("Patches should be disconnected", |(p1, p2)| {
                        !p1.overlaps(&p2) && !p1.adjacent(&p2)
                    })
            }

            pub(super) fn overlapped() -> impl Strategy<Value = (Patch, Patch)> {
                (any_patch_with_len(2..5), any_patch_with_len(2..5))
                    .prop_perturb(|(a, mut b), mut rng| {
                        // Position B at the end of A so that they are partially overlaps
                        let offset = a.len().min(b.len());
                        let offset = rng.gen_range(1..offset) as u32;
                        b.set_addr(a.end() - offset);
                        (a, b)
                    })
                    .prop_perturb(randomize_order)
            }

            /// The first returned patch is covering and the second is covered
            pub(super) fn covered() -> impl Strategy<Value = (Patch, Patch)> {
                (any_patch(), any_patch())
                    .prop_map(|(a, b)| {
                        // reordering so that A is always longer or the same length as B
                        if a.len() > b.len() {
                            (a, b)
                        } else {
                            (b, a)
                        }
                    })
                    .prop_flat_map(|(a, b)| {
                        // generating offset of B into A, so that B is randomized, but A is still fully covering B
                        let offset = a.len() - b.len() + 1;
                        (Just(a), Just(b), 0..offset)
                    })
                    .prop_map(|(a, mut b, b_offset)| {
                        // placing B so that A covers it
                        b.set_addr(a.addr() + b_offset as u32);
                        (a, b)
                    })
            }

            pub(super) fn any_patch_with_len(len: Range<usize>) -> impl Strategy<Value = Patch> {
                prop_oneof![write_patch(len.clone()), reclaim_patch(len)]
            }

            pub(super) fn any_patch() -> impl Strategy<Value = Patch> {
                prop_oneof![write_patch(1..5), reclaim_patch(1..5)]
            }

            pub(super) fn write_patch(len: Range<usize>) -> impl Strategy<Value = Patch> {
                (0..DB_SIZE, vec(any::<u8>(), len))
                    .prop_filter("out of bounds patch", |(offset, bytes)| {
                        offset + bytes.len() < DB_SIZE
                    })
                    .prop_map(|(offset, bytes)| Patch::Write(offset as u32, bytes))
            }

            pub(super) fn reclaim_patch(len: Range<usize>) -> impl Strategy<Value = Patch> {
                (0..DB_SIZE, len)
                    .prop_filter("out of bounds patch", |(offset, len)| {
                        (*offset + *len) < DB_SIZE
                    })
                    .prop_map(|(offset, len)| Patch::Reclaim(offset as u32, len))
            }

            fn randomize_order((mut a, mut b): (Patch, Patch), mut rng: TestRng) -> (Patch, Patch) {
                if rng.gen_bool(0.5) {
                    mem::swap(&mut a, &mut b);
                }
                (a, b)
            }
        }

        fn any_snapshot() -> impl Strategy<Value = Vec<Patch>> {
            vec(patches::any_patch(), 1..10)
        }

        #[track_caller]
        fn assert_buffers_eq(a: &[u8], b: &[u8]) {
            assert_eq!(a.len(), b.len(), "Buffers should have the same length");

            let mut mismatch = a
                .iter()
                .zip(b.into_iter())
                .enumerate()
                .skip_while(|(_, (a, b))| *a == *b)
                .take_while(|(_, (a, b))| *a != *b)
                .map(|(idx, _)| idx);

            if let Some(start) = mismatch.next() {
                let end = mismatch.last().unwrap_or(start) + 1;
                assert_eq!(
                    &a[start..end],
                    &b[start..end],
                    "Mismatch detected at {}..{}",
                    start,
                    end
                );
            }
        }
    }

    #[track_caller]
    fn assert_str_eq<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) {
        let a = String::from_utf8_lossy(a.as_ref());
        let b = String::from_utf8_lossy(b.as_ref());
        assert_eq!(a, b);
    }

    impl From<&[u8]> for CommitedSnapshot {
        fn from(bytes: &[u8]) -> Self {
            assert!(bytes.len() <= PAGE_SIZE, "String is too large");

            CommitedSnapshot {
                patches: vec![Patch::Write(0, bytes.to_vec())],
                base: None,
                pages: 1,
                lsn: 1,
            }
        }
    }

    impl From<&str> for PagePool {
        fn from(value: &str) -> Self {
            let page = Rc::new(CommitedSnapshot::from(value.as_bytes()));
            PagePool { latest: page }
        }
    }

    impl From<&[u8]> for PagePool {
        fn from(value: &[u8]) -> Self {
            let page = Rc::new(CommitedSnapshot::from(value));
            PagePool { latest: page }
        }
    }
}
