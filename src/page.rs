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
use std::ops::Deref;
use std::{borrow::Cow, ops::Range, rc::Rc, usize};

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
#[derive(Debug, Clone)]
enum Patch {
    // Write given data at a specified offset
    Write(PageOffset, Vec<u8>),

    /// Reclaim given amount of bytes at a specified address
    Reclaim(PageOffset, usize),
}

impl Patch {
    pub fn offset(&self) -> u32 {
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
    fn read(&self, addr: PageOffset, len: usize) -> Result<Ref> {
        let snapshot = Rc::clone(&self.latest);
        Ok(Ref::create(snapshot, addr, len))
    }
}

pub struct Ref {
    bytes: Cow<'static, [u8]>,
    _snapshot: Rc<CommitedSnapshot>,
}

impl Ref {
    /// The `Ref::create` function constructs a `Ref` instance by reading bytes from a `CommitedSnapshot`
    /// through a specified range and extending its lifetime to `'static`.
    ///
    /// This is inherently risky since it assumes that the byte slice returned by the `snapshot.read` function
    /// will remain valid for the 'static lifetime, which is not guaranteed by Rust's safety rules.
    /// However, this approach is sound under specific preconditions and with strict usage guidelines:
    ///
    /// # Safety
    ///
    /// 1. External modification or deallocation of `CommitedSnapshot`'s memory is prevented by using
    ///  reference-counted pointer. `Rc`, ensuring that it remains allocated as long as `Ref` holds
    /// a reference to it. This setup ties the lifetime of the snapshot's data indirectly to `Ref`'s usage.
    ///
    /// 2. The `CommitedSnapshot.patches` and consequently, the byte slices it returns, are immutable. This
    /// immutability guarantees that once a `Ref` is created, the data it references cannot change,
    /// thereby making the extended 'static lifetime of the bytes sound.
    ///
    /// 3. By dropping `bytes` before `_snapshot`, we ensure that the borrowed view into the snapshot's data does not
    /// outlive the snapshot itself. This guarantee relies on the Rust drop order of struct fields.
    fn create(snapshot: Rc<CommitedSnapshot>, offset: PageOffset, len: usize) -> Self {
        use std::mem;

        let bytes = snapshot.read(offset, len);
        let bytes = unsafe { mem::transmute(bytes) };
        Self {
            _snapshot: snapshot,
            bytes,
        }
    }
}

impl Deref for Ref {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.bytes.deref()
    }
}

impl AsRef<[u8]> for Ref {
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
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
        use crate::ensure;
        assert!(len <= PAGE_SIZE, "Out of bounds read");

        let (page_no, _) = split_ptr(addr);
        ensure!(page_no < self.pages, Error::OutOfBounds);

        let mut buffer = vec![0; len];
        let range = (addr as usize)..addr as usize + len;

        let mut patch_list = vec![self.patches.as_slice()];
        #[allow(clippy::useless_asref)]
        let snapshots = self
            .base
            .as_ref()
            .map(Rc::clone)
            .map(collect_snapshots)
            .unwrap_or_default();
        patch_list.extend(snapshots.iter().map(|s| s.patches.as_slice()));

        apply_patches(&patch_list, &mut buffer, &range);
        Ok(Cow::Owned(buffer))
    }

    pub fn read_ref(self: &Rc<Self>, addr: PageOffset, len: usize) -> Ref {
        Ref::create(Rc::clone(self), addr, len)
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
        ensure!(page_no < self.pages && len <= PAGE_SIZE, Error::OutOfBounds);
        let mut buffer = vec![0; len];
        let range = (addr as usize)..addr as usize + len;

        // We need to collect a chain of snapshots into a Vec first.
        // Otherwise borrowcher is unable to reason about lifecycles
        let snapshots = collect_snapshots(Rc::clone(&self.base));
        let mut patch_list = vec![self.patches.as_slice()];
        patch_list.extend(snapshots.iter().map(|s| s.patches.as_slice()));

        apply_patches(&patch_list, &mut buffer, &range);
        Ok(Cow::Owned(buffer))
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

fn apply_patches(snapshots: &[&[Patch]], buffer: &mut [u8], range: &Range<usize>) {
    for patches in snapshots.iter().rev() {
        for patch in patches.iter().filter(|p| intersects(p, range)) {
            // Calculating intersection of the path and input interval
            let offset = patch.offset() as usize;
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

fn split_ptr(addr: Addr) -> (PageNo, PageOffset) {
    const PAGE_SIZE_BITS: u32 = PAGE_SIZE.trailing_zeros();
    let page_no = addr >> PAGE_SIZE_BITS;
    let offset = addr & ((PAGE_SIZE - 1) as u32);
    (page_no, offset)
}

/// Returns true of given patch intersects given range of bytes
fn intersects(patch: &Patch, range: &Range<usize>) -> bool {
    let start = patch.offset() as usize;
    let end = start + patch.len();
    start < range.end && end > range.start
}

#[cfg(test)]
mod tests {
    pub use super::*;

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

    mod proptests {
        use super::*;
        use proptest::{collection::vec, prelude::*};

        proptest! {
            #[test]
            #[cfg_attr(miri, ignore)]
            fn arbitrary_writes(snapshots in vec(any_snapshot(), 0..5)) {
                // Mirror buffer where we track all the patches being applied.
                // In the end content should be equal to the mirror buffer
                let mut mirror = vec![0; PAGE_SIZE];
                let mut mem = PagePool::default();

                for patches in snapshots {
                    for patch in patches.into_iter() {
                        match patch {
                            Patch::Write(offset, bytes) => {
                                let mut snapshot = mem.snapshot();
                                snapshot.write(offset, &bytes);

                                let offset = offset as usize;
                                let range = offset..offset + bytes.len();
                                mirror[range].copy_from_slice(bytes.as_slice());
                                mem.commit(snapshot);
                            },
                            Patch::Reclaim(_, _) => {
                                unimplemented!()
                            }
                        }

                    }
                }

                assert_eq!(&*mem.read(0, PAGE_SIZE).unwrap(), mirror);
            }
        }

        fn any_patch() -> impl Strategy<Value = Patch> {
            (0u32..(PAGE_SIZE as u32), vec(any::<u8>(), 1..32))
                .prop_filter("out of bounds patch", |(offset, bytes)| {
                    offset + (bytes.len() as u32) < PAGE_SIZE as u32
                })
                .prop_map(|(offset, bytes)| Patch::Write(offset, bytes))
        }

        fn any_snapshot() -> impl Strategy<Value = Vec<Patch>> {
            vec(any_patch(), 1..10)
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
