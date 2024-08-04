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
//!   interface for interacting with the page memory.
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
//!   any reference to its data remains valid and unchanged until corresponding `Rc` reference is held.
//! - **Linear Snapshot History**: The module enforces a linear history of snapshots, preventing branches in the
//!   snapshot history and ensuring consistency of changes proposed in snapshots.
//!
//! ## Performance Considerations
//!
//! Since snapshots do not require duplicating the entire state of the page pool, they can be created with minimal
//! overhead, making it perfectly valid and cost-effective to create a snapshot even when the intention is only to
//! read data without any modifications.

use std::{borrow::Cow, ops::Range, sync::Arc};

pub const PAGE_SIZE: usize = 1 << 16; // 64Kib
pub type Addr = u32;
pub type PageOffset = u32;
pub type PageNo = u32;
pub type LSN = u64;

/// Represents a modification recorded in a snapshot.
///
/// A `Patch` can either write a range of bytes starting at a specified offset
/// or reclaim a range of bytes starting at a specified offset.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq, Debug))]
enum Patch {
    // Write given data at a specified offset
    Write(PageOffset, Vec<u8>),

    /// Reclaim given amount of bytes at a specified address
    Reclaim(PageOffset, usize),
}

impl Patch {
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
                *data = data.split_off((at - *offset) as usize);
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
        debug_assert!(self.overlaps(&other));

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
            (a @ Reclaim(..), b @ Reclaim(..)) => {
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
        debug_assert!(self.adjacent(&other));

        match self.reorder(other) {
            (Write(addr_a, mut a), Write(addr_b, b)) => {
                if a.len() + b.len() < 1024 {
                    a.extend(b);
                    Merged(Write(addr_a, a))
                } else {
                    Reordered(Write(addr_a, a), Write(addr_b, b))
                }
            }
            (Reclaim(addr, a), Reclaim(_, b)) => Merged(Reclaim(addr, a + b)),
            (a, b) => Reordered(a, b),
        }
    }

    fn normalize_covered(self, other: Patch) -> NormalizedPatches {
        use {NormalizedPatches::*, Patch::*};

        fn reordered_or_split(left: Patch, center: Patch, right: Patch) -> NormalizedPatches {
            assert!(left.len() != 0 || right.len() != 0);
            if left.len() == 0 {
                Reordered(center, right)
            } else if right.len() == 0 {
                Reordered(left, center)
            } else {
                Split(left, center, right)
            }
        }

        debug_assert!(self.fully_covers(&other));

        match (self, other) {
            (Write(a_addr, mut bytes), b @ Reclaim(..)) => {
                let split_idx = (b.addr() - a_addr) as usize + b.len();
                let right = Write(b.end(), bytes.split_off(split_idx));

                bytes.truncate(bytes.len() - b.len());
                let left = Write(a_addr, bytes);

                reordered_or_split(left, b, right)
            }
            (a @ Reclaim(..), b @ Write(..)) => {
                let left = Reclaim(a.addr(), (b.addr() - a.addr()) as usize);
                let right = Reclaim(b.end(), (a.end() - b.end()) as usize);

                reordered_or_split(left, b, right)
            }
            (Write(a_addr, mut a_data), Write(b_addr, b_data)) => {
                let start = (b_addr - a_addr) as usize;
                let end = start + b_data.len();
                a_data[start..end].copy_from_slice(&b_data);

                Merged(Write(a_addr, a_data))
            }
            (a @ Reclaim(..), Reclaim(..)) => Merged(a),
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

trait MemRange {
    fn addr(&self) -> u32;
    fn len(&self) -> usize;

    fn end(&self) -> u32 {
        self.addr() + self.len() as u32
    }

    fn to_range(&self) -> Range<u32> {
        self.addr()..self.addr() + self.len() as u32
    }
}

impl MemRange for Patch {
    fn addr(&self) -> u32 {
        match self {
            Patch::Write(offset, _) => *offset,
            Patch::Reclaim(offset, _) => *offset,
        }
    }

    fn len(&self) -> usize {
        match self {
            Patch::Write(_, bytes) => bytes.len(),
            Patch::Reclaim(_, len) => *len,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum NormalizedPatches {
    Merged(Patch),
    Reordered(Patch, Patch),
    Split(Patch, Patch, Patch),
}

impl NormalizedPatches {
    #[cfg(test)]
    fn len(self) -> usize {
        match self {
            Self::Merged(p1) => p1.len(),
            Self::Reordered(p1, p2) => p1.len() + p2.len(),
            Self::Split(p1, p2, p3) => p1.len() + p2.len() + p3.len(),
        }
    }

    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<&Patch> {
        match self {
            Self::Merged(ref a) => vec![a],
            Self::Reordered(ref a, ref b) => vec![a, b],
            Self::Split(ref a, ref b, ref c) => vec![a, b, c],
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
    latest: Arc<CommittedSnapshot>,
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
    ///   the range of valid addresses that can be written to in snapshots derived from this pool.
    pub fn new(pages: usize) -> Self {
        let snapshot = CommittedSnapshot {
            patches: vec![],
            base: None,
            pages: u32::try_from(pages).unwrap(),
            lsn: 1,
        };
        Self {
            latest: Arc::new(snapshot),
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
            base: Arc::clone(&self.latest),
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
            Arc::ptr_eq(&self.latest, &snapshot.base),
            "Proposed snapshot is not linear"
        );
        let lsn = self.latest.lsn + 1;

        if self.latest.patches.len() + snapshot.patches.len() < 100 {
            // merging consecutive snapshots because they are small
            let mut patch_clone = CommittedSnapshot::default();
            patch_clone.clone_from(&self.latest);
            patch_clone.lsn = lsn;
            let patches = snapshot.patches;
            for patch in patches {
                push_patch(&mut patch_clone.patches, patch);
            }
            self.latest = Arc::new(patch_clone);
        } else {
            self.latest = Arc::new(CommittedSnapshot {
                patches: snapshot.patches,
                base: Some(Arc::clone(&self.latest)),
                pages: snapshot.pages,
                lsn,
            });
        }
    }

    #[cfg(test)]
    fn read(&self, addr: PageOffset, len: usize) -> Cow<[u8]> {
        let mut buffer = vec![0; len];
        let mut buf_ranges = vec![0..len];
        self.latest.read_to_buf(addr, &mut buffer, &mut buf_ranges);
        Cow::Owned(buffer)
    }
}

/// Represents a committed snapshot of a page pool.
///
/// A `CommittedSnapshot` captures the state of a page pool at a specific point in time,
/// including any patches (modifications) that have been applied up to that point. It serves
/// as a read-only view into the historical state of the pool, allowing for consistent reads
/// of pages as they existed at the time of the snapshot.
///
/// Each `CommittedSnapshot` can optionally reference a base snapshot, forming a chain
/// that represents the full history of modifications leading up to the current state.
/// This chain is traversed backwards when reading from a snapshot to reconstruct the state
/// of a page by applying patches in reverse chronological order.
#[derive(Clone)]
pub struct CommittedSnapshot {
    /// A patches that have been applied in this snapshot.
    patches: Vec<Patch>,

    /// A reference to the base snapshot from which this snapshot was derived.
    /// If present, the base snapshot represents the state of the page pool
    /// immediately before the current snapshot's patches were applied.
    base: Option<Arc<CommittedSnapshot>>,

    /// The total number of pages represented by this snapshot. This is used to
    /// validate read requests and ensure they do not exceed the bounds of the snapshot.
    pages: PageNo,

    /// A log sequence number (LSN) that uniquely identifies this snapshot. The LSN
    /// is used internally to ensure that snapshots are applied in a linear and consistent order.
    lsn: LSN,
}

impl Default for CommittedSnapshot {
    fn default() -> Self {
        Self {
            patches: vec![],
            base: None,
            pages: 1,
            lsn: 1,
        }
    }
}

impl CommittedSnapshot {
    fn read_to_buf(self: &Arc<Self>, addr: Addr, buf: &mut [u8], buf_mask: &mut Vec<Range<usize>>) {
        split_ptr_checked(addr, buf.len(), self.pages);

        let mut snapshot = Some(Arc::clone(self));
        while !buf_mask.is_empty() && snapshot.is_some() {
            let s = snapshot.take().unwrap();

            apply_patches(&s.patches, addr as usize, buf, buf_mask);
            snapshot.clone_from(&s.base);
        }
    }
}

impl Drop for CommittedSnapshot {
    /// Custom drop logic is necessary here to prevent a stack overflow that could
    /// occur due to recursive drop calls on a long chain of `Rc` references to base snapshot.
    /// Each `Rc` decrement could potentially trigger the drop of another `Rc` in the chain,
    /// leading to deep recursion.
    ///
    /// By explicitly unwrapping and handling the inner `Rc` references, we ensure that the drop sequence
    /// is performed without any recursion
    fn drop(&mut self) {
        let mut next_base = self.base.take();
        while let Some(base) = next_base {
            next_base = Arc::try_unwrap(base)
                .map(|mut base| base.base.take())
                .unwrap_or(None);
        }
    }
}

#[derive(Clone)]
pub struct Snapshot {
    patches: Vec<Patch>,
    base: Arc<CommittedSnapshot>,
    pages: PageNo,
}

impl Snapshot {
    pub fn write(&mut self, addr: Addr, bytes: impl Into<Vec<u8>>) {
        let bytes = bytes.into();
        split_ptr_checked(addr, bytes.len(), self.pages);

        if !bytes.is_empty() {
            push_patch(&mut self.patches, Patch::Write(addr, bytes))
        }
    }

    /// Frees the given segment of memory.
    ///
    /// Reclaim is guaranteed to follow zeroing semantics. The read operation from the corresponding segment
    /// after reclaiming will return zeros.
    ///
    /// The main purpose of reclaim is to mark memory as not used, so it can be freed from persistent storage
    /// after snapshot compaction.
    pub fn reclaim(&mut self, addr: Addr, len: usize) {
        if len > 0 {
            push_patch(&mut self.patches, Patch::Reclaim(addr, len))
        }
    }

    pub fn read(&self, addr: Addr, len: usize) -> Cow<[u8]> {
        split_ptr_checked(addr, len, self.pages);

        let mut buf = vec![0; len];
        #[allow(clippy::single_range_in_vec_init)]
        let mut buf_mask = vec![0..len];

        // Apply uncommitted changes
        apply_patches(&self.patches, addr as usize, &mut buf, &mut buf_mask);

        // Apply committed changes
        self.base.read_to_buf(addr, &mut buf, &mut buf_mask);

        Cow::Owned(buf)
    }

    pub fn valid_range(&self, addr: Addr, len: usize) -> bool {
        is_valid_ptr(addr, len, self.pages)
    }

    #[cfg(test)]
    fn resize(&mut self, pages: usize) {
        self.pages = pages as PageNo;
    }
}

/// Pushes a patch to a snapshot ensuring the following invariants hold:
///
/// 1. All patches are sorted by [Patch:addr()]
/// 2. All patches are non-overlapping
fn push_patch(patches: &mut Vec<Patch>, patch: Patch) {
    assert!(patch.len() > 0, "Patch should not be empty");
    let connected = find_connected_ranges(&patches, &patch);

    if connected.is_empty() {
        // inserting new patch preserving order
        let insert_idx = patches
            .binary_search_by(|i| i.addr().cmp(&patch.addr()))
            // because new patch is not connected to anything we must get Err() here
            .unwrap_err();
        patches.insert(insert_idx, patch);
    } else {
        // Splitting all the patches on left and right disconnected parts and a middle part
        // that contains connected patches that need to be normalized. We keep left part in place,
        // there is no need to move it anywhere
        let mut middle = patches.split_off(connected.start);
        let right = middle.split_off(connected.len());

        // Normalizing all connected patches one by one with the new patch.
        // After normalizing any particular pair we need to keep continue normalizing
        // "tail" patch with the rest of connected patches
        let mut tail_patch = patch;
        for p in middle {
            tail_patch = match p.normalize(tail_patch) {
                NormalizedPatches::Merged(a) => a,
                NormalizedPatches::Reordered(a, b) => {
                    patches.push(a);
                    b
                }
                NormalizedPatches::Split(a, b, c) => {
                    patches.extend([a, b]);
                    c
                }
            }
        }
        patches.push(tail_patch);
        patches.extend(right);
    }

    debug_assert_sorted_and_has_no_overlaps(&patches);
}

/// Applies a list of patches to a given buffer within a specified range.
///
/// This function iterates over the provided snapshots and applies each patch from each snapshot
/// that intersects with the specified range to the given buffer.
///
/// # Arguments
///
/// * `snapshot` - A slice of `Patch` instances.
/// * `addr` - An address where to read data from.
/// * `buf` - A mutable reference to the buffer where the patches will be applied.
/// * `buf_mask` - A mutable vector of ranges within the buffer where the patches need to be applied.
///
/// This functions will find and apply corresponding patches to the buffer using `addr` as a base address
/// for a buffer. It will also update `buf_mask` vector with ranges of buffer that still need to be patched.
/// Therefore it is required to start with a full buffer range and then this function will update it with
/// ranges that still need to be patched.
fn apply_patches(
    patches: &[Patch],
    addr: usize,
    buf: &mut [u8],
    buf_masks: &mut Vec<Range<usize>>,
) {
    debug_assert_sorted_and_has_no_overlaps(patches);

    let masks_cnt = buf_masks.len();
    let mut mask_idx = 0;
    // When iterating over mask regions we will push new masks that can not be processed by current snapshot
    // to the same Vec. We need to keep track of the number of masks at the start to avoid
    // processing the same mask multiple times.
    while mask_idx < buf_masks.len().min(masks_cnt) {
        let start_addr = addr + buf_masks[mask_idx].start;
        let end_addr = start_addr + buf_masks[mask_idx].len();
        let first_matching_idx = patches
            // because end address is exclusive we need to subtract 1 from it to find idx of first possibly
            // intersecting patch
            .binary_search_by(|i| (i.end() - 1).cmp(&(start_addr as u32)))
            .unwrap_or_else(|idx| idx);

        let overlapping_patches = patches[first_matching_idx..]
            .iter()
            .take_while(|p| intersects(p, &(start_addr..end_addr)));
        for patch in overlapping_patches {
            // Calculating intersection of the path and input interval
            let patch_addr = patch.addr() as usize;

            let start_addr = start_addr.max(patch_addr);
            let end_addr = end_addr.min(patch_addr + patch.len());
            let len = end_addr - start_addr;

            let buf_range = {
                let from = start_addr.saturating_sub(addr);
                from..from + len
            };

            let patch_range = {
                let from = start_addr.saturating_sub(patch_addr);
                from..from + len
            };
            debug_assert!(patch_range.len() == buf_range.len());

            if buf_range.start > buf_masks[mask_idx].start {
                // patch starts after the current mask, we need to split the mask
                // and add the missing range to the list to process later by a different snapshot
                buf_masks.push(buf_masks[mask_idx].start..buf_range.start);
            }

            // Shrinking current mask according to the bytes processed by the patch
            buf_masks[mask_idx].start = buf_range.end;

            match patch {
                Patch::Write(_, bytes) => buf[buf_range].copy_from_slice(&bytes[patch_range]),
                Patch::Reclaim(..) => buf[buf_range].fill(0),
            }
        }
        if buf_masks[mask_idx].is_empty() {
            // current mask was fully processed, we can remove it, but we don't need to increment mask_idx.
            // Because of swap_remove the next mask will be at the same index
            buf_masks.remove(mask_idx);
        } else {
            // current mask was processed only partially, leaving it in place and moving to the next one
            mask_idx += 1;
        }
    }
}

/// Finds the range of indices in a sorted list of ranges that intersect or adgacent with a given range.
///
/// This function assumes that the input list of ranges is sorted and non-overlapping. It uses binary search
/// to efficiently find the starting and ending indices of the intersecting ranges.
///
/// # Arguments
///
/// * `range` - A reference to the target range to find intersections with.
/// * `ranges` - A slice of sorted ranges to search within.
///
/// # Returns
///
/// A `Range` of indices indicating the start and end of the intersecting ranges within the input slice.
///
/// # Panics
///
/// This function will panic in debug mode if the input list of ranges is not sorted or contains overlapping ranges
fn find_connected_ranges<T: MemRange>(ranges: &[T], range: &T) -> Range<usize> {
    debug_assert_sorted_and_has_no_overlaps(ranges);

    let start_idx = ranges
        .binary_search_by(|i| i.end().cmp(&range.addr()))
        .unwrap_or_else(|idx| idx);
    let end_idx = ranges
        .binary_search_by(|i| i.addr().cmp(&range.end()))
        // because the Range::end is exclusive we need to increment index by 1 on exact match
        .map(|i| i + 1)
        .unwrap_or_else(|idx| idx);
    start_idx..end_idx
}

fn debug_assert_sorted_and_has_no_overlaps<T: MemRange>(ranges: &[T]) {
    debug_assert!(
        ranges.windows(2).all(|p| p[0].end() <= p[1].addr()),
        "Patches should be ordered and not overlapping"
    )
}

fn split_ptr_checked(addr: Addr, len: usize, pages: u32) -> (PageNo, PageOffset) {
    assert!(
        is_valid_ptr(addr, len, pages),
        "Ptr address is out of bounds"
    );

    split_ptr(addr)
}

fn is_valid_ptr(addr: Addr, len: usize, pages: u32) -> bool {
    let (page_no, _) = split_ptr(addr);
    // we need to subtract 1 byte, because end address exclusive
    let (end_page_no, _) = split_ptr(addr + (len as Addr) - 1);

    page_no < pages && end_page_no < pages
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
    use std::{panic, thread};

    #[test]
    fn committed_changes_should_be_visible_on_a_page() {
        let mut mem = PagePool::from("Jekyll");

        let mut snapshot = mem.snapshot();
        snapshot.write(0, b"Hide");
        mem.commit(snapshot);

        assert_str_eq(mem.read(0, 4), b"Hide");
    }

    #[test]
    fn uncommitted_changes_should_be_visible_only_on_the_snapshot() {
        let mem = PagePool::from("Jekyll");

        let mut snapshot = mem.snapshot();
        snapshot.write(0, b"Hide");

        assert_str_eq(snapshot.read(0, 4), "Hide");
        assert_str_eq(mem.read(0, 6), "Jekyll");
    }

    #[test]
    fn patch_page() {
        let mut mem = PagePool::from("Hello panic!");

        let mut snapshot = mem.snapshot();
        snapshot.write(6, b"world");
        mem.commit(snapshot);

        assert_str_eq(mem.read(0, 12), "Hello world!");
        assert_str_eq(mem.read(0, 8), "Hello wo");
        assert_str_eq(mem.read(3, 9), "lo world!");
        assert_str_eq(mem.read(6, 5), "world");
        assert_str_eq(mem.read(8, 4), "rld!");
        assert_str_eq(mem.read(7, 3), "orl");
    }

    #[test]
    fn test_regression() {
        let mut mem = PagePool::default();

        let mut snapshot = mem.snapshot();
        snapshot.write(70, &[0, 0, 1]);
        mem.commit(snapshot);
        let mut snapshot = mem.snapshot();
        snapshot.reclaim(71, 2);
        snapshot.write(73, &[0]);
        mem.commit(snapshot);

        assert_eq!(mem.read(70, 4).as_ref(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_regression2() {
        let mut mem = PagePool::default();

        let mut snapshot = mem.snapshot();
        snapshot.write(70, &[0, 0, 1]);
        mem.commit(snapshot);
        let mut snapshot = mem.snapshot();
        snapshot.write(0, &[0]);
        mem.commit(snapshot);
        let mut snapshot = mem.snapshot();
        snapshot.reclaim(71, 2);
        mem.commit(snapshot);

        assert_eq!(mem.read(70, 4).as_ref(), &[0, 0, 0, 0]);
    }

    #[test]
    fn data_across_multiple_pages_can_be_written() {
        let mut mem = PagePool::new(3);

        let alice = b"Alice";
        let bob = b"Bob";
        let charlie = b"Charlie";

        // Addresses on 3 consequent pages
        let page_a = 0;
        let page_b = PAGE_SIZE as Addr;
        let page_c = 2 * PAGE_SIZE as Addr;

        let mut snapshot = mem.snapshot();
        snapshot.write(page_a, alice);
        snapshot.write(page_b, bob);
        snapshot.write(page_c, charlie);

        // Checking that data is visible in snapshot
        assert_str_eq(snapshot.read(page_a, alice.len()), alice);
        assert_str_eq(snapshot.read(page_b, bob.len()), bob);
        assert_str_eq(snapshot.read(page_c, charlie.len()), charlie);

        mem.commit(snapshot);

        // Checking that data is visible after commit to page pool
        assert_str_eq(mem.read(page_a, alice.len()), alice);
        assert_str_eq(mem.read(page_b, bob.len()), bob);
        assert_str_eq(mem.read(page_c, charlie.len()), charlie);
    }

    #[test]
    fn data_can_be_removed_on_snapshot() {
        let data = [1, 2, 3, 4, 5];
        let mem = PagePool::from(data.as_slice());

        let mut snapshot = mem.snapshot();
        snapshot.reclaim(1, 3);

        assert_eq!(snapshot.read(0, 5), [1u8, 0, 0, 0, 5].as_slice());
    }

    #[test]
    #[should_panic(expected = "address is out of bounds")]
    fn page_pool_should_return_error_on_oob_read() {
        let mem = PagePool::default();
        let snapshot = mem.snapshot();

        // reading in a such a way that start address is still valid, but end address is not
        let start = PAGE_SIZE as u32 - 10;
        let len = 20;
        let _ = snapshot.read(start, len);
    }

    #[test]
    #[should_panic(expected = "address is out of bounds")]
    fn page_pool_should_return_error_on_oob_write() {
        let mem = PagePool::default();

        let mut snapshot = mem.snapshot();
        let bytes = [0; 20];
        let addr = PAGE_SIZE as u32 - 10;
        snapshot.write(addr, bytes);
    }

    #[test]
    fn change_number_of_pages_on_commit() {
        let mut mem = PagePool::new(1); // Initially 1 page

        let page_a = 0;
        let page_b = PAGE_SIZE as Addr;

        let alice = b"Alice";
        let bob = b"Bob";

        let mut snapshot = mem.snapshot();
        snapshot.write(page_a, alice);
        assert!(!snapshot.valid_range(page_b, bob.len()));
        snapshot.resize(2); // adding second page
        snapshot.write(page_b, bob);
        mem.commit(snapshot);
    }

    /// When dropping PagePool all related snapshots will be removed. It may lead
    /// to stackoverflow if snapshots removed recursively.
    #[test]
    fn deep_snapshot_should_not_cause_stack_overflow() {
        thread::Builder::new()
            .name("deep_snapshot_should_not_cause_stack_overflow".to_string())
            // setting stacksize explicitly so not to rely on the running environment
            .stack_size(1024)
            .spawn(|| {
                let mut mem = PagePool::new(1);
                for _ in 0..1000 {
                    mem.commit(mem.snapshot())
                }
            })
            .unwrap()
            .join()
            .unwrap();
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
            assert_split(
                Write(0, vec![0, 1, 2]),
                Reclaim(1, 1),
                (Write(0, vec![0]), Reclaim(1, 1), Write(2, vec![2])),
            );

            assert_split(
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
                result => panic!("Patch should be merged: {:?}", result),
            }
        }

        fn assert_reordered(a: Patch, b: Patch, expected: (Patch, Patch)) {
            match a.normalize(b) {
                NormalizedPatches::Reordered(p1, p2) => assert_eq!((p1, p2), expected),
                result => panic!("Patch should be merged: {:?}", result),
            }
        }

        fn assert_split(a: Patch, b: Patch, expected: (Patch, Patch, Patch)) {
            match a.normalize(b) {
                NormalizedPatches::Split(p1, p2, p3) => assert_eq!((p1, p2, p3), expected),
                result => panic!("Patch should be split: {:?}", result),
            }
        }

        fn assert_not_merged(a: Patch, b: Patch) {
            match a.clone().normalize(b.clone()) {
                NormalizedPatches::Reordered(p1, p2) => {
                    assert_eq!(p1, a);
                    assert_eq!(p2, b);
                }
                result => panic!("Patch should not be merged: {:?}", result),
            }
        }
    }

    mod ptr {
        use super::*;

        #[test]
        fn split_ptr_generic() {
            let addr = 0x0AF0_1234;
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0x0AF0);
            assert_eq!(offset, 0x1234);
        }

        #[test]
        fn split_ptr_at_boundary() {
            let addr = PAGE_SIZE as u32 - 1; // Last address of the first page
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0,);
            assert_eq!(offset, PAGE_SIZE as u32 - 1,);
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
            let addr = Addr::MAX;
            let (page_no, offset) = split_ptr(addr);

            assert_eq!(page_no, 0xFFFF,);
            assert_eq!(offset, 0xFFFF);
        }
    }

    #[cfg(not(miri))]
    mod proptests {
        use super::*;
        use proptest::{collection::vec, prelude::*};
        use NormalizedPatches::*;

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
                                shadow_buffer[range].copy_from_slice(bytes.as_slice());
                                snapshot.write(offset, bytes);
                            },
                            Patch::Reclaim(offset, len) => {
                                shadow_buffer[range].fill(0);
                                snapshot.reclaim(offset, len);

                            }
                        }
                    }
                    mem.commit(snapshot);
                }

                assert_buffers_eq(&mem.read(0, DB_SIZE), shadow_buffer.as_slice())?;
            }

            #[test]
            fn any_patches_length_should_be_positive((a, b) in patches::any_pair()) {
                for patch in a.normalize(b).to_vec() {
                    prop_assert!(patch.len() > 0, "{:?}", patch)
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
                match covered.normalize(covering.clone()) {
                    Merged(result) => prop_assert_eq!(covering, result),
                    r => prop_assert!(false, "Merged() expected. Got: {:?}", r),
                }
            }

            #[test]
            fn covered_patches_adjacency((covering, covered) in patches::adjacent()) {
                match covering.normalize(covered.clone()) {
                    Merged(_) => {}
                    Reordered(p1, p2) =>
                        prop_assert!(p1.adjacent(&p2), "Patches must be adjacent: {:?}, {:?}", p1, p2),
                    r @ Split(..) => prop_assert!(false, "Merged()/Reordered() expected. Got: {:?}", r)
                }
            }

            #[test]
            fn connected_patches_are_adjacent_after_normalization((a, b) in patches::any_connected_pair()) {
                for p in a.normalize(b).to_vec().windows(2) {
                    prop_assert!(p[0].adjacent(p[1]), "Patches must be adjacent: {:?}, {:?}", p[0],p[1]);
                }
            }

            #[test]
            fn patches_are_ordered_after_normalization((a, b) in patches::any_pair()) {
                for p in a.normalize(b).to_vec().windows(2) {
                    prop_assert!(p[0].end() <= p[1].addr(), "Patches must be ordered: {:?}, {:?}", p[0], p[1]);
                }
            }

            #[test]
            fn patches_have_positive_length((a, b) in patches::any_pair()) {
                for patch in a.normalize(b).to_vec() {
                    prop_assert!(patch.len() > 0);
                }
            }

            #[test]
            fn normalized_patches_do_not_overlaps((a, b) in patches::any_pair()) {
                for p in a.normalize(b).to_vec().windows(2) {
                    prop_assert!(!p[0].overlaps(p[1]), "Patches are overlapping: {:?}, {:?}", p[0], p[1]);
                }
            }

            #[test]
            fn overlapping_patches((a, b) in patches::overlapped()) {
                let start = a.addr().min(b.addr());
                let end = a.end().max(b.end());
                let expected_len = (end - start) as usize;

                let patch = a.normalize(b);
                if let Split(..) = &patch {
                    panic!("Merged()/Reordered() extected")
                }
                prop_assert_eq!(patch.len(), expected_len);
            }

            #[test]
            fn ranges_intersect(ranges in any_non_intersecting_ranges(), range in any_range()) {
                let connected = find_connected_ranges(ranges.as_slice(), &range);

                // calculating left and right parts that are guaranteed to be disconnected from the target range
                let left = &ranges[..connected.start];
                let right = &ranges[connected.end..];

                for r in left.iter().chain(right.iter()) {
                    prop_assert!(!ranges_connected(r, &range), "Must not be connected: {:?}, {:?}", r, range)
                }

                for r in &ranges[connected] {
                    prop_assert!(ranges_connected(r, &range), "Must be connected: {:?}, {:?}", r, range)
                }
            }
        }

        fn any_range() -> impl Strategy<Value = Range<u32>> {
            (0u32..10, 1u32..5).prop_map(|(offset, length)| offset..(offset + length))
        }

        fn any_non_intersecting_ranges() -> impl Strategy<Value = Vec<Range<u32>>> {
            (0usize..10)
                .prop_flat_map(|size| (vec(1u32..3, size), vec(0u32..3, size)))
                .prop_map(|(size, skip)| {
                    size.into_iter()
                        .zip(skip)
                        .scan(0, |offset, (size, skip)| {
                            let start = *offset + skip;
                            let end = start + size;
                            *offset = end;
                            Some(start..end)
                        })
                        .collect()
                })
        }

        fn ranges_connected<T: PartialOrd + PartialEq>(a: &Range<T>, b: &Range<T>) -> bool {
            let overlapping = a.contains(&b.start) || b.contains(&a.start);
            let adjacent = a.start == b.end || b.start == a.end;
            overlapping || adjacent
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
                        !p1.overlaps(p2) && !p1.adjacent(p2)
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
        fn assert_buffers_eq(a: &[u8], b: &[u8]) -> std::result::Result<(), TestCaseError> {
            prop_assert_eq!(a.len(), b.len(), "Buffers should have the same length");

            let mut mismatch = a
                .iter()
                .zip(b)
                .enumerate()
                .skip_while(|(_, (a, b))| *a == *b)
                .take_while(|(_, (a, b))| *a != *b)
                .map(|(idx, _)| idx);

            if let Some(start) = mismatch.next() {
                let end = mismatch.last().unwrap_or(start) + 1;
                prop_assert_eq!(
                    &a[start..end],
                    &b[start..end],
                    "Mismatch detected at {}..{}",
                    start,
                    end
                );
            }
            Ok(())
        }
    }

    #[track_caller]
    fn assert_str_eq<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) {
        let a = String::from_utf8_lossy(a.as_ref());
        let b = String::from_utf8_lossy(b.as_ref());
        assert_eq!(a, b);
    }

    impl From<&[u8]> for CommittedSnapshot {
        fn from(bytes: &[u8]) -> Self {
            assert!(bytes.len() <= PAGE_SIZE, "String is too large");

            CommittedSnapshot {
                patches: vec![Patch::Write(0, bytes.to_vec())],
                base: None,
                pages: 1,
                lsn: 1,
            }
        }
    }

    impl From<&str> for PagePool {
        fn from(value: &str) -> Self {
            let page = Arc::new(CommittedSnapshot::from(value.as_bytes()));
            PagePool { latest: page }
        }
    }

    impl From<&[u8]> for PagePool {
        fn from(value: &[u8]) -> Self {
            let page = Arc::new(CommittedSnapshot::from(value));
            PagePool { latest: page }
        }
    }

    impl MemRange for Range<u32> {
        fn addr(&self) -> u32 {
            self.start
        }

        fn len(&self) -> usize {
            (self.end - self.start) as usize
        }
    }
}
