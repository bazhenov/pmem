//! # Page Management Module
//!
//! This module provides a system for managing and manipulating snapshots of a memory. It is designed
//! to facilitate operations on persistent memory (pmem), allowing for efficient snapshots, modifications, and commits
//! of data changes. The core functionality revolves around the [`PagePool`] structure, which manages a pool of pages,
//! and the [`Transaction`] structure, which represents a modifiable session of the page pool's state at a given point
//! in time.
//!
//! ## Concepts
//!
//! - [`PagePool`]: A collection of pages that can be snapshoted, modified, and committed. It acts as the primary
//!   interface for interacting with the page memory.
//! - [`Transaction`]: A read/write session allowing modifications of the page pool.
//!   It can be modified independently of the pool, and later committed back to the pool to update its state.
//! - [`Snapshot`]: A frozen state of the page pool at some point in time. It is immutable and only can be
//!   used to read data.
//! - [`Patch`]: A modification recorded in a transaction. It consists of the address where the modification starts and
//!   the bytes that were written.
//! - [`PagePoolHandle`]: A read-only view of the page pool that can be used to read data from the pool and
//!   wait for new snapshots to be committed. It's useful for concurrent access to the pool.
//!
//! ## Usage
//!
//! The module is designed to be used as follows:
//!
//! 1. **Initialization**: Create a new [`PagePool`].
//! 2. **Transactioning**: Create a [`Transaction`] of the current state.
//! 3. **Modification**: Use the [`Transaction`] to perform modifications. Each modification is recorded as a patch.
//! 4. **Commit**: Commit the snapshot back to the [`PagePool`], applying all the patches and updating
//!    the pool's state.
//! 5. **Concurrent Access**: Optionally, create a [`PagePoolHandle`] for read-only access to the pool
//!    from other threads.
//!
//! ## Example
//!
//! ```rust
//! use pmem::page::{PagePool, TxRead, TxWrite};
//!
//! let mut pool = PagePool::new(5);    // Initialize a pool with 5 pages
//!
//! // Create a handle for concurrent read-only access
//! let mut handle = pool.handle();
//!
//! let mut tx = pool.start();    // Create a new transaction
//! tx.write(0, &[1, 2, 3, 4]);   // Write 4 bytes at offset 0
//! pool.commit(tx);              // Commit the changes back to the pool
//!
//! let snapshot = handle.wait_for_commit();
//! assert_eq!(snapshot.read(0, 4), vec![1, 2, 3, 4]); // Read using TxRead trait
//! ```
//!
//! The [`TxRead`] and [`TxWrite`] traits provide methods for reading from and writing to snapshots, respectively.
//!
//! ## Performance Considerations
//!
//! Since snapshots do not require duplicating the entire state of the page pool, they can be created with minimal
//! overhead, making it perfectly valid and cost-effective to create a snapshot even when the intention is only to
//! read data without any modifications.

use arc_swap::ArcSwap;
use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter},
    ops::Range,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

pub const PAGE_SIZE_BITS: usize = 16;
pub const PAGE_SIZE: usize = 1 << PAGE_SIZE_BITS; // 64Kib
pub type Addr = u64;
pub type PageOffset = u32;
pub type PageNo = u32;
pub type LSN = u64;

/// Represents a modification recorded in a snapshot.
///
/// A `Patch` can either write a range of bytes starting at a specified offset
/// or reclaim a range of bytes starting at a specified offset.
#[derive(Clone, PartialEq, Debug)]
pub enum Patch {
    // Write given data at a specified offset
    Write(Addr, Vec<u8>),

    /// Reclaim given amount of bytes at a specified address
    Reclaim(Addr, usize),
}

impl Patch {
    fn normalize(self, other: Patch) -> NormalizedPatches {
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

    fn trim_after(&mut self, at: Addr) {
        let end = self.end();
        assert!(self.addr() < at && at < end, "Out-of-bounds");
        match self {
            Patch::Write(offset, data) => data.truncate((at - *offset) as usize),
            Patch::Reclaim(_, len) => {
                *len -= (end - at) as usize;
            }
        }
    }

    fn trim_before(&mut self, at: Addr) {
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

                Merged(Write(start as Addr, result_data))
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
                if a.len() + b.len() < PAGE_SIZE {
                    a.extend(b);
                    Merged(Write(addr_a, a))
                } else {
                    Reordered(Write(addr_a, a), Write(addr_b, b))
                }
            }
            (Reclaim(addr_a, a), Reclaim(_, b)) => Merged(Reclaim(addr_a, a + b)),
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
    fn set_addr(&mut self, value: Addr) {
        match self {
            Patch::Write(addr, _) => *addr = value,
            Patch::Reclaim(addr, _) => *addr = value,
        }
    }
}

fn are_on_the_same_page(a1: Addr, a2: Addr) -> bool {
    a1 / PAGE_SIZE as Addr == a2 / PAGE_SIZE as Addr
}

impl Display for Patch {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Patch::Write(addr, bytes) => write!(f, "Write({:#x}, {})", addr, bytes.len()),
            Patch::Reclaim(addr, len) => write!(f, "Reclaim({:#x}, {})", addr, len),
        }
    }
}

trait MemRange {
    fn addr(&self) -> Addr;
    fn len(&self) -> usize;

    fn end(&self) -> Addr {
        self.addr() + self.len() as Addr
    }

    fn to_range(&self) -> Range<Addr> {
        self.addr()..self.addr() + self.len() as Addr
    }
}

impl MemRange for Patch {
    fn addr(&self) -> Addr {
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

    #[error("No valid snapshot found for lsn: {0}. Latest LSN: {1}")]
    NoSnapshot(u64, u64),
}

struct Page {
    page_no: PageNo,
    data: Box<[u8; PAGE_SIZE]>,
}

impl Page {
    fn new(page_no: PageNo) -> Self {
        Self {
            data: Box::new([0; PAGE_SIZE]),
            page_no,
        }
    }
}

/// A logically contiguous set of pages that can be transacted on.
///
/// The `PagePool` struct allows for the creation of:
///
/// - [`Snapshot`] – readonly views of the pool's state at a particular point in time.
/// - [`Transaction`] - mutable views of the pool's state that can be modified and
///   eventually committed back to the pool.
///
/// # Examples
///
/// Modify the contents of a pool using a transaction:
///
/// ```
/// use pmem::page::{PagePool, TxWrite};
///
/// let mut pool = PagePool::new(5);  // Initialize a pool with 5 pages
/// let mut tx = pool.start();        // Create a new transaction
/// tx.write(0, &[0, 1, 2, 3]);       // Modify the snapshot
/// pool.commit(tx);                  // Commit the changes back to the pool
/// ```
///
/// Creating a snapshot of the pool:
/// ```
/// use pmem::page::{PagePool, TxWrite, TxRead};
///
/// let mut pool = PagePool::new(5);
/// let snapshot = pool.snapshot();   // Create a new snapshot
///
/// let mut tx = pool.start();
/// tx.write(0, &[0, 1, 2, 3]);
/// pool.commit(tx);
///
/// // The snapshot remains unchanged even after committing changes to the pool
/// assert_eq!(&*snapshot.read(0, 4), &[0, 0, 0, 0]);
/// ```
///
/// This structure is particularly useful for systems that require consistent views of data
/// at different points in time, or for implementing undo/redo functionality where each snapshot
/// can represent a state in the history of changes.
pub struct PagePool {
    pages: Arc<Mutex<Vec<Page>>>,

    /// Reference to the [`UndoEntry`] for the last committed transaction.
    ///
    /// When committing a transaction, this is used to create a snapshot and subsequent commits
    /// will append undo log with the changes required to maintain REPEATABLE READ isolation level for the
    /// snapshot.
    ///
    /// Ideally it should not contain `Option`, because it is required to be present
    /// for the snapshot to be created. Unfortunately, this is not possible due to the fact
    /// that [`UndoEntry::next`] which is [`ArcSwap`] must contains `Option` to be able
    /// to stop reference chain at some point and we can not have both `Arc<T>` and `Arc<Option<T>>` as the same time.
    undo_log: Arc<Option<UndoEntry>>,
    commit_log: Arc<Option<Commit>>,

    /// A log sequence number (LSN) that uniquely identifies this snapshot. The LSN
    /// is used internally to ensure that snapshots are applied in a linear and consistent order.
    lsn: LSN,
    notify: Arc<Condvar>,
}

impl PagePool {
    /// Constructs a new `PagePool` with a specified number of pages.
    ///
    /// This function initializes a `PagePool` instance with an empty set of patches and
    /// a given number of pages. See [`Self::with_capacity`] for creating a pool with a specified
    /// capacity in bytes.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages the pool should initially contain. This determines
    ///   the range of valid addresses that can be written to in snapshots derived from this pool.
    pub fn new(page_cnt: usize) -> Self {
        assert!(page_cnt > 0, "The number of pages must be greater than 0");
        let commit = Commit {
            changes: vec![],
            next: ArcSwap::from_pointee(None),
            lsn: 0,
        };
        let mut pages = Vec::with_capacity(page_cnt);
        for i in 0..page_cnt {
            pages.push(Page::new(i as u32));
        }
        Self {
            pages: Arc::new(Mutex::new(pages)),
            undo_log: Arc::new(Some(UndoEntry::default())),
            commit_log: Arc::new(Some(commit)),
            notify: Arc::new(Condvar::new()),
            lsn: 0,
        }
    }

    pub fn with_capacity(bytes: usize) -> Self {
        let pages = (bytes + PAGE_SIZE) / PAGE_SIZE;
        Self::new(pages)
    }

    /// Creates a new transaction over the current state of the page pool.
    ///
    /// The transaction can be used to perform modifications independently of the pool. These modifications
    /// are not reflected in the pool until the snapshot is committed using the [`commit`] method.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use pmem::page::PagePool;
    /// let mut pool = PagePool::new(1);
    /// let tx = pool.start();
    /// // tx modifications won't affect the original pool until committed.
    /// ```
    ///
    /// [`commit`]: Self::commit
    pub fn start(&self) -> Transaction {
        Transaction {
            uncommitted: vec![],
            base: self.snapshot(),
        }
    }

    /// Commits the changes made in a transaction back to the page pool.
    ///
    /// This method updates the state of the page pool to reflect the modifications
    /// recorded in the provided transaction. Once committed, its changes are visible in subsequent operations.
    ///
    /// Each transaction is linked to the pool state it was created from. If the page poll was changed
    /// since the moment when transaction was created, attempt to commit such a transaction will return an
    /// [`Result::Err`], because such changes might not be consistent anymore.
    pub fn commit(&mut self, tx: impl Into<Transaction>) -> Result<u64, u64> {
        let tx: Transaction = tx.into();
        if tx.base.lsn != self.lsn {
            return Err(self.lsn);
        }

        let lsn = self.lsn + 1;

        // Updating pages and undo log
        {
            let mut locked_pages: MutexGuard<Vec<Page>> = self.pages.lock().unwrap();

            let mut patches = Vec::with_capacity(tx.uncommitted.len());
            for patch in &tx.uncommitted {
                let segments = PageSegments::new(patch.addr(), patch.len());
                let mut undo_patch = vec![0; patch.len()];
                for (addr, slice_range) in segments {
                    let (page_no, offset) = split_ptr(addr);
                    let offset = offset as usize;
                    let len = slice_range.len();
                    let page_range = offset..(offset + len);

                    let page = find_or_create_page(&mut locked_pages, page_no);

                    // Forming undo patch
                    undo_patch[slice_range.clone()].copy_from_slice(&page.data[page_range.clone()]);

                    // Applying change to the page
                    match patch {
                        Patch::Write(_, data) => {
                            page.data[page_range].copy_from_slice(&data[slice_range])
                        }
                        Patch::Reclaim(..) => page.data[page_range].fill(0),
                    }
                }
                patches.push(Patch::Write(patch.addr(), undo_patch));
            }
            let undo = Arc::new(Some(UndoEntry {
                next: ArcSwap::from_pointee(None),
                patches,
                lsn,
            }));
            (*self.undo_log)
                .as_ref()
                .expect("Undo log is missing")
                .next
                .store(Arc::clone(&undo));
            self.undo_log = undo;
        }

        // Updating redo log
        let new_commit = Arc::new(Some(Commit {
            changes: tx.uncommitted,
            next: ArcSwap::from_pointee(None),
            lsn,
        }));
        (*self.commit_log)
            .as_ref()
            .expect("Commit log is missing")
            .next
            .store(Arc::clone(&new_commit));
        self.commit_log = new_commit;

        self.lsn = lsn;
        self.notify.notify_all();
        Ok(lsn)
    }

    pub fn commit_notify(&self) -> CommitNotify {
        CommitNotify {
            last_seen_lsn: self.lsn,
            notify: Arc::clone(&self.notify),
            commit: Arc::clone(&self.commit_log),
            pages_count: self.pages.lock().unwrap().len() as u32,
            pages: Arc::clone(&self.pages),
        }
    }

    /// Creates read-only handle to the page pool that may be used to read data from it from different threads.
    pub fn handle(&self) -> PagePoolHandle {
        PagePoolHandle {
            notify: self.commit_notify(),
            pages: Arc::clone(&self.pages),
            pages_count: self.pages.lock().unwrap().len() as u32,
            undo_log: Arc::clone(&self.undo_log),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        debug_assert!(self.undo_log.is_some());

        Snapshot {
            lsn: self.lsn,
            pages_count: self.pages.lock().unwrap().len() as u32,
            pages: Arc::clone(&self.pages),
            undo_log: Arc::clone(&self.undo_log),
        }
    }

    #[cfg(test)]
    fn read(&self, addr: Addr, len: usize) -> Cow<[u8]> {
        let mut buffer = vec![0; len];

        let snapshot = self.snapshot();
        let buf = snapshot.read(addr, len);
        buffer.copy_from_slice(&buf);
        Cow::Owned(buffer)
    }
}

#[cfg(test)]
impl Default for PagePool {
    fn default() -> Self {
        Self::new(1)
    }
}

fn find_or_create_page<'a>(pages: &'a mut MutexGuard<Vec<Page>>, page: PageNo) -> &'a mut Page {
    let result = pages.binary_search_by_key(&page, |p| p.page_no);
    if let Err(idx) = result {
        pages.insert(idx, Page::new(page));
    }
    let idx = result.unwrap_or_else(|idx| idx);
    &mut pages[idx]
}

fn find_page<'a>(pages: &'a mut MutexGuard<Vec<Page>>, page: PageNo) -> Option<&'a mut Page> {
    pages
        .binary_search_by_key(&page, |p| p.page_no)
        .ok()
        .map(|idx| &mut pages[idx])
}

/// Describes a changes that need to be applied to the page to restore it to the state
/// before particular transaction was committed.
///
/// `UndoEntry` is referenced from old to new and only last position is stored in [`PagePool`].
/// It enables automatic cleanup of old entries that are not referenced anymore by any snapshot or transaction.
#[derive(Default)]
struct UndoEntry {
    next: ArcSwap<Option<UndoEntry>>,
    patches: Vec<Patch>,
    lsn: LSN,
}

#[derive(Clone)]
pub struct PagePoolHandle {
    notify: CommitNotify,
    pages_count: PageNo,
    undo_log: Arc<Option<UndoEntry>>,
    pages: Arc<Mutex<Vec<Page>>>,
}

impl PagePoolHandle {
    /// Blocks until next snapshot is available and returns it
    ///
    /// The next snapshot is the one that is the most recent at the time this method is called.
    /// It might be several snapshots ahead of the last seen snapshot.
    pub fn wait_for_commit(&mut self) -> Snapshot {
        let commit = self.notify.next_commit();

        Snapshot {
            lsn: commit.lsn,
            pages_count: self.pages_count,
            pages: Arc::clone(&self.pages),
            undo_log: Arc::clone(&self.undo_log),
        }
    }
}

#[derive(Clone)]
pub struct CommitNotify {
    pages: Arc<Mutex<Vec<Page>>>,
    commit: Arc<Option<Commit>>,
    last_seen_lsn: u64,
    notify: Arc<Condvar>,
    pages_count: PageNo,
}

impl CommitNotify {
    /// Blocks until next commit is available and returns it
    ///
    /// If several commits happened since the last call to this method, this function will return
    /// all of them in order.
    pub fn next_commit(&mut self) -> &Commit {
        let last_commit = (*self.commit).as_ref().unwrap();
        if last_commit.next().is_none() {
            let mut locked_commit = self.pages.lock().unwrap();
            // Need to check again after acquiring the lock, otherwise it is a race condition
            // because we speculatively checked the condition before acquiring the lock to prevent
            // contention when possible
            while last_commit.next().is_none() {
                locked_commit = self.notify.wait(locked_commit).unwrap();
            }
        }
        let next_commit = last_commit.next();
        debug_assert!((*next_commit).as_ref().unwrap().lsn() > self.last_seen_lsn());
        self.commit = Arc::clone(&next_commit);
        (*self.commit).as_ref().unwrap()
    }

    pub fn last_seen_lsn(&self) -> u64 {
        self.last_seen_lsn
    }

    pub fn pages(&self) -> PageNo {
        // TODO is this correct? Can the number of pages change?
        self.pages_count
    }
}

/// Trait describing a read-only snapshot of a page pool.
///
/// Represents a consistent snapshot of a page pool at a specific point in time.
pub trait TxRead {
    /// Reads the specified number of bytes from the given address.
    ///
    /// # Panics
    /// Panic if the address is out of bounds. See [`Self::valid_range`] for bounds checking.
    fn read(&self, addr: Addr, len: usize) -> Cow<[u8]>;

    /// Checks if the specified range of addresses is valid.
    ///
    /// The address is not valid if it addresses the pages that are outside of the pool bounds.
    fn valid_range(&self, addr: Addr, len: usize) -> bool;
}

/// Trait describing a transaction that can modify data in a page pool.
///
/// Compared to [`TxRead`], this trait allows for writing and reclaiming data.
pub trait TxWrite: TxRead {
    /// Writes the specified bytes to the given address.
    ///
    /// # Panics
    /// Panic if the address is out of bounds. See [`TxRead::valid_range`] for bounds checking.
    fn write(&mut self, addr: Addr, bytes: impl Into<Vec<u8>>);

    /// Frees the given segment of memory.
    ///
    /// Reclaim is guaranteed to follow zeroing semantics. The read operation from the corresponding segment
    /// after reclaiming will return zeros.
    ///
    /// The main purpose of reclaim is to mark memory as not used, so it can be freed from persistent storage
    /// after snapshot compaction.
    fn reclaim(&mut self, addr: Addr, len: usize);
}

/// Represents a committed snapshot of a page pool. Created by [`PagePool::snapshot`] method.
///
/// A `Snapshot` captures the state of a page pool at a specific point in time,
/// including any patches (modifications) that have been applied up to that point. It serves
/// as a read-only view into the historical state of the pool, allowing for consistent reads
/// of pages as they existed at the time of the snapshot.
///
/// Snapshots are following repeatable read isolation level, meaning that they are not affected
/// by any changes made to the pool after the snapshot was taken.
#[derive(Clone)]
pub struct Snapshot {
    pages_count: PageNo,
    undo_log: Arc<Option<UndoEntry>>,
    pages: Arc<Mutex<Vec<Page>>>,
    lsn: LSN,
}

impl Snapshot {
    // Main method that implements MVCC read logic of the system.
    //
    // Because [`apply_patches()`] is implementing first patch wins logic, the correct order of reading
    // data at a given LSN is following:
    //
    // 1. Read whatever data is in the pages.
    // 2. Apply changes from own patches. Any transaction should see its own not yet committed cnanges.
    // 3. Apply changes from undo log, to restore the state of the page that is possibly changed
    //    by concurrently committed transaction. This is needed to maintain REPETABLE READ
    //    isolation guarantee.
    fn read_uncommitted(&self, addr: Addr, len: usize, uncommitted: &[Patch]) -> Cow<[u8]> {
        split_ptr_checked(addr, len, self.pages_count);
        let mut buf = vec![0; len];

        // Reading the pages
        {
            let mut locked_pages = self.pages.lock().unwrap();
            for (addr, range) in PageSegments::new(addr, len) {
                let (page_no, offset) = split_ptr(addr);
                let offset = offset as usize;
                if let Some(page) = find_page(&mut locked_pages, page_no) {
                    let len = range.len();
                    buf[range].copy_from_slice(&page.data[offset..offset + len]);
                }
            }
        }

        #[allow(clippy::single_range_in_vec_init)]
        let mut buf_mask = vec![0..len];

        // Apply own uncommitted changes
        apply_patches(uncommitted, addr as usize, &mut buf, &mut buf_mask);

        // Applying with undo log
        // We need to skip first entry in undo log, because it is describing how to undo the changes
        // of previously applied transaction.
        let mut log = (*self.undo_log).as_ref().unwrap().next.load_full();
        while let Some(undo) = log.as_ref() {
            if undo.lsn > self.lsn {
                apply_patches(&undo.patches, addr as usize, &mut buf, &mut buf_mask);
            }
            log = undo.next.load_full();
        }

        Cow::Owned(buf)
    }

    pub fn lsn(&self) -> LSN {
        self.lsn
    }
}

impl TxRead for Snapshot {
    fn valid_range(&self, addr: Addr, len: usize) -> bool {
        is_valid_ptr(addr, len, self.pages_count)
    }

    fn read(&self, addr: Addr, len: usize) -> Cow<[u8]> {
        self.read_uncommitted(addr, len, &[])
    }
}

pub struct Commit {
    /// A changes that have been applied in this snapshot.
    ///
    /// This array is sorted by the address of the change and all patches are non-overlapping.
    changes: Vec<Patch>,

    /// A reference to the next commit in the chain.
    ///
    /// This is updated atomically by the thread that executing commit operation.
    next: ArcSwap<Option<Commit>>,

    /// A log sequence number (LSN) that uniquely identifies this commit.
    /// LSN numbers are monotonically increasing.
    lsn: LSN,
}

impl Commit {
    pub fn patches(&self) -> &[Patch] {
        self.changes.as_slice()
    }

    pub fn lsn(&self) -> LSN {
        self.lsn
    }

    pub fn next(&self) -> Arc<Option<Commit>> {
        self.next.load_full()
    }
}

#[derive(Clone)]
pub struct Transaction {
    uncommitted: Vec<Patch>,
    base: Snapshot,
}

impl TxRead for Transaction {
    fn valid_range(&self, addr: Addr, len: usize) -> bool {
        self.base.valid_range(addr, len)
    }

    fn read(&self, addr: Addr, len: usize) -> Cow<[u8]> {
        self.base.read_uncommitted(addr, len, &self.uncommitted)
    }
}

impl TxWrite for Transaction {
    fn write(&mut self, addr: Addr, bytes: impl Into<Vec<u8>>) {
        let bytes = bytes.into();
        split_ptr_checked(addr, bytes.len(), self.base.pages_count);

        if !bytes.is_empty() {
            push_patch(&mut self.uncommitted, Patch::Write(addr, bytes));
        }
    }

    fn reclaim(&mut self, addr: Addr, len: usize) {
        split_ptr_checked(addr, len, self.base.pages_count);
        if len > 0 {
            push_patch(&mut self.uncommitted, Patch::Reclaim(addr, len));
        }
    }
}

/// Pushes a patch to a list ensuring the following invariants hold:
///
/// 1. All patches are sorted by [`Patch::addr()`]
/// 2. All patches are non-overlapping
fn push_patch(patches: &mut Vec<Patch>, patch: Patch) {
    assert!(patch.len() > 0, "Patch should not be empty");
    let connected = find_connected_ranges(patches, &patch);

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

    debug_assert_sorted_and_has_no_overlaps(patches);
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
/// ranges that still need to be patched. Therefore when calling several times this function will implement
/// the "first patch wins" logic.
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
            .binary_search_by(|i| (i.end() - 1).cmp(&(start_addr as Addr)))
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
        "Address range is out of bounds 0x{:08x}-0x{:08x}. Max address 0x{:08x}",
        addr,
        addr + len as Addr,
        pages as u64 * PAGE_SIZE as u64
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
    let page_no = (addr >> PAGE_SIZE_BITS) & 0xFFFF_FFFF;
    let offset = addr & (PAGE_SIZE - 1) as u64;
    (page_no.try_into().unwrap(), offset.try_into().unwrap())
}

/// Returns true of given patch intersects given range of bytes
fn intersects(patch: &Patch, range: &Range<usize>) -> bool {
    let start = patch.addr() as usize;
    let end = start + patch.len();
    start < range.end && end > range.start
}

/// Iterator over page segments for a given address and size.
///
/// Because data is stored in pages when you want to read or write data you must must always align
/// operations to page boundaries. This iterator provides a way to calculate page numbers, and corresponding
/// offsets to copy to/from the page.
///
/// Page segments are represented by following tuple:
/// - address – the page address for the next operation
/// - range – the range within slice to operate on (relative to base address)
///
/// # Example
///
/// This example will read data without any cross-page operations.
///
/// ```nocompile
/// use crate::PageSegments;
/// use crate::PAGE_SIZE;
///
/// let base_addr = 0x1000;
/// let size = 0x100;
/// let buffer = vec![0; size];
///
/// let segments = PageSegments::new(base_addr, size);
/// for (addr, range) in segments {
///     let result = tx.read(addr, range.len());
///     buffer[range].copy_from_slice(&result);
/// }
/// ```
struct PageSegments {
    base_addr: Addr,
    start_addr: Addr,
    end_addr: Addr,
}

impl PageSegments {
    fn new(base_addr: Addr, size: usize) -> Self {
        PageSegments {
            base_addr,
            start_addr: base_addr,
            end_addr: base_addr + size as Addr,
        }
    }
}

impl Iterator for PageSegments {
    type Item = (Addr, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.end_addr == self.start_addr {
            return None;
        }
        let len = if are_on_the_same_page(self.start_addr, self.end_addr - 1) {
            self.end_addr - self.start_addr
        } else {
            let (page, _) = split_ptr(self.start_addr);
            let next_page_addr = (page + 1) as Addr * PAGE_SIZE as Addr;
            next_page_addr - self.start_addr
        };
        let offset = (self.start_addr - self.base_addr) as usize;
        let addr = self.start_addr;
        self.start_addr += len as Addr;
        Some((addr, offset..(offset + len as usize)))
    }
}

impl DoubleEndedIterator for PageSegments {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end_addr == self.start_addr {
            return None;
        }
        let len = if are_on_the_same_page(self.start_addr, self.end_addr - 1) {
            (self.end_addr - self.start_addr) as usize
        } else {
            let (page, _) = split_ptr(self.end_addr - 1);
            let page_start_addr = page as Addr * PAGE_SIZE as Addr;
            (self.end_addr - page_start_addr) as usize
        };
        let offset = (self.end_addr - self.base_addr - len as Addr) as usize;
        let end_addr = self.end_addr;
        self.end_addr -= len as Addr;
        Some((end_addr - len as Addr, offset..(offset + len)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{panic, thread};

    #[test]
    fn first_commit_should_have_lsn_1() {
        // In several places in codebase we assume that first commit has LSN 1
        // and LSN 0 is synthetic LSN used for base empty snapshot
        let mut mem = PagePool::default();
        let tx = mem.start();
        assert_eq!(mem.commit(tx).unwrap(), 1);
    }

    #[test]
    fn non_linear_commits_must_be_rejected() {
        let mut mem = PagePool::default();
        let mut tx1 = mem.start();
        let mut tx2 = mem.start();

        tx1.write(0, b"Hello");
        mem.commit(tx1).unwrap();

        tx2.write(0, b"World");
        assert!(mem.commit(tx2).is_err());
    }

    #[test]
    fn committed_changes_should_be_visible_on_a_page() {
        let mut mem = PagePool::from("Jekyll");

        let mut tx = mem.start();
        tx.write(0, b"Hide");
        mem.commit(tx).unwrap();

        assert_str_eq(mem.read(0, 4), b"Hide");
    }

    #[test]
    fn uncommitted_changes_should_be_visible_only_on_the_snapshot() {
        let mem = PagePool::from("Jekyll");

        let mut tx = mem.start();
        tx.write(0, b"Hide");

        assert_str_eq(tx.read(0, 4), "Hide");
        assert_str_eq(mem.read(0, 6), "Jekyll");
    }

    #[test]
    fn snapshot_should_provide_repeatable_read_isolation() {
        let mut mem = PagePool::default();

        let zero = mem.snapshot();

        let mut hide = mem.start();
        hide.write(0, b"Hide");
        mem.commit(hide).unwrap();
        let hide = mem.start();

        let mut jekyll = mem.start();
        jekyll.write(0, b"Jekyll");
        mem.commit(jekyll).unwrap();
        let jekyll = mem.start();

        assert_eq!(&*zero.read(0, 6), b"\0\0\0\0\0\0");
        assert_eq!(&*hide.read(0, 6), b"Hide\0\0");
        assert_eq!(&*jekyll.read(0, 6), b"Jekyll");
    }

    #[test]
    fn patch_page() {
        let mut mem = PagePool::from("Hello panic!");

        let mut tx = mem.start();
        tx.write(6, b"world");
        mem.commit(tx).unwrap();

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

        let mut tx = mem.start();
        tx.write(70, [0, 0, 1]);
        mem.commit(tx).unwrap();
        let mut tx = mem.start();
        tx.reclaim(71, 2);
        tx.write(73, [0]);
        mem.commit(tx).unwrap();

        assert_eq!(mem.read(70, 4).as_ref(), [0, 0, 0, 0]);
    }

    #[test]
    fn test_regression2() {
        let mut mem = PagePool::default();

        let mut tx = mem.start();
        tx.write(70, [0, 0, 1]);
        mem.commit(tx).unwrap();
        let mut tx = mem.start();
        tx.write(0, [0]);
        mem.commit(tx).unwrap();
        let mut tx = mem.start();
        tx.reclaim(71, 2);
        mem.commit(tx).unwrap();

        assert_eq!(mem.read(70, 4).as_ref(), [0, 0, 0, 0]);
    }

    #[test]
    fn data_across_multiple_pages_can_be_written() {
        let mut mem = PagePool::new(2);

        // Choosing address so that data is split across 2 pages
        let addr = PAGE_SIZE as Addr - 2;
        let alice = b"Alice";

        let mut tx = mem.start();
        tx.write(addr, alice);

        // Checking that data is visible in snapshot
        assert_str_eq(tx.read(addr, alice.len()), alice);

        // Checking that data is visible after commit to page pool
        mem.commit(tx).unwrap();
        assert_str_eq(mem.read(addr, alice.len()), alice);
    }

    #[test]
    fn data_can_be_read_from_snapshot() {
        let data = [1, 2, 3, 4, 5];
        let mem = PagePool::from(data.as_slice());

        let tx = mem.start();
        assert_eq!(tx.read(0, 5), data.as_slice());
    }

    #[test]
    fn data_can_be_removed_on_snapshot() {
        let data = [1, 2, 3, 4, 5];
        let mem = PagePool::from(data.as_slice());

        let mut tx = mem.start();
        tx.reclaim(1, 3);

        assert_eq!(tx.read(0, 5), [1u8, 0, 0, 0, 5].as_slice());
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_read() {
        // reading in a such a way that start address is still valid, but end address is not
        PagePool::default().start().read(PAGE_SIZE as Addr - 10, 20);
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_write() {
        PagePool::default()
            .start()
            .write(PAGE_SIZE as Addr - 10, [0; 20]);
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_reclaim() {
        PagePool::default()
            .start()
            .reclaim(PAGE_SIZE as Addr - 10, 20);
    }

    /// When dropping PagePool all related snapshots will be removed. It may lead
    /// to stackoverflow if snapshots removed recursively.
    #[test]
    fn deep_snapshot_should_not_cause_stack_overflow() {
        thread::Builder::new()
            .name("deep_snapshot_should_not_cause_stack_overflow".to_string())
            // setting stacksize explicitly so not to rely on the running environment
            .stack_size(100 * 1024)
            .spawn(|| {
                let mut mem = PagePool::new(100);
                for _ in 0..1000 {
                    mem.commit(mem.start()).unwrap();
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn data_can_be_read_from_a_different_thread() {
        let mut pool = PagePool::new(1);

        let mut handle = pool.handle();

        let lsn = thread::spawn(move || {
            let mut tx = pool.start();
            tx.write(0, [1, 2, 3, 4]);
            pool.commit(tx).unwrap()
        });

        let commit = handle.wait_for_commit();
        let bytes = commit.read(0, 4);

        let lsn = lsn.join().unwrap();
        assert_eq!(commit.lsn, lsn);
        assert_eq!(&*bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn can_get_notifications_about_commit() {
        let mut pool = PagePool::new(1);
        let mut n1 = pool.commit_notify();
        let mut n2 = pool.commit_notify();

        let lsn = thread::spawn(move || {
            let mut tx = pool.start();
            tx.write(0, [1]);
            pool.commit(tx).unwrap()
        });

        let lsn2 = n2.next_commit().lsn();
        let lsn1 = n1.next_commit().lsn();

        let lsn = lsn.join().unwrap();

        assert_eq!(lsn1, lsn);
        assert_eq!(lsn2, lsn);
    }

    #[test]
    fn each_commit_snapshot_can_be_addressed() {
        let mut pool = PagePool::new(1);
        let mut notify = pool.commit_notify();

        for i in 1..=10 {
            let mut tx = pool.start();
            tx.write(i, [i as u8]);
            pool.commit(tx).unwrap();
        }

        for lsn in 1..=10 {
            let commit = notify.next_commit();
            assert_eq!(lsn, commit.lsn());
        }
    }

    #[test]
    fn can_wait_for_a_snapshot_in_a_thread() {
        let mut pool = PagePool::new(1);
        let mut notify = pool.commit_notify();

        let lsn = thread::spawn(move || notify.next_commit().lsn());

        let mut tx = pool.start();
        tx.write(0, [0]);
        let expected_lsn = pool.commit(tx).unwrap();

        let lsn = lsn.join().unwrap();
        assert_eq!(lsn, expected_lsn);
    }

    #[test]
    fn check_iterator_over_page_segments() {
        assert_eq!(PageSegments::new(10, 0).collect::<Vec<_>>(), vec![]);

        assert_eq!(
            PageSegments::new(0, 10).collect::<Vec<_>>(),
            vec![(0, 0..10)]
        );

        assert_eq!(
            PageSegments::new(PAGE_SIZE as Addr / 2, PAGE_SIZE).collect::<Vec<_>>(),
            vec![
                (PAGE_SIZE as Addr / 2, 0..PAGE_SIZE / 2),
                (PAGE_SIZE as Addr, PAGE_SIZE / 2..PAGE_SIZE),
            ]
        );

        assert_eq!(
            PageSegments::new(1, 2 * PAGE_SIZE).collect::<Vec<_>>(),
            vec![
                (1, 0..PAGE_SIZE - 1),
                (PAGE_SIZE as Addr, (PAGE_SIZE - 1)..(2 * PAGE_SIZE - 1)),
                (2 * PAGE_SIZE as Addr, (2 * PAGE_SIZE - 1)..(2 * PAGE_SIZE)),
            ]
        );
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
            let addr = PAGE_SIZE as Addr - 1; // Last address of the first page
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

            assert_eq!(page_no, 0xFFFFFFFF);
            assert_eq!(offset, 0xFFFF);
        }
    }

    #[cfg(not(miri))]
    mod proptests {
        use super::*;
        use proptest::{collection::vec, prelude::*};
        use NormalizedPatches::*;

        /// The size of database for testing
        const DB_SIZE: usize = PAGE_SIZE * 2;

        proptest! {
            #[test]
            fn page_segments_len((addr, len) in any_addr_and_len()) {
                let interval = PageSegments::new(addr, len);
                let len_sum = interval.map(|(_, slice_range)| slice_range.len()).sum::<usize>();
                prop_assert_eq!(len_sum, len);
            }

            #[test]
            fn page_segments_are_connected((addr, len) in any_addr_and_len()) {
                let segments = PageSegments::new(addr, len)
                    .map(|(_, slice_range)| slice_range)
                    .collect::<Vec<_>>();
                for i in segments.windows(2) {
                    prop_assert_eq!(i[0].end, i[1].start);
                }
            }

            #[test]
            fn page_segments_are_equivalent_to_reverted((addr, len) in any_addr_and_len()) {
                let segments = PageSegments::new(addr, len).collect::<Vec<_>>();

                let mut segments_rev = PageSegments::new(addr, len).rev().collect::<Vec<_>>();
                segments_rev.sort_by_key(|(addr, _)| *addr);

                prop_assert_eq!(segments, segments_rev);
            }

            /// This test uses "shadow writes" to check if snapshot writing and reading
            /// algorithms are consistent. We do it by mirroring all patches to a shadow buffer sequentially.
            /// In the end, the final snapshot state should be equal to the shadow buffer.
            #[test]
            fn shadow_write(snapshots in vec(any_snapshot(), 0..3)) {
                let mut shadow_buffer = vec![0; DB_SIZE];
                let mut mem = PagePool::with_capacity(DB_SIZE);

                for patches in snapshots {
                    let mut tx = mem.start();
                    for patch in patches {
                        let offset = patch.addr() as usize;
                        let range = offset..offset + patch.len();
                        match patch {
                            Patch::Write(offset, bytes) => {
                                shadow_buffer[range].copy_from_slice(bytes.as_slice());
                                tx.write(offset, bytes);
                            },
                            Patch::Reclaim(offset, len) => {
                                shadow_buffer[range].fill(0);
                                tx.reclaim(offset, len);

                            }
                        }
                    }
                    mem.commit(tx).unwrap();
                }

                assert_buffers_eq(&mem.read(0, DB_SIZE), shadow_buffer.as_slice())?;
            }

            /// This test ensure that no matter transactions are committed, snapshot should always
            /// return the same initial state.
            #[test]
            fn repeatable_read(snapshots in vec(any_snapshot(), 1..5)) {
                const VALUE: u8 = 42;
                let initial = vec![VALUE; DB_SIZE];
                let mut mem = PagePool::from(initial.as_slice());
                let s = mem.snapshot();

                for patches in snapshots {
                    let mut tx = mem.start();
                    for patch in patches {
                        match patch {
                            Patch::Write(offset, bytes) => {
                                tx.write(offset, bytes);
                            },
                            Patch::Reclaim(offset, len) => {
                                tx.reclaim(offset, len);
                            }
                        }
                    }

                    prop_assert!(s.read(0, DB_SIZE).iter().all(|b| *b == VALUE));
                    mem.commit(tx).unwrap();
                    prop_assert!(s.read(0, DB_SIZE).iter().all(|b| *b == VALUE));
                }
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
            fn connected_patches_are_adjacent_after_normalization((a, b) in patches::connected()) {
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

        fn any_addr_and_len() -> impl Strategy<Value = (Addr, usize)> {
            (0..100 * PAGE_SIZE as Addr, 0..20 * PAGE_SIZE)
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

            pub(super) fn connected() -> impl Strategy<Value = (Patch, Patch)> {
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
                        let offset = rng.gen_range(1..offset) as Addr;
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
                        b.set_addr(a.addr() + b_offset as Addr);
                        (a, b)
                    })
            }

            pub(super) fn any_patch_with_len(len: Range<usize>) -> impl Strategy<Value = Patch> {
                prop_oneof![write_patch(len.clone()), reclaim_patch(len)]
            }

            pub(super) fn any_patch() -> impl Strategy<Value = Patch> {
                prop_oneof![write_patch(1..16), reclaim_patch(1..16)]
            }

            pub(super) fn write_patch(len: Range<usize>) -> impl Strategy<Value = Patch> {
                (0..DB_SIZE, vec(any::<u8>(), len))
                    .prop_filter("out of bounds patch", |(offset, bytes)| {
                        offset + bytes.len() < DB_SIZE
                    })
                    .prop_map(|(offset, bytes)| Patch::Write(offset as Addr, bytes))
            }

            pub(super) fn reclaim_patch(len: Range<usize>) -> impl Strategy<Value = Patch> {
                (0..DB_SIZE, len)
                    .prop_filter("out of bounds patch", |(offset, len)| {
                        (*offset + *len) < DB_SIZE
                    })
                    .prop_map(|(offset, len)| Patch::Reclaim(offset as Addr, len))
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

    impl From<&str> for PagePool {
        fn from(value: &str) -> Self {
            Self::from(value.as_bytes())
        }
    }

    impl From<&[u8]> for PagePool {
        fn from(value: &[u8]) -> Self {
            let capactiy = PAGE_SIZE.max(value.len());
            let mut pool = PagePool::with_capacity(capactiy);
            let mut tx = pool.start();
            tx.write(0, value);
            pool.commit(tx).unwrap();
            pool
        }
    }

    impl MemRange for Range<u32> {
        fn addr(&self) -> Addr {
            self.start as Addr
        }

        fn len(&self) -> usize {
            (self.end - self.start) as usize
        }
    }
}
