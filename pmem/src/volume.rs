//! # Page Management Module
//!
//! This module provides a system for managing and manipulating snapshots of a memory. It is designed
//! to facilitate operations on persistent memory (pmem), allowing for efficient snapshots, modifications, and commits
//! of data changes. The core functionality revolves around the [`Volume`] structure, which manages a volume of pages,
//! and the [`Transaction`] structure, which represents a modifiable session of the volume's state at a given point
//! in time.
//!
//! ## Concepts
//!
//! - [`Volume`]: A collection of pages that can be snapshoted, modified, and committed. It acts as the primary
//!   interface for interacting with the page memory.
//! - [`Transaction`]: A read/write session allowing modifications of the volume.
//!   It can be modified independently of the volume, and later committed back to the volume to update its state.
//! - [`Snapshot`]: A frozen state of the volume at some point in time. It is immutable and only can be
//!   used to read data.
//! - [`Patch`]: A modification recorded in a transaction. It consists of the address where the modification starts and
//!   the bytes that were written.
//! - [`Volume`]: A read-only view of the volume that can be used to read data from the volume and
//!   wait for new snapshots to be committed. It's useful for concurrent access to the volume.
//!
//! ## Usage
//!
//! The module is designed to be used as follows:
//!
//! 1. **Initialization**: Create a new [`Volume`].
//! 2. **Transactioning**: Create a [`Transaction`] of the current state.
//! 3. **Modification**: Use the [`Transaction`] to perform modifications. Each modification is recorded as a patch.
//! 4. **Commit**: Commit the snapshot back to the [`Volume`], applying all the patches and updating
//!    the volume's state.
//! 5. **Concurrent Access**: Optionally, create a [`Volume`] for read-only access to the volume
//!    from other threads.
//!
//! ## Example
//!
//! ```rust
//! use pmem::volume::{Volume, TxRead, TxWrite};
//!
//! let mut volume = Volume::new_in_memory(5);    // Initialize a volume with 5 pages
//!
//! // Create a handle for concurrent read-only access
//! let mut handle = volume.handle();
//!
//! let mut tx = volume.start();  // Create a new transaction
//! tx.write(0, &[1, 2, 3, 4]);   // Write 4 bytes at offset 0
//! volume.commit(tx);            // Commit the changes back to the volume
//!
//! let snapshot = handle.wait();
//! assert_eq!(snapshot.read(0, 4), vec![1, 2, 3, 4]); // Read using TxRead trait
//! ```
//!
//! The [`TxRead`] and [`TxWrite`] traits provide methods for reading from and writing to snapshots, respectively.
//!
//! ## Performance Considerations
//!
//! Since snapshots do not require duplicating the entire state of the volume, they can be created with minimal
//! overhead, making it perfectly valid and cost-effective to create a snapshot even when the intention is only to
//! read data without any modifications.

use crate::driver::{NoDriver, PageDriver};
use arc_swap::ArcSwap;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    io,
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
};
use tracing::{debug, error, info, trace};

pub const PAGE_SIZE_BITS: usize = 16;

/// The size of a page in bytes.
///
/// All memory is divided into pages of this size. The pages are the smallest unit of memory that can be
/// read or written to disk or network.
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

    #[cfg(test)]
    fn read_corresponding_segment<'a>(&self, snapshot: &'a impl TxRead) -> Cow<'a, [u8]> {
        snapshot.read(self.addr(), self.len())
    }

    #[cfg(test)]
    fn write_to(self, tx: &mut impl TxWrite) {
        match self {
            Patch::Write(addr, bytes) => tx.write(addr, bytes),
            Patch::Reclaim(addr, size) => tx.reclaim(addr, size),
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

#[allow(clippy::len_without_is_empty)]
pub trait MemRange {
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

/// A logically contiguous set of pages that can be transacted on.
///
/// The `Volume` struct allows for the creation of:
///
/// - [`Snapshot`] – readonly views of the volume's state at a particular point in time.
/// - [`Transaction`] - mutable views of the volume's state that can be modified and
///   eventually committed back to the volume.
///
/// # Examples
///
/// Modify the contents of a volume using a transaction:
///
/// ```
/// use pmem::volume::{Volume, TxWrite};
///
/// let mut volume = Volume::new_in_memory(5);  // Initialize a volume with 5 pages
/// let mut tx = volume.start();        // Create a new transaction
/// tx.write(0, &[0, 1, 2, 3]);       // Modify the snapshot
/// volume.commit(tx);                  // Commit the changes back to the volume
/// ```
///
/// Creating a snapshot of the volume:
/// ```
/// use pmem::volume::{Volume, TxWrite, TxRead};
///
/// let mut volume = Volume::new_in_memory(5);
/// let snapshot = volume.snapshot();   // Create a new snapshot
///
/// let mut tx = volume.start();
/// tx.write(0, &[0, 1, 2, 3]);
/// volume.commit(tx);
///
/// // The snapshot remains unchanged even after committing changes to the volume
/// assert_eq!(&*snapshot.read(0, 4), &[0, 0, 0, 0]);
/// ```
///
/// This structure is particularly useful for systems that require consistent views of data
/// at different points in time, or for implementing undo/redo functionality where each snapshot
/// can represent a state in the history of changes.
pub struct Volume {
    pages: Arc<Pages>,
    pages_count: PageNo,

    /// A log sequence number (LSN) that uniquely identifies this snapshot. The LSN
    /// is used internally to ensure that snapshots are applied in a linear and consistent order.
    lsn: AtomicU64,

    /// Reference to the latest commit and condition variable used to notify waiting threads
    /// that a new commit has been completed
    ///
    /// When committing a transaction, this is used to create a snapshot and subsequent commits
    /// will append undo log with the changes required to maintain REPEATABLE READ isolation level for the
    /// snapshot.
    ///
    /// Ideally it should not contain `Option`, because it is required to be present
    /// for the snapshot to be created. Unfortunately, this is not possible due to the fact
    /// that [`Commit::next`] which is [`ArcSwap`] must contains `Option` to be able
    /// to stop reference chain at some point and we can not have both `Arc<T>` and `Arc<Option<T>>` as the same time.
    latest_commit: Arc<(Mutex<Arc<Commit>>, Condvar)>,
}

impl Volume {
    /// Constructs a new `Volume` with a specified number of pages with memory page-driver.
    /// The data in this volume is **not persisted** to disk.
    ///
    /// This function initializes a `Volume` instance with an empty set of patches and
    /// a given number of pages. See [`Self::with_capacity`] for creating a volume with a specified
    /// capacity in bytes.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages the volume should initially contain. This determines
    ///   the range of valid addresses that can be written to in snapshots derived from this volume.
    pub fn new_in_memory(page_cnt: PageNo) -> Self {
        Self::new_with_driver(page_cnt, NoDriver::default())
    }

    // TODO page_cnt should be saved with driver
    pub fn from_commit(
        page_cnt: PageNo,
        commit: Commit,
        driver: impl PageDriver + 'static,
    ) -> Self {
        let commit = Arc::new(commit);
        let pages = Arc::new(Pages::new(driver, commit.clone()));
        let lsn = commit.lsn;
        let pages_count = page_cnt;
        let latest_commit = Arc::new((Mutex::new(commit), Condvar::new()));
        Self {
            pages,
            pages_count,
            lsn: AtomicU64::new(lsn),
            latest_commit,
        }
    }

    pub fn with_capacity(bytes: usize) -> Self {
        let pages = (bytes + PAGE_SIZE) / PAGE_SIZE;
        let pages = u32::try_from(pages).expect("Too large capacity for the volume");
        Self::new_in_memory(pages)
    }

    pub fn with_capacity_and_driver(bytes: usize, driver: impl PageDriver + 'static) -> Self {
        let pages = (bytes + PAGE_SIZE) / PAGE_SIZE;
        let pages = u32::try_from(pages).expect("Too large capacity for the volume");
        Self::new_with_driver(pages, driver)
    }

    pub fn new_with_driver(page_cnt: u32, driver: impl PageDriver + 'static) -> Self {
        assert!(page_cnt > 0, "The number of pages must be greater than 0");
        let commit = Commit {
            changes: vec![],
            undo: vec![],
            next: ArcSwap::from_pointee(None),
            lsn: 0,
        };
        let commit = Arc::new(commit);
        Self {
            pages: Arc::new(Pages::new(driver, Arc::clone(&commit))),
            pages_count: page_cnt as PageNo,
            latest_commit: Arc::new((Mutex::new(commit), Condvar::new())),
            lsn: AtomicU64::new(0),
        }
    }

    /// Creates a new transaction over the current state of the volume.
    ///
    /// The transaction can be used to perform modifications independently of the volume. These modifications
    /// are not reflected in the volume until the snapshot is committed using the [`commit`] method.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use pmem::volume::Volume;
    /// let mut volume = Volume::new_in_memory(1);
    /// let tx = volume.start();
    /// // tx modifications won't affect the original volume until committed.
    /// ```
    ///
    /// [`commit`]: Self::commit
    pub fn start(&self) -> Transaction {
        let snapshot = self.do_create_snapshot();
        trace!(base_lsn = snapshot.lsn(), "Creating transaction");
        Transaction {
            uncommitted: vec![],
            base: snapshot,
        }
    }

    /// Commits the changes made in a transaction back to the volume.
    ///
    /// This method updates the state of the volume to reflect the modifications
    /// recorded in the provided transaction. Once committed, its changes are visible in subsequent operations.
    ///
    /// Each transaction is linked to the volume state it was created from. If the page poll was changed
    /// since the moment when transaction was created, attempt to commit such a transaction will return an
    /// [`Result::Err`], because such changes might not be consistent anymore.
    pub fn commit(&mut self, tx: impl Into<Transaction>) -> io::Result<u64> {
        let tx: Transaction = tx.into();

        // Form undo patches
        let mut undo_patches = Vec::with_capacity(tx.uncommitted.len());
        for patch in &tx.uncommitted {
            let mut undo_patch = vec![0; patch.len()];
            self.pages.read(patch.addr(), undo_patch.as_mut()).unwrap();
            undo_patches.push(Patch::Write(patch.addr(), undo_patch));
        }

        let lsn = tx.base.lsn() + 1;
        // Updating redo log
        let new_commit = Commit {
            changes: tx.uncommitted,
            undo: undo_patches,
            next: ArcSwap::from_pointee(None),
            lsn,
        };

        self.apply_commit(new_commit)?;

        Ok(lsn)
    }

    /// This is internal function and is not supposed to be used by the end user.
    /// Made public for replication purposes.
    pub fn apply_commit(&mut self, commit: Commit) -> io::Result<()> {
        assert_patches_consistent(&commit.changes, &commit.undo);

        let (lock, notify) = self.latest_commit.as_ref();
        let mut last_commit = lock.lock().unwrap();

        let current_lsn = self.lsn.load(Ordering::Acquire);
        if commit.lsn != current_lsn + 1 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "The volume was changed since the transaction was created",
            ));
        }
        self.lsn.store(commit.lsn, Ordering::Release);

        let patches = commit.changes.clone();

        // Updating commit log
        // We need to update commit log before updating pages. Reading process should always have
        // undo logs to be able to rollback in-flight ch
        let lsn = commit.lsn;
        let changes = commit.changes.len();
        let commit = Arc::new(commit);
        let commit_clone = Arc::clone(&commit);
        last_commit.next.store(Arc::new(Some(Arc::clone(&commit))));
        *last_commit = commit;

        // Updating page content
        for patch in &patches {
            match patch {
                Patch::Write(addr, data) => self.pages.write(*addr, data)?,
                Patch::Reclaim(addr, len) => self.pages.zero(*addr, *len)?,
            }
        }
        self.pages.flush(commit_clone)?;

        notify.notify_all();

        info!(lsn, changes, "Commit completed");
        Ok(())
    }

    pub fn commit_notify(&self) -> CommitNotify {
        let commit = self.latest_commit.0.lock().unwrap();
        CommitNotify {
            last_seen_lsn: commit.lsn(),
            latest_commit: Arc::clone(&self.latest_commit),
            commit: Arc::clone(&commit),
            pages_count: self.pages_count,
        }
    }

    /// Creates read-only handle to the volume that may be used to read data from it from different threads.
    pub fn handle(&self) -> VolumeHandle {
        let notify = self.commit_notify();
        let commit_log = Arc::clone(&self.latest_commit.0.lock().unwrap());
        VolumeHandle {
            pages: Arc::clone(&self.pages),
            pages_count: self.pages_count,
            notify,
            commit_log,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let snapshot = self.do_create_snapshot();
        trace!(lsn = snapshot.lsn(), "Creating snapshot");
        snapshot
    }

    fn do_create_snapshot(&self) -> Snapshot {
        let commit = self.latest_commit.0.lock().unwrap();
        Snapshot {
            pages_count: self.pages_count,
            pages: Arc::clone(&self.pages),
            commit_log: Arc::clone(&commit),
        }
    }

    pub fn into_driver(self) -> Option<Box<dyn PageDriver>> {
        let pages = Arc::into_inner(self.pages)?;
        Some(pages.into_inner())
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

fn assert_patches_consistent(changes: &[Patch], undo: &[Patch]) {
    assert_eq!(
        changes.len(),
        undo.len(),
        "Undo/redo patches are not the same"
    );
    for (change, undo) in changes.iter().zip(undo.iter()) {
        assert_eq!(
            change.addr(),
            undo.addr(),
            "Undo/redo patches are not the same"
        );
        assert_eq!(
            change.len(),
            undo.len(),
            "Undo/redo patches are not the same"
        );
    }
}

#[cfg(test)]
impl Default for Volume {
    fn default() -> Self {
        Self::new_in_memory(1)
    }
}

struct Pages {
    pages: Mutex<BTreeMap<PageNo, Page>>,
    last_commit: Mutex<Arc<Commit>>,
    driver: Box<dyn PageDriver>,
}

impl Pages {
    pub fn new(driver: impl PageDriver + 'static, commit: Arc<Commit>) -> Self {
        Self {
            pages: Mutex::new(BTreeMap::new()),
            driver: Box::new(driver),
            last_commit: Mutex::new(commit),
        }
    }

    fn with_page(&self, page_no: PageNo, mut f: impl FnMut(&mut Page)) -> io::Result<()> {
        let mut pages = self.pages.lock().unwrap();
        if let Some(page) = pages.get_mut(&page_no) {
            f(page);
            Ok(())
        } else {
            // Driver may block due disk or network IO. Dropping lock before reading page.
            drop(pages);

            let mut data = Box::new([0; PAGE_SIZE]);
            if let Some(lsn) = self.driver.read_page(page_no, data.as_mut())? {
                let current_commit =
                    Arc::clone(&self.last_commit.lock().expect("No commit was given"));
                let mut last_commit_lsn = current_commit.lsn();
                let mut commit = current_commit.next.load_full();
                trace!(page_no, lsn, last_commit_lsn, "Implanting page");

                #[allow(clippy::single_range_in_vec_init)]
                let mut buf_mask = vec![0..PAGE_SIZE];

                while let Some(com) = commit.as_ref() {
                    if buf_mask.is_empty() || com.lsn() > lsn {
                        break;
                    }
                    last_commit_lsn = com.lsn();
                    apply_patches(
                        com.undo(),
                        page_no as usize * PAGE_SIZE,
                        data.as_mut(),
                        buf_mask.as_mut(),
                    );

                    commit = com.next.load_full();
                }
                assert!(
                    last_commit_lsn == lsn,
                    "Could not compensate page no. {} with lsn {}. Last commit LSN is {}",
                    page_no,
                    lsn,
                    last_commit_lsn
                );
            }

            let mut page = Page {
                data,
                lsn: self.last_commit.lock().unwrap().lsn(),
                dirty: false,
            };

            let mut pages = self.pages.lock().unwrap();
            f(&mut page);
            pages.insert(page_no, page);
            Ok(())
        }
    }

    fn with_page_opt(&self, page_no: PageNo, mut f: impl FnMut(&mut Page)) -> io::Result<bool> {
        let mut pages = self.pages.lock().unwrap();
        if let Some(page) = pages.get_mut(&page_no) {
            f(page);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn read(&self, addr: Addr, buf: &mut [u8]) -> io::Result<()> {
        let segments = PageSegments::new(addr, buf.len());
        for (addr, slice_range) in segments {
            let (page_no, offset) = split_ptr(addr);
            let offset = offset as usize;
            let len = slice_range.len();

            let buf_slice = &mut buf[slice_range];
            self.with_page(page_no, move |page| {
                let page_range = offset..offset + len;
                buf_slice.copy_from_slice(&page.data[page_range]);
            })?;
        }
        Ok(())
    }

    // TODO proptests
    fn write(&self, addr: Addr, buf: &[u8]) -> io::Result<()> {
        let segments = PageSegments::new(addr, buf.len());
        for (addr, slice_range) in segments {
            let (page_no, offset) = split_ptr(addr);
            let offset = offset as usize;
            let len = slice_range.len();

            self.with_page_opt(page_no, move |page| {
                let slice_range = slice_range.clone();
                let page_range = offset..offset + len;
                let page_buf = page.data.as_mut_slice();
                page_buf[page_range].copy_from_slice(&buf[slice_range]);
                page.dirty = true;
            })?;
        }
        Ok(())
    }

    fn zero(&self, addr: Addr, len: usize) -> io::Result<()> {
        let segments = PageSegments::new(addr, len);
        for (addr, slice_range) in segments {
            let (page_no, offset) = split_ptr(addr);
            let offset = offset as usize;
            let len = slice_range.len();

            self.with_page_opt(page_no, |page| {
                let page_range = offset..offset + len;
                let page_buf = page.data.as_mut_slice();
                page_buf[page_range].fill(0);
                page.dirty = true;
            })?;
        }
        Ok(())
    }

    fn flush(&self, commit: Arc<Commit>) -> io::Result<()> {
        let lsn = commit.lsn();
        let mut pages = self.pages.lock().unwrap();
        for (page_no, page) in pages.iter_mut() {
            if page.dirty {
                debug!(page_no, "Flushing page");
                self.driver.write_page(*page_no, page.data.as_ref(), lsn)?;
                page.dirty = false;
                page.lsn = lsn;
            }
        }
        *self.last_commit.lock().unwrap() = commit;
        self.driver.flush()
    }

    fn into_inner(self) -> Box<dyn PageDriver> {
        self.driver
    }

    /// Clears the loaded pages cache
    ///
    /// All subsequent reads will be done from the driver
    #[cfg(test)]
    fn clear_cache(&self) {
        self.pages.lock().unwrap().clear();
    }
}

struct Page {
    data: Box<[u8; PAGE_SIZE]>,
    dirty: bool,
    lsn: LSN,
}

#[derive(Clone)]
pub struct VolumeHandle {
    notify: CommitNotify,
    pages_count: PageNo,
    commit_log: Arc<Commit>,
    pages: Arc<Pages>,
}

impl VolumeHandle {
    /// Blocks until next snapshot is available and returns it
    ///
    /// The next snapshot is the one that is the most recent at the time this method is called.
    /// It might be several snapshots ahead of the last seen snapshot.
    pub fn wait(&mut self) -> Snapshot {
        // TODO: this is incorrect, we should return not the next, but latest snapshot
        let last_lsn = self.notify.last_seen_lsn();
        let commit = self.notify.next_commit();
        debug_assert!(commit.lsn() == last_lsn + 1);

        Snapshot {
            pages_count: self.pages_count,
            commit_log: Arc::clone(&self.notify.commit),
            pages: Arc::clone(&self.pages),
        }
    }

    pub fn current_commit(&self) -> &Commit {
        self.notify.commit.as_ref()
    }

    pub fn advance_to_latest(&mut self) -> u64 {
        self.notify.advance_to_latest()
    }

    pub fn wait_commit(&mut self) -> &Commit {
        self.notify.next_commit()
    }

    pub fn pages(&self) -> PageNo {
        self.pages_count
    }

    pub fn last_seen_lsn(&self) -> u64 {
        self.notify.last_seen_lsn()
    }

    pub fn snapshot(&mut self) -> Snapshot {
        while self.commit_log.next.load().is_some() {
            self.commit_log = self
                .commit_log
                .next
                .load_full()
                .as_ref()
                .as_ref()
                .unwrap()
                .clone();
        }

        let snapshot = Snapshot {
            pages_count: self.pages_count,
            commit_log: Arc::clone(&self.commit_log),
            pages: Arc::clone(&self.pages),
        };
        trace!(lsn = snapshot.lsn(), "Creating snapshot");
        snapshot
    }
}

#[derive(Clone)]
pub struct CommitNotify {
    // Last processed commit.
    commit: Arc<Commit>,
    last_seen_lsn: u64,

    /// Latest commit that is available for reading in the volume. May be way ahead of `commit`.
    latest_commit: Arc<(Mutex<Arc<Commit>>, Condvar)>,
    pages_count: PageNo,
}

impl CommitNotify {
    /// Blocks until next commit is available and returns it
    ///
    /// If several commits happened since the last call to this method, this function will return
    /// all of them in order.
    pub fn next_commit(&mut self) -> &Commit {
        // Pay attention to the order of operations and the fact that algorithm uses 2 commits:
        // 1. `commit` - the last processed commit by this method.
        // 2. `latest_commit` - the latest commit that is available for reading in the volume.
        //    `latest_commit` may be way ahead of `commit` if client of this function can't keep up with
        //    ongoing commits.
        //
        // Access to `commit` is synchronized by `Arc` and most of the times we don't need to acquire
        // the lock to check if there is a new commit available. We need to acquire global lock
        // if we are contending for the latest commit in the volume.
        let commit = Arc::clone(&self.commit);

        let mut lock = self.latest_commit.0.lock().unwrap();
        // 1. Need to check again after acquiring the lock, otherwise it is a race condition
        //    because we speculatively checked the condition before acquiring the lock to prevent
        //    contention when possible
        // 2. spourious wakeups are possible, so we need to check the condition in a loop
        while commit.next().is_none() {
            lock = self.latest_commit.1.wait(lock).unwrap();
        }

        let next_commit = commit.next.load_full().as_ref().as_ref().unwrap().clone();
        self.commit = Arc::clone(&next_commit);
        self.last_seen_lsn = self.commit.lsn();
        self.commit.as_ref()
    }

    /// Advances the `commit` to the latest commit available in the volume.
    fn advance_to_latest(&mut self) -> u64 {
        while self.commit.next.load().is_some() {
            self.commit = self
                .commit
                .next
                .load_full()
                .as_ref()
                .as_ref()
                .unwrap()
                .clone();
        }
        self.last_seen_lsn = self.commit.lsn();
        self.last_seen_lsn
    }

    pub fn last_seen_lsn(&self) -> u64 {
        self.last_seen_lsn
    }

    pub fn pages(&self) -> PageNo {
        // TODO is this correct? Can the number of pages change?
        self.pages_count
    }
}

/// Trait describing a read-only snapshot of a volume.
///
/// Represents a consistent snapshot of a volume at a specific point in time.
pub trait TxRead {
    /// Reads the specified number of bytes from the given address.
    ///
    /// # Panics
    /// Panic if the address is out of bounds. See [`Self::valid_range`] for bounds checking.
    fn read(&self, addr: Addr, len: usize) -> Cow<[u8]>;

    /// Checks if the specified range of addresses is valid.
    ///
    /// The address is not valid if it addresses the pages that are outside of the volume bounds.
    fn valid_range(&self, addr: Addr, len: usize) -> bool;
}

/// Trait describing a transaction that can modify data in a volume.
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

/// Represents a committed snapshot of a volume. Created by [`Volume::snapshot`] method.
///
/// A `Snapshot` captures the state of a volume at a specific point in time,
/// including any patches (modifications) that have been applied up to that point. It serves
/// as a read-only view into the historical state of the volume, allowing for consistent reads
/// of pages as they existed at the time of the snapshot.
///
/// Snapshots are following repeatable read isolation level, meaning that they are not affected
/// by any changes made to the volume after the snapshot was taken.
#[derive(Clone)]
pub struct Snapshot {
    pages_count: PageNo,
    commit_log: Arc<Commit>,
    pages: Arc<Pages>,
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

        self.pages.read(addr, &mut buf).unwrap();

        #[allow(clippy::single_range_in_vec_init)]
        let mut buf_mask = vec![0..len];

        // Apply own uncommitted changes from the given transaction
        apply_patches(uncommitted, addr as usize, &mut buf, &mut buf_mask);

        // Applying with undo log
        // We need to skip first entry in undo log, because it is describing how to undo the changes
        // of current snapshot itself.
        let mut commit_log = self.commit_log.as_ref().next.load_full();
        while let Some(commit) = commit_log.as_ref() {
            if buf_mask.is_empty() {
                break;
            };
            if commit.lsn() > self.lsn() {
                apply_patches(&commit.undo, addr as usize, &mut buf, &mut buf_mask);
            }
            commit_log = commit.next.load_full();
        }

        Cow::Owned(buf)
    }

    pub fn lsn(&self) -> LSN {
        self.commit_log.lsn()
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

/// `Commit` is referenced from old to new and only last position is stored in [`Volume`].
/// It enables automatic cleanup of old entries that are not referenced anymore by any snapshot or transaction.
pub struct Commit {
    /// A changes that have been applied in this commit.
    ///
    /// This array is sorted by the address of the change and all patches are non-overlapping.
    changes: Vec<Patch>,

    /// Undo patches of a commit
    ///
    /// Changes that need to be applied to volume to restore it to the state before this transaction
    /// was committed.
    ///
    /// As with `changes`, this array is sorted by the address of the change and all patches are non-overlapping.
    undo: Vec<Patch>,

    /// A reference to the next commit in the chain.
    ///
    /// This is updated atomically by the thread that executing commit operation.
    next: ArcSwap<Option<Arc<Commit>>>,

    /// A log sequence number (LSN) that uniquely identifies this commit.
    /// LSN numbers are monotonically increasing.
    lsn: LSN,
}

impl Commit {
    pub fn new(changes: Vec<Patch>, undo: Vec<Patch>, lsn: LSN) -> Self {
        Self {
            changes,
            undo,
            next: ArcSwap::from_pointee(None),
            lsn,
        }
    }

    pub fn patches(&self) -> &[Patch] {
        self.changes.as_slice()
    }

    pub fn undo(&self) -> &[Patch] {
        self.undo.as_slice()
    }

    pub fn lsn(&self) -> LSN {
        self.lsn
    }

    pub fn next(&self) -> Option<Arc<Commit>> {
        self.next.load_full().as_ref().as_ref().map(Arc::clone)
    }
}

/// Because this type forms a linked list, standard `Drop` implementation will cause stack overflow.
/// This implements iterative drop instead.
impl Drop for Commit {
    fn drop(&mut self) {
        // SAFETY:
        //  1. ArcSwap is owned and not impl Clone, so we're the only owner of it.
        //  2. We're in Drop, so there are no other references to self.
        // Therefore, we can safely move an `Arc` out of self.next.
        let mut next = self.next.swap(Arc::new(None));
        while let Some(Some(entry)) = Arc::into_inner(next) {
            // What we're doing here is making one more ref to the next entry. This effectively prevents
            // recursive drop(). 1 call to drop() will still be made, but it will return without recursion,
            // because current stack frame holding a ref to the `self.next.next`. This way we can remove
            // the whole list iteratively.
            next = entry.next.load_full();
        }
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
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tempfile::tempdir;

    use crate::driver::{FileDriver, TestPageDriver};

    use super::*;
    use std::thread;

    #[test]
    fn first_commit_should_have_lsn_1() {
        // In several places in codebase we assume that first commit has LSN 1
        // and LSN 0 is synthetic LSN used for base empty snapshot
        let mut mem = Volume::default();
        let tx = mem.start();
        assert_eq!(mem.commit(tx).unwrap(), 1);
    }

    #[test]
    fn non_linear_commits_must_be_rejected() {
        let mut mem = Volume::default();
        let mut tx1 = mem.start();
        let mut tx2 = mem.start();

        tx1.write(0, b"Hello");
        mem.commit(tx1).unwrap();

        tx2.write(0, b"World");
        assert!(mem.commit(tx2).is_err());
    }

    #[test]
    fn committed_changes_should_be_visible_on_a_page() {
        let mut mem = Volume::from("Jekyll");

        let mut tx = mem.start();
        tx.write(0, b"Hide");
        mem.commit(tx).unwrap();

        assert_str_eq(mem.read(0, 4), b"Hide");
    }

    #[test]
    fn committed_changes_should_be_visible_via_handle() {
        let mut mem = Volume::from("Jekyll");
        let mut handle = mem.handle();

        let mut tx = mem.start();
        tx.write(0, b"Hide");
        let expected_lsn = mem.commit(tx).unwrap();

        let s = handle.snapshot();
        assert_str_eq(s.read(0, 4), b"Hide");
        assert_eq!(s.lsn(), expected_lsn);
        // TODO
        // assert_eq!(handle.last_seen_lsn(), expected_lsn);
    }

    #[test]
    fn uncommitted_changes_should_be_visible_only_on_the_snapshot() {
        let mem = Volume::from("Jekyll");

        let mut tx = mem.start();
        tx.write(0, b"Hide");

        assert_str_eq(tx.read(0, 4), "Hide");
        assert_str_eq(mem.read(0, 6), "Jekyll");
    }

    #[test]
    fn snapshot_should_provide_repeatable_read_isolation() {
        let mut mem = Volume::default();

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
        let mut mem = Volume::from("Hello panic!");

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
        let mut mem = Volume::default();

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
        let mut mem = Volume::default();

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
        let mut mem = Volume::new_in_memory(2);

        // Choosing address so that data is split across 2 pages
        let addr = PAGE_SIZE as Addr - 2;
        let alice = b"Alice";

        let mut tx = mem.start();
        tx.write(addr, alice);

        // Checking that data is visible in snapshot
        assert_str_eq(tx.read(addr, alice.len()), alice);

        // Checking that data is visible after commit to volume
        mem.commit(tx).unwrap();
        assert_str_eq(mem.read(addr, alice.len()), alice);
    }

    #[test]
    fn data_can_be_read_from_snapshot() {
        let data = [1, 2, 3, 4, 5];
        let mem = Volume::from(data.as_slice());

        let tx = mem.start();
        assert_eq!(tx.read(0, 5), data.as_slice());
    }

    #[test]
    fn data_can_be_removed_on_snapshot() {
        let data = [1, 2, 3, 4, 5];
        let mem = Volume::from(data.as_slice());

        let mut tx = mem.start();
        tx.reclaim(1, 3);

        assert_eq!(tx.read(0, 5), [1u8, 0, 0, 0, 5].as_slice());
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_read() {
        // reading in a such a way that start address is still valid, but end address is not
        Volume::default().start().read(PAGE_SIZE as Addr - 10, 20);
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_write() {
        Volume::default()
            .start()
            .write(PAGE_SIZE as Addr - 10, [0; 20]);
    }

    #[test]
    #[should_panic(expected = "Address range is out of bounds")]
    fn must_panic_on_oob_reclaim() {
        Volume::default()
            .start()
            .reclaim(PAGE_SIZE as Addr - 10, 20);
    }

    /// When dropping transaction with a long commit log stackoverflow may happened if removed recursively.
    #[test]
    fn deep_commit_notify_should_not_cause_stack_overflow() {
        thread::Builder::new()
            .name("deep_snapshot_should_not_cause_stack_overflow".to_string())
            // setting small stacksize explicitly so it will be easier to reproduce a problem
            // and not to rely on the environment
            .stack_size(100 * 1024)
            .spawn(|| {
                let mut mem = Volume::new_in_memory(100);

                // Creating notify handle and a snapshot to check if in both cases
                // stack overflow is not happening
                let notify = mem.commit_notify();
                let tx = mem.start();

                for _ in 0..1000 {
                    mem.commit(mem.start()).unwrap();
                }

                // Explicitly dropping both, so they will not be eliminated before transactions are done
                drop(notify);
                drop(tx);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn data_can_be_read_from_a_different_thread() {
        let mut volume = Volume::default();

        let mut handle = volume.handle();

        let lsn = thread::spawn(move || {
            let mut tx = volume.start();
            tx.write(0, [1, 2, 3, 4]);
            volume.commit(tx).unwrap()
        });

        let s1 = handle.wait();
        let lsn = lsn.join().unwrap();
        let s2 = handle.snapshot();

        assert_eq!(s1.lsn(), lsn);
        assert_eq!(&*s1.read(0, 4), [1, 2, 3, 4]);

        assert_eq!(s2.lsn(), lsn);
        assert_eq!(&*s2.read(0, 4), [1, 2, 3, 4]);
    }

    #[test]
    fn can_get_notifications_about_commit() {
        let mut volume = Volume::default();
        let mut n1 = volume.commit_notify();
        let mut n2 = volume.commit_notify();

        let lsn = thread::spawn(move || {
            let mut tx = volume.start();
            tx.write(0, [1]);
            volume.commit(tx).unwrap()
        });

        let lsn2 = n2.next_commit().lsn();
        let lsn1 = n1.next_commit().lsn();

        let lsn = lsn.join().unwrap();

        assert_eq!(lsn1, lsn);
        assert_eq!(lsn2, lsn);
    }

    #[test]
    fn each_commit_snapshot_can_be_addressed() {
        let mut volume = Volume::default();
        let mut notify = volume.commit_notify();

        for i in 1..=10 {
            let mut tx = volume.start();
            tx.write(i, [i as u8]);
            volume.commit(tx).unwrap();
        }

        for lsn in 1..=10 {
            let commit = notify.next_commit();
            assert_eq!(lsn, commit.lsn());
        }
    }

    #[test]
    fn can_wait_for_a_snapshot_in_a_thread() {
        let mut volume = Volume::default();
        let mut notify = volume.commit_notify();

        let lsn = thread::spawn(move || notify.next_commit().lsn());

        let mut tx = volume.start();
        tx.write(0, [0]);
        let expected_lsn = volume.commit(tx).unwrap();

        let lsn = lsn.join().unwrap();
        assert_eq!(lsn, expected_lsn);
    }

    #[test]
    fn commit_stress_test() {
        const ITERATIONS: usize = 1000;

        let mut volume = Volume::default();
        let mut notify = volume.commit_notify();

        let handle = thread::spawn(move || {
            for _ in 0..ITERATIONS {
                let mut tx = volume.start();
                tx.write(0, [0]);
                volume.commit(tx).unwrap();
            }
        });

        for i in 1..=ITERATIONS {
            let lsn = notify.next_commit().lsn();
            assert_eq!(lsn, i as u64);
        }

        handle.join().unwrap();
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

    #[test]
    #[ignore = "temporary disabled. LSN should be restored after reopen"] // TODO
    fn page_volume_can_be_restored_after_reopen() -> io::Result<()> {
        let tempdir = tempdir()?;
        let db_file = tempdir.path().join("test.db");
        let mut volume = Volume::new_with_driver(1, FileDriver::from_file(&db_file)?);

        let mut tx = volume.start();
        tx.write(0, [42]);
        volume.commit(tx).unwrap();

        let volume = Volume::new_with_driver(1, FileDriver::from_file(&db_file)?);
        assert_eq!(&*volume.read(0, 1), [42]);
        Ok(())
    }

    /// This tests writes random data to the volume in a way that the total sum of all bytes is always 0.
    /// It then checks that the sum is still 0 when observing volume snapshots from a different thread.
    #[test]
    fn consistency_check() -> io::Result<()> {
        const PAGES: usize = 2;
        const SIZE: usize = PAGE_SIZE * PAGES;
        const PATCH_LEN: usize = 100;
        const TRANSACTIONS: usize = 100;
        const THREADS: usize = 20;

        fn wrapping_sum(tx: &impl TxRead) -> u8 {
            tx.read(0, SIZE)
                .iter()
                .copied()
                .reduce(|a, b| a.wrapping_add(b))
                .unwrap()
        }

        fn check_transaction_consistency(mut handle: VolumeHandle) -> io::Result<()> {
            for i in 1..=TRANSACTIONS {
                let snapshot = handle.wait();
                assert_eq!(i as LSN, snapshot.lsn());
                let sum = wrapping_sum(&snapshot);
                if sum != 0 {
                    let lsn = snapshot.lsn();
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Wrapping sum is {} on LSN: {} (should be 0)", sum, lsn),
                    ));
                }
            }
            Ok(())
        }

        let mut rng = SmallRng::from_entropy();
        let mut patch = vec![0u8; PATCH_LEN];

        let mut volume = Volume::new_in_memory(PAGES as PageNo);
        let mut join_handles = vec![];

        // Spawn threads that will check the consistency of the volume snapshots.
        for _ in 0..THREADS {
            let handle = volume.handle();
            let join = thread::spawn(move || check_transaction_consistency(handle));
            join_handles.push(join);
        }

        // Write random patches to the volume and correct the first byte to keep the sum 0.
        for _ in 0..TRANSACTIONS {
            let offset = rng.gen_range(0..SIZE - PATCH_LEN);
            rng.fill(patch.as_mut_slice());

            let mut tx = volume.start();
            tx.write(offset as Addr, patch.as_slice());
            let sum = wrapping_sum(&tx);
            let delta = 255 - sum.wrapping_sub(tx.read(0, 1)[0]).wrapping_sub(1);
            tx.write(0, [delta]);
            volume.commit(tx)?;
        }

        for join in join_handles {
            join.join().unwrap()?;
        }

        Ok(())
    }

    #[test]
    fn check_pages_read_write() -> io::Result<()> {
        let pages = Pages::new(NoDriver::default(), commit_log(&[]));

        let mut buf = vec![0; 5];
        pages.read(0, &mut buf)?;
        assert_eq!(buf, vec![0; 5]);

        pages.write(0, &[1, 2, 3, 4, 5])?;

        pages.read(0, &mut buf)?;
        assert_eq!(buf, &[1, 2, 3, 4, 5]);

        Ok(())
    }

    #[test]
    fn check_implant_page() -> io::Result<()> {
        let driver = TestPageDriver {
            pages: vec![
                (3, [3; PAGE_SIZE]), // LSN + PageContent
            ],
        };

        let commit = commit_log(&[
            &[Patch::Write(0, vec![1, 1, 1])],
            &[Patch::Write(0, vec![2, 2, 2])],
            &[Patch::Write(0, vec![3, 3, 3])],
        ]);

        let mut buf = [0; 3];

        let pages = Pages::new(driver, Arc::clone(&commit));

        pages.read(0, &mut buf)?;
        assert_eq!(buf, [0, 0, 0]);

        let commit = commit.next();
        pages.flush(commit.as_ref().unwrap().clone())?;
        pages.clear_cache();
        pages.read(0, &mut buf)?;
        assert_eq!(buf, [1, 1, 1]);

        let commit = commit.unwrap().next();
        pages.flush(commit.as_ref().unwrap().clone())?;
        pages.clear_cache();
        pages.read(0, &mut buf)?;
        assert_eq!(buf, [2, 2, 2]);

        let commit = commit.unwrap().next();
        pages.flush(commit.as_ref().unwrap().clone())?;
        pages.clear_cache();
        pages.read(0, &mut buf)?;
        assert_eq!(buf, [3, 3, 3]);

        Ok(())
    }

    #[test]
    fn volume_can_be_created_from_initial_commit() {
        let commit_log = commit_log(&[
            &[Patch::Write(0, vec![1, 2, 3])],
            &[Patch::Write(3, vec![4, 5, 6])],
        ]);

        // Skipping commit 0 (initial) and 1
        // TODO move to Commit::advance()/Commit::latest()
        let commit = commit_log.next().as_ref().unwrap().next();
        drop(commit_log);
        let commit = Arc::into_inner(commit.unwrap()).expect("Unable to take ownership of commit");
        println!("Commit LSN: {}", commit.lsn());

        let volume = Volume::from_commit(1, commit, NoDriver::default());
        assert_eq!(volume.snapshot().lsn(), 2);
    }

    fn commit_log(changes: &[&[Patch]]) -> Arc<Commit> {
        let mut volume = Volume::new_in_memory(1);
        // Creating a snapshot before the changes are applied
        // so that it will accumulate the commit log.
        let snapshot = volume.snapshot();

        for change in changes {
            let mut tx = volume.start();
            for patch in *change {
                match patch {
                    Patch::Write(addr, bytes) => tx.write(*addr, bytes.as_slice()),
                    Patch::Reclaim(_, _) => todo!(),
                };
            }
            volume.commit(tx).unwrap();
        }

        snapshot.commit_log
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

    mod proptests {
        use super::*;
        use proptest::{collection::vec, prelude::*};
        use NormalizedPatches::*;

        /// The size of database for testing
        const DB_SIZE: usize = PAGE_SIZE * 2;

        proptest! {
            #[test]
            fn undo_log_is_consistent_with_previous_snapshot_content(patches in vec(patches::any_patch(), 10)) {
                let mut v = Volume::with_capacity(DB_SIZE);
                let mut commit_notify = v.commit_notify();

                for patch in patches {
                    let snapshot = v.snapshot();
                    let prev = patch.read_corresponding_segment(&snapshot);

                    let mut tx = v.start();
                    patch.write_to(&mut tx);
                    v.commit(tx)?;

                    let commit = commit_notify.next_commit();
                    prop_assert_eq!(commit.undo.len(), 1);
                    let Patch::Write(_, bytes) = &commit.undo[0] else {
                        panic!("Invalid patch type")
                    };

                    prop_assert_eq!(bytes, &*prev);
                }
            }

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
                let mut mem = Volume::with_capacity(DB_SIZE);

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
                let mut mem = Volume::from(initial.as_slice());
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

    impl From<&str> for Volume {
        fn from(value: &str) -> Self {
            Self::from(value.as_bytes())
        }
    }

    impl From<&[u8]> for Volume {
        fn from(value: &[u8]) -> Self {
            let capactiy = PAGE_SIZE.max(value.len());
            let mut volume = Volume::with_capacity(capactiy);
            let mut tx = volume.start();
            tx.write(0, value);
            volume.commit(tx).unwrap();
            volume
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
