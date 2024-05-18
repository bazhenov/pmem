use std::ops::Deref;
use std::{borrow::Cow, cell::RefCell, ops::Range, rc::Rc, usize};

const PAGE_SIZE: usize = 1 << 24; // 16Mb
type Patch = (usize, Vec<u8>);
pub type Addr = u32;
pub type PageOffset = u32;
pub type PageNo = u32;
pub type LSN = u64;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("Memory not allocated")]
    NotAllocated,
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Default)]
pub struct Page {
    snapshot: Rc<CommitedSnapshot>,
}

impl Page {
    #[cfg(test)]
    fn read(&self, addr: PageOffset, len: PageOffset) -> Cow<'_, [u8]> {
        self.snapshot.read(addr, len)
    }
}

impl From<CommitedSnapshot> for Page {
    fn from(snapshot: CommitedSnapshot) -> Self {
        Self {
            snapshot: Rc::new(snapshot),
        }
    }
}

#[derive(Default)]
pub struct PagePool {
    page: Rc<RefCell<Page>>,
    lsn: LSN,
    max_page_no: PageNo,
}

impl PagePool {
    pub fn new(pages: usize) -> Self {
        Self {
            page: Rc::default(),
            lsn: 0,
            max_page_no: (pages - 1) as PageNo,
        }
    }
    pub fn snapshot(&self) -> Snapshot {
        let page = RefCell::borrow(&self.page);
        Snapshot {
            patches: vec![],
            base: Rc::clone(&page.snapshot),
        }
    }

    pub fn commit(&mut self, snapshot: Snapshot) {
        // Given snapshot should have current snapshot as a parent
        // Otherwise changes are not linear
        let mut page = RefCell::borrow_mut(&self.page);
        assert!(
            Rc::ptr_eq(&page.snapshot, &snapshot.base),
            "Proposed snaphot is not linear"
        );
        self.lsn += 1;
        page.snapshot = Rc::new(CommitedSnapshot {
            patches: snapshot.patches,
            parent: Some(Rc::clone(&page.snapshot)),
        });
    }

    #[cfg(test)]
    fn read(&self, addr: PageOffset, len: PageOffset) -> Result<Ref> {
        use crate::ensure;

        let (page_no, offset) = split_ptr(addr);
        ensure!(page_no <= self.max_page_no, Error::NotAllocated);

        let page = RefCell::borrow(&self.page);
        let snapshot = Rc::clone(&page.snapshot);
        Ok(Ref::create(snapshot, offset, len))
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
    fn create(snapshot: Rc<CommitedSnapshot>, offset: PageOffset, len: PageOffset) -> Self {
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

#[derive(Default)]
pub struct CommitedSnapshot {
    patches: Vec<Patch>,
    parent: Option<Rc<CommitedSnapshot>>,
}

impl CommitedSnapshot {
    pub fn read(&self, addr: PageOffset, len: PageOffset) -> Cow<'_, [u8]> {
        assert!(len <= PAGE_SIZE as u32, "Out of bounds read");
        let mut buffer = vec![0; len as usize];
        let range = (addr as usize)..(addr + len) as usize;

        let mut patch_list = vec![self.patches.as_slice()];
        let snapshots = self
            .parent
            .as_ref()
            .map(Rc::clone)
            .map(collect_snapshots)
            .unwrap_or_default();
        patch_list.extend(snapshots.iter().map(|s| s.patches.as_slice()));

        apply_patches(&patch_list, &mut buffer, &range);
        Cow::Owned(buffer)
    }

    pub fn read_ref<'a>(self: &Rc<Self>, addr: PageOffset, len: PageOffset) -> Ref {
        Ref::create(Rc::clone(self), addr, len)
    }
}

pub struct Snapshot {
    patches: Vec<Patch>,
    base: Rc<CommitedSnapshot>,
}

impl Snapshot {
    pub fn write(&mut self, addr: Addr, bytes: &[u8]) {
        assert!(bytes.len() <= PAGE_SIZE, "Buffer too large");
        self.patches.push((addr as usize, bytes.to_vec()))
    }

    pub fn read(&self, addr: PageOffset, len: PageOffset) -> Cow<'_, [u8]> {
        assert!(len <= PAGE_SIZE as u32, "Out of bounds read");
        let mut buffer = vec![0; len as usize];
        let range = (addr as usize)..(addr + len) as usize;

        // We need to collect a chain of snapshots into a Vec first.
        // Otherwise borrowcher is unable to reason about lifecycles
        let snapshots = collect_snapshots(Rc::clone(&self.base));
        let mut patch_list = vec![self.patches.as_slice()];
        patch_list.extend(snapshots.iter().map(|s| s.patches.as_slice()));

        apply_patches(&patch_list, &mut buffer, &range);
        Cow::Owned(buffer)
    }
}

fn collect_snapshots(snapshot: Rc<CommitedSnapshot>) -> Vec<Rc<CommitedSnapshot>> {
    let mut snapshots = vec![];
    let mut snapshot = Some(snapshot);
    while let Some(s) = snapshot {
        snapshots.push(Rc::clone(&s));
        snapshot = s.parent.as_ref().map(Rc::clone);
    }
    snapshots
}

fn apply_patches(snapshots: &[&[Patch]], buffer: &mut [u8], range: &Range<usize>) {
    for patches in snapshots.iter().rev() {
        for (offset, bytes) in patches.iter().filter(|p| intersects(p, range)) {
            // Calculating intersection of the path and input interval
            let start = range.start.max(*offset);
            let end = range.end.min(offset + bytes.len());
            let len = end - start;

            let patch_range = {
                let from = start.saturating_sub(*offset);
                from..from + len
            };

            let slice_range = {
                let from = start.saturating_sub(range.start);
                from..from + len
            };

            buffer[slice_range].copy_from_slice(&bytes[patch_range])
        }
    }
}

fn split_ptr(addr: Addr) -> (PageNo, PageOffset) {
    const PAGE_SIZE_BITS: u32 = PAGE_SIZE.trailing_zeros() as u32;
    let page_no = addr >> PAGE_SIZE_BITS;
    let offset = addr & ((PAGE_SIZE - 1) as u32);
    (page_no, offset)
}

/// Returns true of given patch intersects given range of bytes
fn intersects((offset, patch): &Patch, range: &Range<usize>) -> bool {
    *offset < range.end && offset + patch.len() > range.start
}

#[cfg(test)]
mod tests {
    pub use super::*;

    #[test]
    fn create_new_page() -> Result<()> {
        let page = Page::from("foo");
        assert_str_eq(page.read(0, 3), "foo");
        assert_str_eq(page.read(3, 1), [0]);
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

        assert_str_eq(snapshot.read(0, 4), "Hide");
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
    fn page_pool_should_return_error_of_ptr_out_of_bounds() -> Result<()> {
        let mem = PagePool::default();

        let Err(Error::NotAllocated) = mem.read(PAGE_SIZE as u32, 1) else {
            panic!("NotAllocated should be geberated");
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
            fn arbitrary_page_patches(snapshots in vec(any_snapshot(), 0..5)) {
                // Mirror buffer where we track all the patches being applied.
                // In the end page content should be equal mirror buffer
                let mut mirror = vec![0; PAGE_SIZE];
                let mut mem = PagePool::default();

                for patches in snapshots {
                    for (offset, bytes) in patches.into_iter() {
                        let mut snapshot = mem.snapshot();
                        snapshot.write(offset as u32, &bytes);

                        let range = offset..offset + bytes.len();
                        mirror[range].copy_from_slice(bytes.as_slice());
                        mem.commit(snapshot);
                    }
                }

                assert_eq!(&*mem.read(0, PAGE_SIZE as PageOffset).unwrap(), mirror);
            }
        }

        fn any_patch() -> impl Strategy<Value = Patch> {
            (0usize..PAGE_SIZE, vec(any::<u8>(), 1..32))
                .prop_filter("out of bounds patch", |(offset, bytes)| {
                    offset + bytes.len() < PAGE_SIZE
                })
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

    impl<T: AsRef<str>> From<T> for Page {
        fn from(value: T) -> Self {
            let bytes = value.as_ref().as_bytes();
            assert!(bytes.len() <= PAGE_SIZE, "String is too large");

            let snapshot = CommitedSnapshot {
                patches: vec![(0, bytes.to_vec())],
                parent: None,
            };
            Page {
                snapshot: Rc::new(snapshot),
            }
        }
    }

    impl<T: AsRef<str>> From<T> for PagePool {
        fn from(value: T) -> Self {
            let page = Rc::new(RefCell::new(Page::from(value)));
            PagePool {
                page,
                lsn: 0,
                max_page_no: 0,
            }
        }
    }
}
