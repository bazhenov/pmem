use std::{borrow::Cow, ops::Range, rc::Rc};

const PAGE_SIZE: usize = 1 << 16; // 64KB
type Patch = (usize, Vec<u8>);
pub type Addr = u32;
pub type PageOffset = u32;
pub type LSN = u64;

pub struct Page {
    snapshot: Rc<CommitedSnapshot>,
}

impl Page {
    pub fn new() -> Self {
        Self {
            snapshot: Rc::default(),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            patches: vec![],
            base: Rc::clone(&self.snapshot),
        }
    }

    pub fn commit(&mut self, snapshot: Snapshot, lsn: LSN) {
        // Given snapshot should have current snapshot as a parent
        // Otherwise changes are not linear
        assert!(
            Rc::ptr_eq(&self.snapshot, &snapshot.base),
            "Proposed snaphot is not linear"
        );
        assert!(lsn > self.snapshot.lsn, "New LSN should be larger");
        self.snapshot = Rc::new(CommitedSnapshot {
            lsn,
            patches: snapshot.patches,
            parent: Some(Rc::clone(&self.snapshot)),
        });
    }

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
pub struct CommitedSnapshot {
    lsn: LSN,
    patches: Vec<Patch>,
    parent: Option<Rc<CommitedSnapshot>>,
}

impl CommitedSnapshot {
    #[cfg(test)]
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

/// Returns true of given patch intersects given range of bytes
fn intersects((offset, patch): &Patch, range: &Range<usize>) -> bool {
    *offset < range.end && offset + patch.len() > range.start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_new_page() {
        let page = Page::from("foo");
        assert_str_eq(page.read(0, 3), "foo");
        assert_str_eq(page.read(3, 1), [0]);
    }

    #[test]
    fn commited_changes_should_be_visible_on_a_page() {
        let mut page = Page::from("Jekyll");
        let mut snapshot = page.snapshot();
        snapshot.write(0, b"Hide");
        page.commit(snapshot, 2);
        assert_str_eq(page.read(0, 4), b"Hide");
    }

    #[test]
    fn uncommited_changes_should_be_visible_only_on_the_snapshot() {
        let page = Page::from("Jekyll");
        let mut snapshot = page.snapshot();

        snapshot.write(0, b"Hide");
        assert_str_eq(snapshot.read(0, 4), "Hide");
        assert_str_eq(page.read(0, 6), "Jekyll");
    }

    #[test]
    fn patch_page() {
        let mut page = Page::from("Hello panic!");

        let mut snapshot = page.snapshot();
        snapshot.write(6, b"world");
        page.commit(snapshot, 2);

        assert_str_eq(page.read(0, 12), "Hello world!");
        assert_str_eq(page.read(0, 8), "Hello wo");
        assert_str_eq(page.read(3, 9), "lo world!");
        assert_str_eq(page.read(6, 5), "world");
        assert_str_eq(page.read(8, 4), "rld!");
        assert_str_eq(page.read(7, 3), "orl");
    }

    fn as_bytes(page: &Page, addr: PageOffset, len: PageOffset) -> Cow<[u8]> {
        page.read(addr, len)
    }

    impl<T: AsRef<str>> From<T> for Page {
        fn from(value: T) -> Self {
            let bytes = value.as_ref().as_bytes();
            assert!(bytes.len() <= PAGE_SIZE, "String is too large");

            let mut page = Page::new();
            let mut snapshot = page.snapshot();
            snapshot.write(0, bytes);
            page.commit(snapshot, 1);

            page
        }
    }

    mod proptests {
        use super::super::*;
        use super::*;
        use proptest::{collection::vec, prelude::*};

        proptest! {
            #[test]
            fn arbitrary_page_patches(snapshots in vec(any_snapshot(), 0..5)) {
                // Mirror buffer where we track all the patches being applied.
                // In the end page content should be equal mirror buffer
                let mut mirror = [0; PAGE_SIZE];
                let mut page = Page::new();

                let mut lsn = 1;
                for patches in snapshots {
                    for (offset, bytes) in patches.into_iter() {
                        let mut snapshot = page.snapshot();
                        snapshot.write(offset as u32, &bytes);

                        let range = offset..offset + bytes.len();
                        mirror[range].copy_from_slice(bytes.as_slice());
                        page.commit(snapshot, lsn);
                        lsn += 1;
                    }
                }

                assert_eq!(&*as_bytes(&page, 0, PAGE_SIZE as PageOffset), mirror);
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
}
