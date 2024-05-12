use std::{
    borrow::Cow,
    ops::{Add, Range},
};

const PAGE_SIZE: usize = 1 << 16; // 64KB
type Patch = (usize, Vec<u8>);
pub type Addr = u32;

pub struct Page {
    snapshots: Vec<Snapshot>,
}

impl Page {
    pub fn new() -> Self {
        Self { snapshots: vec![] }
    }

    pub fn as_bytes(&self, range: Range<usize>) -> Cow<[u8]> {
        as_bytes_with_patches(range, self.snapshots.iter().flat_map(|s| s.patches.iter()))
    }

    pub fn as_bytes_uncommited(&self, range: Range<usize>, snapshot: &Snapshot) -> Cow<'_, [u8]> {
        let patches = self
            .snapshots
            .iter()
            .flat_map(|s| s.patches.iter())
            .chain(snapshot.patches.iter());
        as_bytes_with_patches(range, patches)
    }

    pub fn commit(&mut self, snapshot: Snapshot) {
        self.snapshots.push(snapshot);
    }
}

impl From<Snapshot> for Page {
    fn from(value: Snapshot) -> Self {
        Self {
            snapshots: vec![value],
        }
    }
}

fn as_bytes_with_patches<'a, 'b>(
    range: Range<usize>,
    patches: impl Iterator<Item = &'a Patch>,
) -> Cow<'b, [u8]> {
    assert!(
        range.len() <= PAGE_SIZE && range.end <= PAGE_SIZE,
        "Out of bounds read"
    );
    let mut slice = vec![0; range.len()];

    for (offset, bytes) in patches.filter(|p| intersects(p, &range)) {
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

        slice[slice_range].copy_from_slice(&bytes[patch_range])
    }

    Cow::Owned(slice)
}

#[derive(Default)]
pub struct Snapshot {
    patches: Vec<Patch>,
}

impl Snapshot {
    pub fn zeroed(&mut self, idx: Range<usize>) -> &mut [u8] {
        assert!(
            0 < idx.len() && idx.len() <= PAGE_SIZE,
            "idx range should be at least 1 byte ({:?})",
            idx
        );
        assert!(idx.end < PAGE_SIZE, "idx.end out of page bounds");
        self.patches.push((idx.start, vec![0; idx.len()]));
        let (_, patch) = self.patches.last_mut().unwrap();
        patch.as_mut_slice()
    }

    pub fn write(&mut self, addr: Addr, bytes: &[u8]) {
        assert!(bytes.len() <= PAGE_SIZE, "Buffer too large");
        self.patches.push((addr as usize, bytes.to_vec()))
    }
}

impl From<Patch> for Snapshot {
    fn from(value: Patch) -> Self {
        Self {
            patches: vec![value],
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
        assert_eq!(&*page.as_bytes(0..3), b"foo");
        assert_eq!(&*page.as_bytes(3..4), [0]);
    }

    #[test]
    fn commited_changes_should_be_visible_via_commit() {
        let mut page = Page::from("Jekyll");

        let mut snapshot = Snapshot::default();
        snapshot.write(0, b"Hide");

        page.commit(snapshot);
        assert_eq!(&*page.as_bytes(0..4), b"Hide");
    }

    #[test]
    fn uncommited_changes_should_be_visible_via_as_bytes_uncommited() {
        let mut page = Page::from("Jekyll");
        let mut snapshot = Snapshot::default();
        snapshot.write(0, b"Hide");

        assert_eq!(&*page.as_bytes_uncommited(0..4, &snapshot), b"Hide");
    }

    #[test]
    fn patch_page() {
        let mut page = Page::from("Hello panic!");

        let mut snapshot = Snapshot::default();
        snapshot.write(6, b"world");

        page.commit(snapshot);

        assert_eq!(&*page.as_bytes(0..12), b"Hello world!");
        assert_eq!(&*page.as_bytes(0..8), b"Hello wo");
        assert_eq!(&*page.as_bytes(3..12), b"lo world!");
        assert_eq!(&*page.as_bytes(6..11), b"world");
        assert_eq!(&*page.as_bytes(8..12), b"rld!");
        assert_eq!(&*page.as_bytes(7..10), b"orl");
    }

    impl<T: AsRef<str>> From<T> for Page {
        fn from(value: T) -> Self {
            let bytes = value.as_ref().as_bytes();
            assert!(bytes.len() <= PAGE_SIZE, "String is too large");

            Page::from(Snapshot::from((0, bytes.to_vec())))
        }
    }

    mod proptests {
        use super::super::*;
        use proptest::{collection::vec, prelude::*};
        use std::ops::Deref;

        proptest! {
            #[test]
            fn arbitrary_page_patches(snapshots in vec(any_snapshot(), 0..5)) {
                // Mirror buffer where we track all the patches being applied.
                // In the end page content should be equal mirror buffer
                let mut mirror = [0; PAGE_SIZE];
                let mut page = Page::new();

                for patches in snapshots {
                    for (offset, bytes) in patches {
                        let mut snapshot = Snapshot::default();
                        snapshot.write(offset as u32, &bytes);

                        let range = offset..offset + bytes.len();
                        mirror[range].copy_from_slice(bytes.as_slice());
                        page.commit(snapshot);
                    }
                }

                assert_eq!(page.as_bytes(0..PAGE_SIZE).deref(), mirror);
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
}
