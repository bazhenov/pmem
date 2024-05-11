use std::{borrow::Cow, ops::Range};

const PAGE_SIZE: usize = 1 << 16; // 64KB
type Patch = (usize, Vec<u8>);

pub struct Page {
    data: [u8; PAGE_SIZE],
    uncommited: Vec<Patch>,
    patches: Vec<Patch>,
}

impl Page {
    pub fn new() -> Self {
        Self {
            data: [0; PAGE_SIZE],
            uncommited: vec![],
            patches: vec![],
        }
    }

    pub fn as_bytes_mut(&mut self, idx: Range<usize>) -> &mut [u8] {
        assert!(
            0 < idx.len() && idx.len() <= PAGE_SIZE,
            "idx range should be at least 1 byte ({:?})",
            idx
        );
        assert!(idx.end < PAGE_SIZE, "idx.end out of page bounds");
        self.uncommited.push((idx.start, vec![0; idx.len()]));
        let (_, patch) = self.uncommited.last_mut().unwrap();
        patch.as_mut_slice()
    }

    pub fn read_bytes<const N: usize>(&self, offset: usize) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.as_bytes_uncommited(offset..offset + N);
        for (to, from) in ret.iter_mut().zip(bytes.into_iter()) {
            *to = *from;
        }
        ret
    }

    pub fn as_bytes(&self, range: Range<usize>) -> Cow<[u8]> {
        self.as_bytes_with_patches(range, self.patches.iter())
    }

    pub fn as_bytes_uncommited(&self, range: Range<usize>) -> Cow<[u8]> {
        let patches = self.patches.iter().chain(self.uncommited.iter());
        self.as_bytes_with_patches(range, patches)
    }

    fn as_bytes_with_patches<'a>(
        &self,
        range: Range<usize>,
        patches: impl Iterator<Item = &'a Patch>,
    ) -> Cow<[u8]> {
        assert!(
            range.len() <= PAGE_SIZE && range.end <= PAGE_SIZE,
            "Out of bounds read"
        );
        let mut slice = vec![0; range.len()];
        slice.copy_from_slice(&self.data[range.clone()]);

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

    pub fn commit(&mut self) {
        self.patches.extend(self.uncommited.drain(..));
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
    fn uncommited_changes_should_not_be_visible_via_as_bytes_mut() {
        let mut page = Page::from("Jekyll");
        page.as_bytes_mut(0..4).copy_from_slice(b"Hide");
        assert_eq!(&*page.as_bytes(0..6), b"Jekyll");
    }

    #[test]
    fn uncommited_changes_should_not_be_visible_via_as_bytes_mut_uncommited() {
        let mut page = Page::from("Jekyll");
        page.as_bytes_mut(0..4).copy_from_slice(b"Hide");
        assert_eq!(&*page.as_bytes_uncommited(0..4), b"Hide");
    }

    #[test]
    fn patch_page() {
        let mut page = Page::from("Hello panic!");

        page.as_bytes_mut(6..11).copy_from_slice(b"world");

        page.commit();

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
            let mut data = [0; PAGE_SIZE];

            data[..bytes.len()].copy_from_slice(bytes);
            Self {
                data,
                patches: vec![],
                uncommited: vec![],
            }
        }
    }

    mod proptests {
        use super::super::*;
        use proptest::{collection::vec, prelude::*};
        use std::ops::Deref;

        proptest! {
            #[test]
            fn arbitrary_page_patches(snapshots in vec(any_snapshot(), 0..5)) {
                // Mirror buffer where we track all the patches being applied
                // in the end page content should be equal mirror buffer
                let mut mirror = [0; PAGE_SIZE];
                let mut page = Page::new();

                for patches in snapshots {
                    for (offset, bytes) in patches {
                        let range = offset..offset + bytes.len();
                        page.as_bytes_mut(range.clone()).copy_from_slice(bytes.as_slice());
                        mirror[range].copy_from_slice(bytes.as_slice());
                    }
                    page.commit();
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
