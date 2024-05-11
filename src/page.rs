use std::{borrow::Cow, ops::Range, rc::Rc};

const PAGE_SIZE: usize = 1 << 16; // 64KB

pub struct Page {
    data: [u8; PAGE_SIZE],
    patches: Vec<(usize, Vec<u8>)>,
    commited_patches: Vec<(usize, Vec<u8>)>,
}

impl Page {
    pub fn new() -> Self {
        Self {
            data: [0; PAGE_SIZE],
            patches: vec![],
            commited_patches: vec![],
        }
    }

    pub fn as_bytes_mut(&mut self, idx: Range<usize>) -> &mut [u8] {
        assert!(
            idx.len() <= PAGE_SIZE && idx.end < PAGE_SIZE,
            "Out of bounds write"
        );
        self.patches.push((idx.start, vec![0; idx.len()]));
        let (_, patch) = self.patches.last_mut().unwrap();
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
        assert!(
            range.len() <= PAGE_SIZE && range.end < PAGE_SIZE,
            "Out of bounds read"
        );
        let mut slice = vec![0; range.len()];
        slice.copy_from_slice(&self.data[range.clone()]);

        for (offset, patch) in &self.commited_patches {
            if range.contains(offset) {
                // Patch start is in range
                let from = offset - range.start;
                let len = patch.len().min(range.end - offset);
                slice[from..(from + len)].copy_from_slice(&patch[..len])
            } else if range.contains(&(offset + patch.len())) {
                // Patch end is in range
                let from = range.start - offset;
                let len = range.len().min(patch.len() - from);
                slice[..len].copy_from_slice(&patch[from..from + len])
            } else if *offset < range.start && range.end < offset + patch.len() {
                // Patch is fully covering slice
                let from = range.start - offset;
                slice.copy_from_slice(&patch[from..from + range.len()])
            }
        }

        Cow::Owned(slice)
    }

    pub fn as_bytes_uncommited(&self, range: Range<usize>) -> Cow<[u8]> {
        assert!(
            range.len() <= PAGE_SIZE && range.end < PAGE_SIZE,
            "Out of bounds read"
        );
        let mut slice = vec![0; range.len()];
        slice.copy_from_slice(&self.data[range.clone()]);

        let patches = self.patches.iter().chain(self.commited_patches.iter());
        for (offset, patch) in patches {
            if range.contains(offset) {
                // Patch start is in range
                let from = offset - range.start;
                let len = patch.len().min(range.end - offset);
                slice[from..(from + len)].copy_from_slice(&patch[..len])
            } else if range.contains(&(offset + patch.len())) {
                // Patch end is in range
                let from = range.start - offset;
                let len = range.len().min(patch.len() - from);
                slice[..len].copy_from_slice(&patch[from..from + len])
            } else if *offset < range.start && range.end < offset + patch.len() {
                // Patch is fully covering slice
                let from = range.start - offset;
                slice.copy_from_slice(&patch[from..from + range.len()])
            }
        }

        Cow::Owned(slice)
    }

    pub fn commit(&mut self) {
        self.commited_patches.extend(self.patches.drain(..));
    }
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
                commited_patches: vec![],
            }
        }
    }
}
