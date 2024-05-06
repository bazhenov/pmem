use std::{borrow::Cow, ops::Range, rc::Rc};

const PAGE_SIZE: usize = 1 << 16; // 64KB

pub struct Page {
    data: [u8; PAGE_SIZE],
}

impl Page {
    pub fn new() -> Self {
        Self {
            data: [0; PAGE_SIZE],
        }
    }

    fn from_str(string: impl AsRef<str>) -> Self {
        let bytes = string.as_ref().as_bytes();
        assert!(bytes.len() <= PAGE_SIZE, "String is too large");
        let mut data = [0; PAGE_SIZE];

        data[..bytes.len()].copy_from_slice(bytes);
        Self { data }
    }

    pub fn as_bytes(&self, idx: Range<usize>) -> &[u8] {
        &self.data[idx]
    }

    pub fn read_bytes<const N: usize>(&self, offset: usize) -> [u8; N] {
        let mut ret = [0; N];
        ret.copy_from_slice(&self.data[offset..(offset + N)]);
        ret
    }
}

struct PatchedPage {
    base: Rc<Page>,
    patches: Vec<(usize, Vec<u8>)>,
}

impl PatchedPage {
    fn new(base: Rc<Page>) -> Self {
        Self {
            base,
            patches: vec![],
        }
    }
    fn as_bytes_mut(&mut self, idx: Range<usize>) -> &mut [u8] {
        assert!(
            idx.len() <= PAGE_SIZE && idx.end < PAGE_SIZE,
            "Out of bounds write"
        );
        self.patches.push((idx.start, vec![0; idx.len()]));
        let (_, patch) = self.patches.last_mut().unwrap();
        patch.as_mut_slice()
    }

    fn as_bytes(&self, range: Range<usize>) -> Cow<[u8]> {
        assert!(
            range.len() <= PAGE_SIZE && range.end < PAGE_SIZE,
            "Out of bounds read"
        );
        let mut slice = vec![0; range.len()];
        slice.copy_from_slice(self.base.as_bytes(range.clone()));

        for (offset, patch) in &self.patches {
            if range.contains(offset) {
                // Patch start is in range
                let from = offset - range.start;
                let len = patch.len().min(range.end - from);
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
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn create_new_page() {
        let page = Page::from_str("foo");
        assert_eq!(page.as_bytes(0..3), b"foo");
        assert_eq!(page.as_bytes(3..4), [0]);
    }

    #[test]
    fn patch_page() {
        let page = Page::from_str("Hello panic!");
        let mut patched = PatchedPage::new(Rc::new(page));

        patched.as_bytes_mut(6..11).copy_from_slice(b"world");

        assert_eq!(&*patched.as_bytes(0..12), b"Hello world!");
        assert_eq!(&*patched.as_bytes(0..8), b"Hello wo");
        assert_eq!(&*patched.as_bytes(3..12), b"lo world!");
        assert_eq!(&*patched.as_bytes(8..12), b"rld!");
        assert_eq!(&*patched.as_bytes(7..10), b"orl");
    }
}
