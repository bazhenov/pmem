use std::{borrow::Cow, ops::Range, rc::Rc};

const PAGE_SIZE: usize = 1 << 16; // 64KB

struct Page {
    data: [u8; PAGE_SIZE],
}

impl Page {
    fn new() -> Self {
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

    fn as_bytes(&self, idx: Range<usize>) -> &[u8] {
        &self.data[idx]
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

    fn as_bytes(&self, idx: Range<usize>) -> Cow<[u8]> {
        assert!(
            idx.len() <= PAGE_SIZE && idx.end < PAGE_SIZE,
            "Out of bounds read"
        );
        let mut owned_patch = vec![0; idx.len()];
        owned_patch.copy_from_slice(self.base.as_bytes(idx.clone()));

        for (offset, patch) in &self.patches {
            if idx.contains(offset) {
                let from = offset - idx.start;
                let len = patch.len().min(idx.end - from);
                owned_patch[from..(from + len)].copy_from_slice(&patch[..len])
            } else if idx.contains(&(offset + patch.len())) {
                let from = idx.start - offset;
                let len = idx.len().min(patch.len() - from);
                owned_patch[..len].copy_from_slice(&patch[from..from + len])
            } else if *offset < idx.start && idx.end < offset + patch.len() {
                let from = idx.start - offset;
                owned_patch.copy_from_slice(&patch[from..from + idx.len()])
            }
        }

        Cow::Owned(owned_patch)
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
