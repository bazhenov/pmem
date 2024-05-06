use std::mem;

use crate::page::Page;

struct Allocator {
    page: Page,
    offset: usize,
}

impl Allocator {
    fn from(page: Page) -> Self {
        Self { page, offset: 0 }
    }

    fn alloc<T>(&mut self) -> &mut T {
        let size = mem::size_of::<T>();
        let ptr = self
            .page
            .as_bytes(self.offset..(self.offset + size))
            .as_ptr() as *mut T;
        unsafe { &mut *ptr }
    }

    fn into(self) -> Page {
        self.page
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_allocation() {
        let mut allocator = Allocator::from(Page::new());

        let v = allocator.alloc::<u32>();
        *v = 42;
        let page = allocator.into();
        assert_eq!(u32::from_ne_bytes(page.read_bytes::<4>(0)), 42)
    }
}
