use crate::page::Page;
use binary_layout::prelude::*;
use std::{borrow::Cow, marker::PhantomData, mem};

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
            .as_mut_bytes(self.offset..(self.offset + size))
            .as_mut_ptr() as *mut T;
        self.offset += size;
        unsafe { &mut *ptr }
    }

    fn alloc_sized(&mut self, size: usize) -> &mut [u8] {
        self.page
            .as_mut_bytes(self.offset..(self.offset + size))
            .as_mut()
    }

    fn into(self) -> Page {
        self.page
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn check_allocation() {
        let mut allocator = Allocator::from(Page::new());

        let v = allocator.alloc::<u32>();
        *v = 42;
        let page = allocator.into();
        assert_eq!(u32::from_ne_bytes(page.read_bytes::<4>(0)), 42);
    }

    #[test]
    fn allocate_string() {
        let string = "Hello world!";
        let mut allocator = Allocator::from(Page::new());

        let sz = allocator.alloc::<u8>();
        *sz = string.len() as u8;

        let p = allocator.alloc::<[u8; 12]>();
        p.copy_from_slice(string.as_bytes());

        let page = allocator.into();
        let utf = Utf8::from(page.as_bytes(0..100));
        assert_eq!(utf.data, string);
    }

    #[test]
    fn allocate_string2() {
        let string = "Hello world!";
        let mut allocator = Allocator::from(Page::new());

        let bytes = allocator.alloc_sized(string.len() + 1);
        let mut v = string::View::new(bytes);
        v.len_mut().write(string.len() as u8);
        v.data_mut().write(string.as_bytes()).unwrap();

        let page = allocator.into();
        let utf = Utf8::from(page.as_bytes(0..100));
        assert_eq!(utf.data, string);
    }
}

struct Utf8<'a> {
    data: &'a str,
}

trait Bytable {
    type Mmap: for<'a> FromMmap<'a>;
}

trait FromMmap<'a> {
    fn from_mmap(memory: &'a [u8]) -> Self;
}

impl<'a> From<&'a [u8]> for Utf8<'a> {
    fn from(value: &'a [u8]) -> Self {
        let packet = string::View::new(value);
        let len = packet.len().read() as usize;
        Self {
            data: std::str::from_utf8(&value[1..(len + 1)]).unwrap(),
        }
    }
}

binary_layout!(string, LittleEndian, {
  len: u8,
  data: [u8],
});

struct Ref<T> {
    ptr: usize,
    _phantom: PhantomData<T>,
}

struct Node<'a> {
    value: Cow<'a, str>,
    next: Ref<Self>,
}
