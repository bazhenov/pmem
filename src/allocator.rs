use binrw::BinRead;
use binrw::BinWrite;
use thiserror::Error;

use crate::page::Page;
use std::cell::OnceCell;
use std::marker::PhantomData;
use std::{io::Cursor, mem, rc::Rc};

struct Allocator {
    page: Page,
    offset: usize,
}

#[derive(Error, Debug)]
enum Error {
    #[error("Pointer already initialized")]
    AlreadyInitialized,
}

impl Allocator {
    fn from(page: Page) -> Self {
        Self { page, offset: 0 }
    }

    fn new() -> Self {
        Self {
            page: Page::new(),
            offset: 0,
        }
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

    fn alloc_untyped(&mut self, size: usize) -> &mut [u8] {
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
    fn allocate_simple_value() {
        let mut allocator = Allocator::new();
        let page = Page::new();
        let scope = Scope::open(&mut allocator, &page);

        let mut a = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });
        let mut b = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });

        scope.change(a, |a| a.next2 = b.get_ref());
        scope.change(b, |b| b.next2 = a.get_ref());
        // c.value.get_mut().unwrap().value.next = a;
    }

    #[test]
    fn new_linked_list() {
        let mut allocator = Allocator::new();
        let page = Page::new();
        let scope = Scope::open(&mut allocator, &page);

        let mut list = LinkedList::new(scope.clone());
        list.push(12);

        // let list = LinkedList::new(scope);
        assert_eq!(list.len(), 1);
    }
}

#[derive(BinRead, BinWrite)]
struct Ptr<T> {
    addr: u32,
    #[brw(ignore)]
    value: OnceCell<Rc<Handle<T>>>,
}

impl<T> Ptr<T> {
    fn from_addr(addr: u32) -> Ptr<T> {
        Ptr {
            addr,
            value: OnceCell::new(),
        }
    }

    fn new(value: T) -> Ptr<T> {
        Ptr {
            addr: 0,
            value: OnceCell::from(Rc::new(Handle {
                value,
                ref_count: 0,
            })),
        }
    }

    fn null() -> Self {
        Self::from_addr(0)
    }

    fn is_null(&self) -> bool {
        self.addr == 0 && self.value.get().is_none()
    }
}

struct Handle<T> {
    ref_count: u32,
    value: T,
}

impl<T> Handle<T> {
    fn get_ref(&self) -> Ptr2<T> {
        todo!()
    }
}

impl<T> AsRef<T> for Handle<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

struct LinkedList<'a> {
    scope: Scope<'a>,
    len: usize,
    root: Ptr<ListNode>,
}

#[derive(BinRead, BinWrite)]
#[brw(little)]
struct ListNode {
    value: i32,
    next: Ptr<ListNode>,
    next2: Ptr2<ListNode>,
}

impl<'a> LinkedList<'a> {
    fn new(scope: Scope<'a>) -> Self {
        Self {
            scope,
            len: 0,
            root: Ptr::from_addr(0),
        }
    }

    fn push(&mut self, value: i32) {
        let node = ListNode {
            value,
            next: Ptr::null(),
        };
        self.root = Ptr::new(node);
        self.len += 1;
    }

    fn len(&self) -> usize {
        let mut node = &self.root;
        let mut len = 0;
        while !node.is_null() {
            len += 1;
            node = &self.scope.lookup(&mut node).next;
        }
        len
    }
}

struct Utf8<'a> {
    data: &'a str,
}

#[derive(Clone)]
struct Scope<'a> {
    allocator: &'a Allocator,
    page: &'a Page,
}

impl<'a> Scope<'a> {
    fn open(allocator: &'a mut Allocator, page: &'a Page) -> Self {
        Self { allocator, page }
    }

    fn lookup<T: BinRead>(&'a self, ptr: &'a Ptr<T>) -> &'a T
    where
        T::Args<'a>: Default,
    {
        use binrw::BinReaderExt;

        let get_or_init = ptr.value.get_or_init(|| {
            let addr = ptr.addr as usize;
            let size = u32::from_be_bytes(self.page.read_bytes::<4>(addr)) as usize;
            let ref_count = u32::from_be_bytes(self.page.read_bytes::<4>(addr + 4));
            let bytes = self.page.as_bytes((addr + 8)..(addr + 8 + size));
            let mut cursor = Cursor::new(bytes);
            let value = cursor.read_ne().unwrap();
            Rc::new(Handle { value, ref_count })
        });
        get_or_init.as_ref().as_ref()
    }

    fn new<T>(&self, value: T) -> Handle<T> {
        // let addr = self.allocator.alloc::<T>();
        Ptr2 {
            _phantom: PhantomData,
        }
    }

    fn change<T>(&self, ptr: Handle<T>, f: impl Fn(&mut T)) {}
}

#[derive(BinRead, BinWrite, Clone, Copy)]
struct Ptr2<T> {
    _phantom: PhantomData<T>,
}
