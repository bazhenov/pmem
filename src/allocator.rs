use binrw::BinRead;
use binrw::BinWrite;
use thiserror::Error;

use crate::page::Page;
use std::any::Any;
use std::any::TypeId;
use std::cell::OnceCell;
use std::cell::Ref;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::Deref;
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

        let a = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });
        let b = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });

        scope.change(a.ptr(), |a| a.next = b.ptr());
        scope.change(b.ptr(), |b| b.next = a.ptr());
    }

    #[test]
    fn new_linked_list() {
        let mut allocator = Allocator::new();
        let page = Page::new();
        let scope = Scope::open(&mut allocator, &page);

        let mut list = LinkedList::new(scope);
        list.push(12);

        // let list = LinkedList::new(scope);
        assert_eq!(list.len(), 1);
    }
}

#[derive(BinRead, BinWrite)]
struct Ptr<T> {
    addr: u32,
    _phantom: PhantomData<T>,
}

impl<T> Ptr<T> {
    fn from_addr(addr: u32) -> Ptr<T> {
        Ptr {
            addr,
            _phantom: PhantomData::<T>,
        }
    }

    fn new(value: T) -> Ptr<T> {
        Ptr {
            addr: 0,
            _phantom: PhantomData::<T>,
        }
    }

    fn null() -> Self {
        Self::from_addr(0)
    }

    fn is_null(&self) -> bool {
        self.addr == 0 && todo!() //
    }
}

struct Handle<T> {
    ref_count: u32,
    value: RefCell<T>,
}

impl<T> Handle<T> {
    fn ptr(&self) -> Ptr<T> {
        todo!()
    }

    fn get_mut(&self) -> &mut T {
        todo!()
    }

    fn change(&self, f: impl Fn(&mut T)) {}
}

fn borrow_downcast<T: Any>(cell: &RefCell<dyn Any>) -> Option<Ref<T>> {
    let r = cell.borrow();
    if (*r).type_id() == TypeId::of::<T>() {
        Some(Ref::map(r, |x| x.downcast_ref::<T>().unwrap()))
    } else {
        None
    }
}

impl<T> AsRef<T> for Handle<T> {
    fn as_ref(&self) -> &T {
        &self.value.borrow()
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
            node = &self.scope.lookup(&node).unwrap().next;
        }
        len
    }
}

struct Utf8<'a> {
    data: &'a str,
}

type Addr = u32;

struct Scope<'a> {
    allocator: &'a Allocator,
    page: &'a Page,
    active_set: RefCell<HashMap<Addr, Rc<RefCell<dyn Any>>>>,
}

impl<'a> Scope<'a> {
    fn open(allocator: &'a mut Allocator, page: &'a Page) -> Self {
        Self {
            allocator,
            page,
            active_set: RefCell::new(HashMap::new()),
        }
    }

    fn lookup<T: BinRead + 'static>(&'a self, ptr: &'a Ptr<T>) -> Rc<RefCell<T>>
    where
        T::Args<'a>: Default,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr as usize;
        let size = u32::from_be_bytes(self.page.read_bytes::<4>(addr)) as usize;
        let ref_count = u32::from_be_bytes(self.page.read_bytes::<4>(addr + 4));
        let bytes = self.page.as_bytes((addr + 8)..(addr + 8 + size));
        let mut cursor = Cursor::new(bytes);
        let value: T = cursor.read_ne().unwrap();
        let value = Rc::new(RefCell::new(value));

        let mut active_set = self.active_set.borrow_mut();
        active_set.insert(ptr.addr, value);

        let active_set = self.active_set.borrow();
        let ref_cell = active_set.get(&ptr.addr).unwrap().clone();
        let borrow = ref_cell.borrow();
        let v = borrow.downcast_ref::<T>();
    }

    fn new<T>(&self, value: T) -> Handle<T> {
        // let addr = self.allocator.alloc::<T>();
        todo!()
    }

    fn change<T>(&self, ptr: Ptr<T>, f: impl Fn(&mut T)) {}
}

#[derive(BinRead, BinWrite, Clone, Copy)]
struct Ptr2<T> {
    _phantom: PhantomData<T>,
}
