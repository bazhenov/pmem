use binrw::BinRead;
use binrw::BinWrite;
use binrw::BinWriterExt;
use thiserror::Error;

use crate::page::Page;
use crate::page::PatchedPage;
use std::any::Any;
use std::any::TypeId;

use std::borrow::Borrow;
use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::{io::Cursor, mem, rc::Rc};

struct Allocator {
    page: Page,
    offset: usize,
}

const START_ADDR: usize = 4;

#[derive(Error, Debug)]
enum Error {
    #[error("Pointer already initialized")]
    AlreadyInitialized,
}

impl Allocator {
    fn from(page: Page) -> Self {
        Self {
            page,
            offset: START_ADDR,
        }
    }

    fn new() -> Self {
        Self {
            page: Page::new(),
            offset: START_ADDR,
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

    fn alloc_untyped(&mut self, size: usize) -> Addr {
        assert!(size > 0);
        let addr = self.offset;
        self.offset += size;
        addr as u32
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
        let page = Rc::new(Page::new());
        let mut scope = Scope::open(&mut allocator, page);

        let mut a = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });
        let mut b = scope.new(ListNode {
            value: 35,
            next: Ptr::null(),
        });

        a.as_mut().next = b.ptr();
        b.as_mut().next = a.ptr();
    }

    #[test]
    fn new_linked_list() {
        let mut allocator = Allocator::new();
        let page = Rc::new(Page::new());

        let scope = Scope::open(&mut allocator, page);

        let mut list = LinkedList::new(scope);
        list.push(12);

        // let list = LinkedList::new(scope);
        assert_eq!(list.len(), 1);
    }
}

#[derive(Clone, BinRead, BinWrite)]
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
        self.addr == 0
    }
}

fn borrow_downcast<T: Any>(cell: &RefCell<dyn Any>) -> Option<Ref<T>> {
    let r = cell.borrow();
    if (*r).type_id() == TypeId::of::<T>() {
        Some(Ref::map(r, |x| x.downcast_ref::<T>().unwrap()))
    } else {
        None
    }
}

struct LinkedList<'a> {
    scope: Scope<'a>,
    root: Ptr<ListNode>,
}

#[derive(BinRead, BinWrite, Clone)]
#[brw(little)]
struct ListNode {
    value: i32,
    next: Ptr<ListNode>,
}

impl ServiceEntity for ListNode {
    fn size(&self) -> usize {
        8
    }

    fn write_to(&self, buffer: &mut [u8]) {
        let mut cursor = Cursor::new(buffer);
        self.write(&mut cursor).unwrap();
    }
}

impl<'a> LinkedList<'a> {
    fn new(scope: Scope<'a>) -> Self {
        Self {
            scope,
            root: Ptr::from_addr(0),
        }
    }

    fn push(&mut self, value: i32) {
        let handle = self.scope.new(ListNode {
            value,
            next: Ptr::null(),
        });
        self.scope.write(&handle);
        self.root = handle.ptr();
    }

    fn len(&self) -> usize {
        let mut node: Ptr<_> = self.root.clone();
        let mut len = 0;
        while !node.is_null() {
            len += 1;
            node = self.scope.lookup(&node).as_ref().next.clone();
        }
        len
    }
}

struct Utf8<'a> {
    data: &'a str,
}

type Addr = u32;

struct Scope<'a> {
    allocator: &'a mut Allocator,
    page: Rc<Page>,
    patched: PatchedPage,
    active_set: RefCell<HashMap<Addr, Rc<RefCell<dyn Any>>>>,
}

impl<'a> Scope<'a> {
    fn open(allocator: &'a mut Allocator, page: Rc<Page>) -> Self {
        Self {
            allocator,
            page: Rc::clone(&page),
            patched: PatchedPage::new(page),
            active_set: RefCell::new(HashMap::new()),
        }
    }

    fn lookup<T: BinRead + 'static>(&'a self, ptr: &'a Ptr<T>) -> Handle<T>
    where
        T::Args<'a>: Default,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr as usize;
        let size = u32::from_ne_bytes(self.patched.read_bytes::<4>(addr)) as usize;
        println!("Size: {}", size);
        // let ref_count = u32::from_be_bytes(self.page.read_bytes::<4>(addr + 4));
        let bytes = self.patched.as_bytes((addr + 4)..(addr + 4 + size));
        let mut cursor = Cursor::new(bytes);
        let value: T = cursor.read_ne().unwrap();

        Handle {
            addr: addr as u32,
            value: Rc::new(RefCell::new(value)),
        }
        // let mut active_set = self.active_set.borrow_mut();
        // active_set.insert(ptr.addr, value);

        // let active_set = self.active_set.borrow();
        // let ref_cell = active_set.get(&ptr.addr).unwrap().clone();
        // let borrow = ref_cell.borrow();
        // borrow.downcast_ref::<T>()
    }

    fn new<T: ServiceEntity>(&mut self, value: T) -> Handle<T> {
        // let addr = self.allocator.alloc::<T>();
        let size = value.size();
        let addr = self.allocator.alloc_untyped(size);
        Handle {
            addr: addr as u32,
            value: Rc::new(RefCell::new(value)),
        }
    }

    fn change<T>(&self, ptr: Ptr<T>, f: impl Fn(&mut T)) {}

    fn finish(&self) {
        todo!()
    }

    fn write<T: ServiceEntity>(&mut self, handle: &Handle<T>) {
        let entity = RefCell::borrow(&handle.value);
        let size = entity.size() + 4;
        let bytes = self
            .patched
            .as_bytes_mut(handle.addr as usize..handle.addr as usize + (size as usize));
        let (header, body) = bytes.split_at_mut(4);
        let mut cursor = Cursor::new(header);
        cursor.write_ne(&(size as u32)).unwrap();
        entity.write_to(body);
    }
}

struct Handle<T> {
    addr: Addr,
    value: Rc<RefCell<T>>,
}

impl<T> Handle<T> {
    fn as_mut(&mut self) -> RefMut<T> {
        RefCell::borrow_mut(&self.value)
    }

    fn as_ref(&self) -> Ref<T> {
        RefCell::borrow(&self.value)
    }

    fn ptr(&self) -> Ptr<T> {
        Ptr {
            addr: self.addr,
            _phantom: PhantomData::<T>,
        }
    }
}

impl<T: Debug + ServiceEntity + 'static> Handle<T> {
    fn join_handle(&self) -> Box<dyn JoinHandle> {
        Box::new(HandleImpl(Some(Rc::clone(&self.value))))
    }
}

struct HandleImpl<T>(Option<Rc<RefCell<T>>>);

impl<T: Debug + ServiceEntity + 'static> JoinHandle for HandleImpl<T> {
    fn join(&mut self) -> Option<Box<dyn ServiceEntity>> {
        let v = self.0.take().unwrap();
        let entity = Rc::try_unwrap(v).unwrap().into_inner();
        Some(Box::new(entity))
    }
}

trait ServiceEntity {
    fn size(&self) -> usize;
    fn write_to(&self, buffer: &mut [u8]);
}

trait JoinHandle {
    fn join(&mut self) -> Option<Box<dyn ServiceEntity>>;
}
