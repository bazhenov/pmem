use binrw::{BinRead, BinWrite};
use pmem::{Ptr, Scope, ServiceEntity};
use std::{io::Cursor, mem};

fn main() {}

struct LinkedList<'a> {
    scope: Scope<'a>,
    root: Ptr<ListNode>,
}

#[derive(BinRead, BinWrite)]
#[brw(little)]
struct ListNode {
    value: i32,
    next: Ptr<ListNode>,
}

impl ServiceEntity for ListNode {
    fn size(&self) -> usize {
        mem::size_of::<Self>()
    }

    fn write_to(&self, buffer: &mut Cursor<Vec<u8>>) {
        self.write(buffer).unwrap();
    }
}

impl<'a> LinkedList<'a> {
    fn new(scope: Scope<'a>, root: Ptr<ListNode>) -> Self {
        Self { scope, root }
    }

    fn push_front(&mut self, value: i32) {
        let handle = self.scope.write(ListNode {
            value,
            next: self.root,
        });
        self.root = handle.ptr();
    }

    fn len(&self) -> usize {
        let mut node: Ptr<_> = self.root;
        let mut len = 0;
        while !node.is_null() {
            len += 1;
            node = self.scope.lookup(node).as_ref().next;
        }
        len
    }

    // fn iter(&self) -> impl Iterator<Item = i32> {
    //     ListIterator {
    //         scope: self.scope,
    //         ptr: self.root,
    //     }
    // }

    fn ptr(&self) -> Ptr<ListNode> {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmem::Memory;

    #[test]
    fn check_simple_allocation() {
        let mut memory = Memory::new();
        let a_ptr = {
            let mut scope = Scope::new(&mut memory);

            let a = scope.write(ListNode {
                value: 42,
                next: Ptr::null(),
            });
            a.ptr()
        };

        let mut scope = Scope::new(&mut memory);
        let handle = scope.lookup(a_ptr);
        assert_eq!(handle.as_ref().value, 42);
    }

    #[test]
    fn check_complex_allocation() {
        let mut memory = Memory::new();

        let list_ptr = {
            let mut scope = Scope::new(&mut memory);
            let mut b = scope.write(ListNode {
                value: 35,
                next: Ptr::null(),
            });
            let mut a = scope.write(ListNode {
                value: 34,
                next: b.ptr(),
            });
            scope.finish();

            a.ptr()
        };

        let mut scope = Scope::new(&mut memory);
        let list = LinkedList::new(scope, list_ptr);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn check_pushing_values_to_list() {
        let mut memory = Memory::new();

        let root_ptr = {
            let mut scope = Scope::new(&mut memory);
            let mut list = LinkedList::new(scope, Ptr::null());

            list.push_front(3);
            list.push_front(2);
            list.push_front(1);

            list.ptr()
        };

        let scope = Scope::new(&mut memory);
        let list = LinkedList::new(scope, root_ptr);
        // let values = list.iter().collect();
        assert_eq!(list.len(), 3);
    }
}
