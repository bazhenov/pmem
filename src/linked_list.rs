use crate::memory::{Ptr, Scope, ServiceEntity};
use binrw::{BinRead, BinWrite};
use std::io::Cursor;

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

    fn write_to(&self, buffer: &mut Cursor<Vec<u8>>) {
        self.write(buffer).unwrap();
    }
}

impl<'a> LinkedList<'a> {
    fn new(scope: Scope<'a>, root: Ptr<ListNode>) -> Self {
        Self { scope, root }
    }

    fn push(&mut self, value: i32) {
        let handle = self.scope.write(ListNode {
            value,
            next: Ptr::null(),
        });
        self.root = handle.ptr();
    }

    fn len(&self) -> usize {
        let mut node: Ptr<_> = self.root;
        let mut len = 0;
        while !node.is_null() {
            // dbg!(node.addr);
            len += 1;
            node = self.scope.lookup(node).as_ref().next;
        }
        len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

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

        let a_ptr = {
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
        let list = LinkedList::new(scope, a_ptr);
        assert_eq!(list.len(), 2);
    }
}
