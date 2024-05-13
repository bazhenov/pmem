use binrw::{BinRead, BinWrite};
use pmem::{Memory, Ptr, ServiceEntity, Transaction};
use std::io::Cursor;

fn main() {
    let mut memory = Memory::new();

    let root_ptr = memory.change(|tx| {
        let mut list = LinkedList::new(tx, Ptr::null());

        list.push_front(3);
        list.push_front(2);
        list.push_front(1);

        list.ptr()
    });

    let mut tx = memory.start();
    let list = LinkedList::new(&mut tx, root_ptr);
    let values = list.iter().collect::<Vec<_>>();
    assert_eq!(list.len(), 3);
    assert_eq!(values, vec![1, 2, 3]);
}

struct LinkedList<'m, 't> {
    tx: &'t mut Transaction<'m>,
    root: Ptr<ListNode>,
}

#[derive(BinRead, BinWrite)]
#[brw(little)]
struct ListNode {
    value: i32,
    next: Ptr<ListNode>,
}

impl ServiceEntity for ListNode {
    fn write_to(&self, buffer: &mut Cursor<Vec<u8>>) {
        self.write(buffer).unwrap();
    }
}

impl<'m, 't> LinkedList<'m, 't> {
    fn new(tx: &'t mut Transaction<'m>, root: Ptr<ListNode>) -> Self {
        Self { tx, root }
    }

    fn push_front(&mut self, value: i32) {
        let handle = self.tx.write(ListNode {
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
            node = self.tx.lookup(node).as_ref().next;
        }
        len
    }

    fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        ListIterator {
            tx: self.tx,
            ptr: self.root,
        }
    }

    fn ptr(&self) -> Ptr<ListNode> {
        self.root
    }
}

struct ListIterator<'a> {
    tx: &'a Transaction<'a>,
    ptr: Ptr<ListNode>,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            None
        } else {
            let node = self.tx.lookup(self.ptr);
            let node = node.as_ref();
            self.ptr = node.next;
            Some(node.value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_simple_allocation() {
        let mut memory = Memory::new();
        let a_ptr = memory.change(|tx| {
            let a = tx.write(ListNode {
                value: 42,
                next: Ptr::null(),
            });
            a.ptr()
        });

        let handle = memory.start().lookup(a_ptr);
        assert_eq!(handle.as_ref().value, 42);
    }

    #[test]
    fn check_complex_allocation() {
        let mut memory = Memory::new();

        let list_ptr = memory.change(|tx| {
            let b = tx.write(ListNode {
                value: 35,
                next: Ptr::null(),
            });
            let a = tx.write(ListNode {
                value: 34,
                next: b.ptr(),
            });

            a.ptr()
        });

        let mut tx = memory.start();
        let list = LinkedList::new(&mut tx, list_ptr);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn check_pushing_values_to_list() {
        let mut memory = Memory::new();

        let root_ptr = memory.change(|tx| {
            let mut list = LinkedList::new(tx, Ptr::null());

            list.push_front(3);
            list.push_front(2);
            list.push_front(1);

            list.ptr()
        });

        let mut tx = memory.start();
        let list = LinkedList::new(&mut tx, root_ptr);
        let values = list.iter().collect::<Vec<_>>();
        assert_eq!(list.len(), 3);
        assert_eq!(values, vec![1, 2, 3]);
    }
}
