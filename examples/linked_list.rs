use binrw::{BinRead, BinWrite};
use pmem::{Handle, Memory, Ptr, Storable, Transaction};

fn main() {
    let memory = Memory::default();

    let tx = memory.start();

    let mut list = LinkedList::allocate(tx);
    list.push_front(3);
    list.push_front(2);
    list.push_front(1);

    let ptr = list.ptr();
    let tx = list.finish();

    let list: LinkedList = LinkedList::open(tx, ptr);
    let values = list.iter().collect::<Vec<_>>();
    assert_eq!(list.len(), 3);
    assert_eq!(values, vec![1, 2, 3]);
}

struct LinkedList {
    tx: Transaction,
    root: Handle<LinkedListNode>,
    ptr: Ptr<LinkedListNode>,
}

#[derive(BinRead, BinWrite, Debug)]
#[brw(little)]
struct LinkedListNode {
    first: Ptr<ListNode>,
}

impl Storable for LinkedList {
    type Seed = LinkedListNode;

    fn open(tx: Transaction, ptr: Ptr<Self::Seed>) -> Self {
        let root = tx.lookup(ptr);
        Self { tx, root, ptr }
    }

    fn allocate(mut tx: Transaction) -> Self {
        let root = tx.write(LinkedListNode { first: Ptr::null() });
        let ptr = root.ptr();
        Self { tx, root, ptr }
    }

    fn finish(mut self) -> Transaction {
        self.tx.update(&self.root);
        self.tx
    }
}

#[derive(BinRead, BinWrite, Debug)]
#[brw(little)]
struct ListNode {
    value: i32,
    next: Ptr<ListNode>,
}

impl LinkedList {
    fn push_front(&mut self, value: i32) {
        let handle = self.tx.write(ListNode {
            value,
            next: self.root.first,
        });
        self.root.first = handle.ptr();
    }

    fn len(&self) -> usize {
        let mut node: Ptr<_> = self.root.first;
        let mut len = 0;
        while !node.is_null() {
            len += 1;
            node = self.tx.lookup(node).next;
        }
        len
    }

    fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        ListIterator {
            tx: &self.tx,
            ptr: self.root.first,
        }
    }

    fn ptr(&self) -> Ptr<LinkedListNode> {
        self.ptr
    }
}

struct ListIterator<'a> {
    tx: &'a Transaction,
    ptr: Ptr<ListNode>,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            None
        } else {
            let node = self.tx.lookup(self.ptr);
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
        let memory = Memory::default();
        let mut tx = memory.start();
        let a = tx.write(ListNode {
            value: 42,
            next: Ptr::null(),
        });

        let handle = tx.lookup(a.ptr());
        assert_eq!(handle.value, 42);
    }

    #[test]
    fn check_complex_allocation() {
        let memory = Memory::default();

        let mut tx = memory.start();
        let b = tx.write(ListNode {
            value: 35,
            next: Ptr::null(),
        });
        let a = tx.write(ListNode {
            value: 34,
            next: b.ptr(),
        });
        let root = tx.write(LinkedListNode { first: a.ptr() });

        let list = LinkedList::open(tx, root.ptr());
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn check_pushing_values_to_list() {
        let mut memory = Memory::default();

        let tx = memory.start();
        let mut list = LinkedList::allocate(tx);
        list.push_front(3);
        list.push_front(2);
        list.push_front(1);

        let root = list.ptr();
        memory.commit(list.finish());

        let tx = memory.start();
        let list = LinkedList::open(tx, root);
        let values = list.iter().collect::<Vec<_>>();
        assert_eq!(list.len(), 3);
        assert_eq!(values, vec![1, 2, 3]);
    }
}
