use pmem::{memory, page::Snapshot, Handle, Memory, Ptr};
use pmem_derive::Record;

fn main() {
    let memory = Memory::default();

    let mut list = LinkedList::allocate(memory);
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
    mem: Memory<Snapshot>,
    root: Handle<LinkedListNode>,
    ptr: Ptr<LinkedListNode>,
}

#[derive(Debug, Record)]
struct LinkedListNode {
    first: Option<Ptr<ListNode>>,
}

#[derive(Debug, Record)]
struct ListNode {
    value: i32,
    next: Option<Ptr<ListNode>>,
}

impl LinkedList {
    fn open(mem: Memory<Snapshot>, ptr: Ptr<LinkedListNode>) -> Self {
        let root = mem.lookup(ptr).unwrap();
        Self { mem, root, ptr }
    }

    fn allocate(mut mem: Memory<Snapshot>) -> Self {
        let root = mem.write(LinkedListNode { first: None }).unwrap();
        let ptr = root.ptr();
        Self { mem, root, ptr }
    }

    fn finish(mut self) -> Memory<Snapshot> {
        self.mem.update(&self.root).unwrap();
        self.mem
    }

    fn push_front(&mut self, value: i32) {
        let handle = self
            .mem
            .write(ListNode {
                value,
                next: self.root.first,
            })
            .unwrap();
        self.root.first = Some(handle.ptr());
    }

    fn len(&self) -> usize {
        let mut cur_node = self.root.first;
        let mut len = 0;
        while let Some(node) = cur_node {
            len += 1;
            cur_node = self.mem.lookup(node).unwrap().next;
        }
        len
    }

    fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        ListIterator {
            tx: &self.mem,
            ptr: self.root.first,
        }
    }

    fn ptr(&self) -> Ptr<LinkedListNode> {
        self.ptr
    }
}

struct ListIterator<'a> {
    tx: &'a Memory<Snapshot>,
    ptr: Option<Ptr<ListNode>>,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.tx.lookup(self.ptr?).unwrap();
        self.ptr = node.next;
        Some(node.value)
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
            next: None,
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
            next: None,
        });
        let a = tx.write(ListNode {
            value: 34,
            next: Some(b.ptr()),
        });
        let root = tx.write(LinkedListNode {
            first: Some(a.ptr()),
        });

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
