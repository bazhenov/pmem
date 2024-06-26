use crate::page::{Addr, Error, PageOffset, PagePool, Snapshot};
use binrw::{meta::WriteEndian, BinRead, BinWrite};
use std::{
    borrow::Cow,
    io::{Cursor, Seek, SeekFrom, Write},
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
};
const START_ADDR: PageOffset = 4;

/// The size of a header of each entity written to storage
const HEADER_SIZE: usize = mem::size_of::<u32>();

pub struct Memory {
    pool: PagePool,
    next_addr: PageOffset,
    seq: u32,
}

type Result<T> = std::result::Result<T, Error>;

impl Memory {
    pub fn commit(&mut self, tx: Transaction) {
        assert!(tx.next_addr >= self.next_addr);
        assert!(tx.seq == self.seq);
        // Page should be committed first, because it's check for snapshot linearity
        self.pool.commit(tx.snapshot);
        self.seq += 1;
        self.next_addr = tx.next_addr;
    }

    pub fn start(&self) -> Transaction {
        Transaction {
            snapshot: self.pool.snapshot(),
            next_addr: self.next_addr,
            seq: self.seq,
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            pool: PagePool::default(),
            next_addr: START_ADDR,
            seq: 0,
        }
    }
}

pub struct Transaction {
    snapshot: Snapshot,
    next_addr: Addr,
    seq: u32,
}

impl Transaction {
    pub fn write_bytes(&mut self, addr: Addr, bytes: &[u8]) {
        self.snapshot.write(addr, bytes)
    }

    pub fn lookup<'a, T>(&self, ptr: Ptr<T>) -> Handle<T>
    where
        T: BinRead<Args<'a> = ()> + 'static,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr;
        let bytes = self.read_static::<4>(addr);
        let len = u32::from_be_bytes(bytes) as usize;
        let bytes = self.read_uncommited(addr + 4, len).unwrap();
        let mut cursor = Cursor::new(bytes);
        let value: T = cursor.read_ne().unwrap();

        Handle { addr, value }
    }

    pub fn read<T>(&self, ptr: Ptr<T>) -> T
    where
        for<'a> T: BinRead<Args<'a> = ()>,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr;
        let bytes = self.read_static::<4>(addr);
        let len = u32::from_be_bytes(bytes) as usize;
        let bytes = self.read_uncommited(addr + 4, len).unwrap();
        let mut cursor = Cursor::new(bytes);
        cursor.read_ne().unwrap()
    }

    fn read_static<const N: usize>(&self, offset: PageOffset) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.read_uncommited(offset, N).unwrap();
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }

    fn read_uncommited(&self, addr: PageOffset, len: usize) -> Result<Cow<[u8]>> {
        self.snapshot.read(addr, len)
    }

    fn alloc(&mut self, size: usize) -> Addr {
        assert!(size > 0);
        let addr = self.next_addr;
        self.next_addr += size as u32;
        addr
    }

    pub fn write<T>(&mut self, value: T) -> Handle<T>
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian,
    {
        let addr = self.write_to_memory(&value, None).addr;

        Handle { addr, value }
    }

    /// Writes object to a given address or allocates new memory for an object and writes to it
    fn write_to_memory<T>(&mut self, value: &T, ptr: Option<Ptr<T>>) -> Ptr<T>
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian,
    {
        let mut buffer = Cursor::new(Vec::new());

        // reserving space at the beginning for the header
        buffer.write_all(&[0; HEADER_SIZE]).unwrap();

        // writing body
        value.write(&mut buffer).unwrap();
        let size = buffer.position() as usize;
        let ptr = ptr.unwrap_or_else(|| Ptr {
            addr: self.alloc(size),
            _phantom: PhantomData::<T>,
        });

        // writing header with the entity size
        buffer.seek(SeekFrom::Start(0)).unwrap();
        (size as PageOffset).write_be(&mut buffer).unwrap();
        // Making sure we didn't overwrite data by the header
        assert_eq!(buffer.position(), HEADER_SIZE as u64);

        let buffer = buffer.into_inner();
        self.write_bytes(ptr.addr, &buffer);

        ptr
    }

    pub fn update<T>(&mut self, handle: &Handle<T>)
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian,
    {
        self.write_to_memory(&handle.value, Some(handle.ptr()));
    }
}

#[derive(BinRead, BinWrite)]
pub struct Ptr<T> {
    addr: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for Ptr<T> {}

impl<T> Ptr<T> {
    fn from_addr(addr: u32) -> Ptr<T> {
        Ptr {
            addr,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn null() -> Self {
        Self::from_addr(0)
    }

    pub fn is_null(&self) -> bool {
        self.addr == 0
    }
}

pub trait Storable {
    type Seed: Sized;

    fn allocate(tx: Transaction) -> Self;
    fn open(tx: Transaction, ptr: Ptr<Self::Seed>) -> Self;
    fn finish(self) -> Transaction;
}

pub struct Handle<T> {
    addr: Addr,
    value: T,
}

impl<T> Handle<T> {
    pub fn ptr(&self) -> Ptr<T> {
        Ptr {
            addr: self.addr,
            _phantom: PhantomData::<T>,
        }
    }
}

impl<T> Deref for Handle<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Handle<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    const PTR_SIZE: usize = 4;

    /// This test ensures that the size of Ptr is fixed, regardless of its parameter type
    #[test]
    fn check_ptr_size() {
        assert_eq!(mem::size_of::<Ptr<u128>>(), PTR_SIZE);
        assert_eq!(mem::size_of::<Ptr<u8>>(), PTR_SIZE);
    }
}
