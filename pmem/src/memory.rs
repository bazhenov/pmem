use crate::page::{self, Addr, PageOffset, PagePool, Snapshot};
use pmem_derive::Record;
use std::{
    any::type_name,
    array::TryFromSliceError,
    borrow::Cow,
    fmt::{self, Debug},
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
};
const START_ADDR: PageOffset = 4;

/// The size of a header of each entity written to storage
const SLICE_HEADER_SIZE: usize = mem::size_of::<u32>();

pub struct Memory {
    pool: PagePool,
    next_addr: PageOffset,
    seq: u32,
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Page error")]
    PageError(#[from] page::Error),

    #[error("Out of Bounds Read")]
    OutOfBoundsRead(#[from] TryFromSliceError),

    #[error("Null pointer")]
    NullPointer,
}

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

    pub fn lookup<T: Record>(&self, ptr: Ptr<T>) -> Result<Handle<T>> {
        let bytes = self.read_uncommitted(ptr.addr, T::SIZE);
        let value = T::read(&bytes)?;
        let size = T::SIZE;

        Ok(Handle {
            addr: ptr.addr,
            value,
            size,
        })
    }

    pub fn read<T: Record>(&self, ptr: Ptr<T>) -> Result<T> {
        Ok(self.lookup(ptr)?.into_inner())
    }

    pub fn read_slice<T: Record>(&self, ptr: SlicePtr<T>) -> Result<Vec<T>> {
        let addr = ptr.0.addr;
        let items = self.read_static::<SLICE_HEADER_SIZE>(addr);

        let items = u32::from_le_bytes(items) as usize;
        let bytes = self.read_uncommitted(addr + SLICE_HEADER_SIZE as u32, items * T::SIZE);
        let mut result = Vec::with_capacity(items);
        for chunk in bytes.chunks(T::SIZE) {
            result.push(T::read(chunk)?);
        }
        Ok(result)
    }

    fn read_static<const N: usize>(&self, offset: PageOffset) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.read_uncommitted(offset, N);
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }

    fn read_uncommitted(&self, addr: PageOffset, len: usize) -> Cow<[u8]> {
        self.snapshot.read(addr, len)
    }

    fn alloc(&mut self, size: usize) -> Addr {
        assert!(size > 0);
        let addr = self.next_addr;
        self.next_addr += size as u32;
        addr
    }

    pub fn write<T: Record>(&mut self, value: T) -> Result<Handle<T>> {
        let (ptr, size) = self.write_to_memory(&value, None)?;
        let addr = ptr.addr;
        Ok(Handle { addr, value, size })
    }

    pub fn write_slice<T: Record>(&mut self, value: &[T]) -> Result<SlicePtr<T>> {
        assert!(value.len() <= PageOffset::MAX as usize);
        let mut buffer = vec![0u8; SLICE_HEADER_SIZE + T::SIZE * value.len()];

        (value.len() as u32)
            .write(&mut buffer[0..SLICE_HEADER_SIZE])
            .unwrap();

        for (v, chunk) in value
            .iter()
            .zip(buffer[SLICE_HEADER_SIZE..].chunks_mut(T::SIZE))
        {
            v.write(chunk).unwrap();
        }

        let size = buffer.len();
        let ptr = SlicePtr::from_addr(self.alloc(size)).unwrap();
        self.write_bytes(ptr.0.addr, &buffer);
        Ok(ptr)
    }

    /// Writes object to a given address or allocates new memory for an object and writes to it
    /// Returns the pointer and the size of the block
    fn write_to_memory<T: Record>(
        &mut self,
        value: &T,
        ptr: Option<Ptr<T>>,
    ) -> Result<(Ptr<T>, usize)> {
        let mut buffer = vec![0u8; T::SIZE];

        // writing body
        value.write(&mut buffer).unwrap();
        let size = buffer.len();
        let ptr = ptr.unwrap_or_else(|| Ptr {
            addr: self.alloc(size),
            _phantom: PhantomData::<T>,
        });

        self.write_bytes(ptr.addr, &buffer);
        Ok((ptr, size))
    }

    pub fn update<T: Record>(&mut self, handle: &Handle<T>) -> Result<()> {
        self.write_to_memory(&handle.value, Some(handle.ptr()))?;
        Ok(())
    }

    pub fn reclaim<T>(&mut self, handle: Handle<T>) -> T {
        self.snapshot.reclaim(handle.addr, handle.size);
        handle.into_inner()
    }
}

pub struct Ptr<T> {
    addr: u32,
    _phantom: PhantomData<T>,
}

impl<T> Record for Ptr<T> {
    const SIZE: usize = mem::size_of::<Addr>();

    fn read(data: &[u8]) -> Result<Self> {
        let addr = u32::from_le_bytes(data.try_into()?);
        Ptr::from_addr(addr).ok_or(Error::NullPointer)
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        let bytes = self.addr.to_le_bytes();
        data.copy_from_slice(bytes.as_slice());
        Ok(())
    }
}

impl<T> NonZeroRecord for Ptr<T> {}

impl<T> Ptr<T> {
    pub fn unwrap_addr(&self) -> Addr {
        self.addr
    }
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("Ptr<{}>({})", type_name::<T>(), &self.addr))
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for Ptr<T> {}

impl<T> Ptr<T> {
    pub fn from_addr(addr: u32) -> Option<Self> {
        (addr > 0).then_some(Self {
            addr,
            _phantom: PhantomData::<T>,
        })
    }
}
impl<T> NonZeroRecord for SlicePtr<T> {}

#[derive(Record)]
pub struct SlicePtr<T>(Ptr<T>);

impl<T> Clone for SlicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> SlicePtr<T> {
    fn from_addr(addr: u32) -> Option<Self> {
        Ptr::from_addr(addr).map(Self)
    }
}

impl<T> Debug for SlicePtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "SlicePtr<{}>({})",
            type_name::<T>(),
            &self.0.addr
        ))
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for SlicePtr<T> {}

pub trait Record: Sized {
    const SIZE: usize;

    fn read(data: &[u8]) -> Result<Self>;
    fn write(&self, data: &mut [u8]) -> Result<()>;
}

macro_rules! impl_record_for_primitive {
    ($($t:ty),*) => {
        $(
            impl Record for $t {
                const SIZE: usize = mem::size_of::<Self>();

                fn read(data: &[u8]) -> Result<Self> {
                    Ok(<$t>::from_le_bytes(data.try_into()?))
                }

                fn write(&self, data: &mut [u8]) -> Result<()> {
                    data.copy_from_slice(self.to_le_bytes().as_slice());
                    Ok(())
                }
            }
        )*
    };
}

impl_record_for_primitive!(u8, u16, u32, u64);
impl_record_for_primitive!(i8, i16, i32, i64);

/// Indicates that all zero bytes is not valid state for a record and it is represented
/// as an `Option<T>` in type system.
trait NonZeroRecord: Record {}

impl<T: NonZeroRecord> Record for Option<T> {
    const SIZE: usize = T::SIZE;

    fn read(data: &[u8]) -> Result<Self> {
        if data.iter().any(|v| *v != 0) {
            T::read(data).map(Some)
        } else {
            Ok(None)
        }
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        match self {
            Some(value) => value.write(data),
            None => {
                data.fill(0);
                Ok(())
            }
        }
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
    size: usize,
    value: T,
}

impl<T> Handle<T> {
    pub fn ptr(&self) -> Ptr<T> {
        Ptr {
            addr: self.addr,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn into_inner(self) -> T {
        self.value
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
    use pmem_derive::Record;
    use std::mem;

    const PTR_SIZE: usize = 4;

    // Helper container used for testing read/writes
    #[derive(Record, PartialEq, Debug)]
    struct Value(u32);

    /// This test ensures that the size of Ptr is fixed, regardless of its parameter type
    #[test]
    fn check_ptr_size() {
        assert_eq!(mem::size_of::<Ptr<u128>>(), PTR_SIZE);
        assert_eq!(mem::size_of::<Ptr<u8>>(), PTR_SIZE);
    }

    #[test]
    fn read_write_ptr() -> Result<()> {
        let (mut tx, _) = start();

        let value = Value(42);
        let handle = tx.write(value)?;
        let value_copy = tx.read(handle.ptr())?;
        assert_eq!(&value_copy, &*handle);
        Ok(())
    }

    #[test]
    fn read_write_slice() -> Result<()> {
        let (mut tx, _) = start();

        let value: &[u8] = &[0, 1, 2, 3, 4, 5];
        let slice = tx.write_slice(value)?;
        let value_copy = tx.read_slice(slice)?;
        assert_eq!(value_copy, value);
        Ok(())
    }

    #[test]
    fn check_error() {
        let (tx, _) = start();

        #[derive(Debug, Record)]
        struct Data {
            ptr: Ptr<Data>,
        }

        // reading from made up address
        let value = tx.read(Ptr::<Data>::from_addr(0xFF).unwrap());
        assert!(value.is_err(), "Err should be generated");
    }

    #[test]
    fn reclaim() -> Result<()> {
        let (mut tx, _) = start();

        let handle = tx.write(Value(42))?;
        let ptr = handle.ptr();
        tx.reclaim(handle);
        let result = tx.read(ptr)?;
        // should be zero, because we use zero-fill semantics
        assert_eq!(result, Value(0));
        Ok(())
    }

    fn start() -> (Transaction, Memory) {
        let mem = Memory::default();
        (mem.start(), mem)
    }
}
