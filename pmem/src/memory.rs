use crate::page::{self, Addr, PageOffset, PagePool, Snapshot};
use binrw::{meta::WriteEndian, BinRead, BinResult, BinWrite};
use std::{
    any::type_name,
    array::TryFromSliceError,
    borrow::Cow,
    fmt::{self, Debug},
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Page error")]
    PageError(#[from] page::Error),

    #[error("Binrw Error")]
    BinrwError(#[from] binrw::Error),

    #[error("Out of Bounds Read")]
    OutOfBoundsRead(#[from] TryFromSliceError),
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

    pub fn lookup<'a, T>(&self, ptr: Ptr<T>) -> Handle<T>
    where
        T: BinRead<Args<'a> = ()> + 'static,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr;
        let bytes = self.read_static::<4>(addr);
        let len = u32::from_be_bytes(bytes) as usize;
        let bytes = self.read_uncommitted(addr + 4, len).unwrap();
        let mut cursor = Cursor::new(bytes);
        let value: T = cursor.read_ne().unwrap();
        let size = len + 4;

        Handle { addr, value, size }
    }

    pub fn read<T>(&self, ptr: Ptr<T>) -> Result<T>
    where
        for<'a> T: BinRead<Args<'a> = ()>,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr;
        let bytes = self.read_static::<4>(addr);
        let len = u32::from_be_bytes(bytes) as usize;
        let bytes = self.read_uncommitted(addr + 4, len).unwrap();
        let mut cursor = Cursor::new(bytes);
        Ok(cursor.read_ne()?)
    }

    pub fn read_slice<T>(&self, ptr: SlicePtr<T>) -> Vec<T>
    where
        for<'a> T: BinRead<Args<'a> = ()>,
    {
        use binrw::BinReaderExt;

        let addr = ptr.addr;
        let bytes = self.read_static::<4>(addr);
        let len = u32::from_be_bytes(bytes) as usize;
        let bytes = self.read_uncommitted(addr + 4, len).unwrap();
        let mut result = Vec::with_capacity(len);
        for item in bytes.chunks(mem::size_of::<T>()) {
            let mut cursor = Cursor::new(item);
            result.push(cursor.read_ne().unwrap());
        }
        result
    }

    fn read_static<const N: usize>(&self, offset: PageOffset) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.read_uncommitted(offset, N).unwrap();
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }

    fn read_uncommitted(&self, addr: PageOffset, len: usize) -> Result<Cow<[u8]>> {
        Ok(self.snapshot.read(addr, len)?)
    }

    fn alloc(&mut self, size: usize) -> Addr {
        assert!(size > 0);
        let addr = self.next_addr;
        self.next_addr += size as u32;
        addr
    }

    pub fn write<T>(&mut self, value: T) -> Handle<T>
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian + Debug,
    {
        let (ptr, size) = self.write_to_memory(&value, None);
        let addr = ptr.addr;
        Handle { addr, value, size }
    }

    pub fn write_slice<T>(&mut self, value: &[T]) -> SlicePtr<T>
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian,
    {
        assert!(value.len() <= PageOffset::MAX as usize);
        let mut buffer = Cursor::new(Vec::new());

        // reserving space at the beginning for the header
        buffer.write_all(&[0; HEADER_SIZE]).unwrap();

        // writing body
        for v in value {
            let before = buffer.position();
            v.write(&mut buffer).unwrap();
            let after = buffer.position();
            let size = after - before;
            assert!(
                size == mem::size_of::<T>() as u64,
                "Serialized size should be the same as size in memory"
            );
        }

        let size = buffer.position() as usize;
        let ptr = SlicePtr {
            addr: self.alloc(size),
            _phantom: PhantomData::<T>,
        };

        // writing header with the entity size
        buffer.seek(SeekFrom::Start(0)).unwrap();
        (value.len() as PageOffset).write_be(&mut buffer).unwrap();
        // Making sure we didn't overwrite data by the header
        assert!(buffer.position() == HEADER_SIZE as u64);

        let buffer = buffer.into_inner();
        self.write_bytes(ptr.addr, &buffer);

        ptr
    }

    /// Writes object to a given address or allocates new memory for an object and writes to it
    /// Returns the pointer and the size of the block
    fn write_to_memory<T>(&mut self, value: &T, ptr: Option<Ptr<T>>) -> (Ptr<T>, usize)
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian + Debug,
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
        assert!(buffer.position() == HEADER_SIZE as u64);

        let buffer = buffer.into_inner();
        self.write_bytes(ptr.addr, &buffer);

        (ptr, size)
    }

    pub fn update<T>(&mut self, handle: &Handle<T>)
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian + Debug,
    {
        self.write_to_memory(&handle.value, Some(handle.ptr()));
    }

    pub fn reclaim<T>(&mut self, handle: Handle<T>) -> T
    where
        for<'a> T: BinWrite<Args<'a> = ()> + WriteEndian + Debug,
    {
        self.snapshot.reclaim(handle.addr, handle.size);
        handle.into_inner()
    }
}

#[binrw::parser(reader, endian)]
pub fn parse_optional_ptr<T>() -> BinResult<Option<Ptr<T>>> {
    let addr = u32::read_options(reader, endian, ())?;
    if addr == 0 {
        Ok(None)
    } else {
        Ok(Some(Ptr::from_addr(addr)))
    }
}

#[binrw::writer(writer, endian)]
pub fn write_optional_ptr<T>(ptr: &Option<Ptr<T>>) -> BinResult<()> {
    let addr = ptr.map(|p| p.addr).unwrap_or_default();
    addr.write_options(writer, endian, ())
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

impl<T> Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("Ptr<{}>({})", type_name::<T>(), &self.addr))
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for Ptr<T> {}

impl<T> Ptr<T> {
    pub fn from_addr(addr: u32) -> Self {
        assert!(addr > 0);
        Self {
            addr,
            _phantom: PhantomData::<T>,
        }
    }
}

#[derive(BinRead, BinWrite)]
pub struct SlicePtr<T> {
    addr: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for SlicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> SlicePtr<T> {
    fn from_addr(addr: u32) -> Self {
        assert!(addr > 0);
        Self {
            addr,
            _phantom: PhantomData::<T>,
        }
    }
}

impl<T> Debug for SlicePtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "SlicePtr<{}>({})",
            type_name::<T>(),
            &self.addr
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
                    data[..mem::size_of::<Self>()].copy_from_slice(self.to_le_bytes().as_slice());
                    Ok(())
                }
            }
        )*
    };
}

impl_record_for_primitive!(u8, u16, u32, u64);
impl_record_for_primitive!(i8, i16, i32, i64);

impl<T> Record for Option<Ptr<T>> {
    // const SIZE: usize = <Ptr<T> as Record>::SIZE;
    //
    const SIZE: usize = 4;

    fn read(data: &[u8]) -> Result<Self> {
        let addr = u32::from_le_bytes(data[0..4].try_into()?);
        let ptr = Some(addr).filter(|addr| *addr > 0).map(Ptr::from_addr);
        Ok(ptr)
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        let mut c = Cursor::new(data);
        let addr = self.map(|ptr| ptr.addr).unwrap_or(0);
        Ok(addr.write_le(&mut c)?)
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
    use std::mem;

    const PTR_SIZE: usize = 4;

    // Helper container used for testing read/writes
    #[derive(BinRead, BinWrite, PartialEq, Debug)]
    #[brw(little)]
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
        let handle = tx.write(value);
        let value_copy = tx.read(handle.ptr())?;
        assert_eq!(&value_copy, &*handle);
        Ok(())
    }

    #[test]
    fn read_write_slice() {
        let (mut tx, _) = start();

        let value: &[u8] = &[0, 1, 2, 3, 4, 5];
        let slice = tx.write_slice(value);
        let value_copy = tx.read_slice(slice);
        assert_eq!(value_copy, value);
    }

    #[test]
    fn reclaim() -> Result<()> {
        let (mut tx, _) = start();

        let handle = tx.write(Value(42));
        let ptr = handle.ptr();
        tx.reclaim(handle);
        let resulr = tx.read(ptr);
        assert!(resulr.is_err(), "Error should be generated");
        Ok(())
    }

    fn start() -> (Transaction, Memory) {
        let mem = Memory::default();
        (mem.start(), mem)
    }
}
