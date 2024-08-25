use crate::page::{Addr, PageOffset, Transaction, TxRead, TxWrite};
use pmem_derive::Record;
use std::{
    any::type_name,
    array::TryFromSliceError,
    borrow::Cow,
    fmt::{self, Debug},
    io,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
};
/// The size of any pointer in bytes
pub const PTR_SIZE: usize = mem::size_of::<Addr>();
const START_ADDR: Addr = 8;

/// The size of a header of each entity written to storage
const SLICE_HEADER_SIZE: usize = mem::size_of::<u32>();

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Data integrity")]
    DataIntegrity(#[from] TryFromSliceError),

    #[error("No space left")]
    NoSpaceLeft,

    #[error("Null pointer")]
    NullPointer,

    #[error("Unexpected variant code: {0}")]
    UnexpectedVariantCode(u64),

    #[error("Page error: {0}")]
    Page(#[from] crate::page::Error),
}

impl From<Error> for io::Error {
    fn from(error: Error) -> Self {
        let kind = match error {
            Error::DataIntegrity(..) => io::ErrorKind::InvalidData,
            Error::NoSpaceLeft => io::ErrorKind::OutOfMemory,
            Error::NullPointer => io::ErrorKind::InvalidInput,
            Error::UnexpectedVariantCode(..) => io::ErrorKind::InvalidInput,
            Error::Page(..) => io::ErrorKind::Other,
        };
        io::Error::new(kind, error)
    }
}

pub struct Memory<T> {
    tx: T,
    mem_info: MemoryInfo,
}

#[derive(Record, Clone)]
struct MemoryInfo {
    /// Next address to allocate objects
    next_addr: Addr,
}

impl MemoryInfo {
    fn update(&self, tx: &mut impl TxWrite) {
        let mut bytes = [0u8; Self::SIZE];
        self.write(&mut bytes)
            .expect("Unable to update memory info block");
        tx.write(START_ADDR, bytes);
    }
}

impl Memory<Transaction> {
    #[cfg(test)]
    fn new() -> Self {
        let pool = crate::page::PagePool::default();
        Self::init(pool.start())
    }
}

impl<S: TxRead> Memory<S> {
    pub fn open(tx: S) -> Self {
        let bytes = tx.read(START_ADDR, MemoryInfo::SIZE);
        let mem_info = MemoryInfo::read(&bytes).unwrap();
        assert!(
            mem_info.next_addr > 0,
            "Invalid memory info block. Maybe you forgot to call Memory::init()?"
        );
        Self { tx, mem_info }
    }

    pub fn read_super_block<T: Record>(&self) -> Result<Handle<T>> {
        let ptr = Ptr::from_addr(START_ADDR + MemoryInfo::SIZE as Addr).unwrap();
        self.lookup(ptr)
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
        let bytes = self.read_uncommitted(addr + SLICE_HEADER_SIZE as Addr, items * T::SIZE);
        let mut result = Vec::with_capacity(items);
        for chunk in bytes.chunks(T::SIZE) {
            result.push(T::read(chunk)?);
        }
        Ok(result)
    }

    pub fn read_bytes(&self, ptr: SlicePtr<u8>) -> Result<Cow<[u8]>> {
        let addr = ptr.0.addr;
        let items = self.read_static::<SLICE_HEADER_SIZE>(addr);

        let bytes = u32::from_le_bytes(items) as usize;
        Ok(self.read_uncommitted(addr + SLICE_HEADER_SIZE as Addr, bytes))
    }

    fn read_static<const N: usize>(&self, addr: Addr) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.read_uncommitted(addr, N);
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }

    fn read_uncommitted(&self, addr: Addr, len: usize) -> Cow<[u8]> {
        self.tx.read(addr, len)
    }
}

impl<S: TxWrite> Memory<S> {
    pub fn init(mut tx: S) -> Self {
        let next_addr = START_ADDR + MemoryInfo::SIZE as Addr;

        let mem_info = MemoryInfo { next_addr };

        let mut bytes = [0u8; MemoryInfo::SIZE];
        mem_info.write(&mut bytes).unwrap();
        tx.write(START_ADDR, bytes);

        Self { tx, mem_info }
    }

    fn alloc_space(&mut self, size: usize) -> Result<Addr> {
        assert!(size > 0);
        let addr = self.mem_info.next_addr;
        if !self.tx.valid_range(addr, size) {
            return Err(Error::NoSpaceLeft);
        }
        self.mem_info.next_addr += size as Addr;
        Ok(addr)
    }

    pub fn alloc<T: Record>(&mut self) -> Result<Ptr<T>> {
        let addr = self.alloc_space(T::SIZE)?;
        Ok(Ptr::from_addr(addr).unwrap())
    }

    pub fn write_at<T: Record>(&mut self, ptr: Ptr<T>, value: T) -> Result<Handle<T>> {
        let (ptr, size) = self.write_to_memory(&value, Some(ptr))?;
        let addr = ptr.addr;
        Ok(Handle { addr, value, size })
    }

    pub fn write<T: Record>(&mut self, value: T) -> Result<Handle<T>> {
        let (ptr, size) = self.write_to_memory(&value, None)?;
        let addr = ptr.addr;
        Ok(Handle { addr, value, size })
    }

    pub fn write_slice<T: Record>(&mut self, values: &[T]) -> Result<SlicePtr<T>> {
        assert!(values.len() <= u32::MAX as usize);

        let size = SLICE_HEADER_SIZE + T::SIZE * values.len();
        let ptr = SlicePtr::from_addr(self.alloc_space(size)?).expect("Alloc failed");

        let mut buffer = vec![0u8; size];

        (values.len() as u32)
            .write(&mut buffer[0..SLICE_HEADER_SIZE])
            .unwrap();

        let chunks = values
            .iter()
            .zip(buffer[SLICE_HEADER_SIZE..].chunks_mut(T::SIZE));
        for (value, byte_chunk) in chunks {
            value.write(byte_chunk).unwrap();
        }

        {
            let this = &mut *self;
            let addr = ptr.0.addr;
            this.tx.write(addr, buffer)
        };
        Ok(ptr)
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<SlicePtr<u8>> {
        assert!(bytes.len() <= PageOffset::MAX as usize);

        let size = SLICE_HEADER_SIZE + bytes.len();
        let ptr = SlicePtr::from_addr(self.alloc_space(size)?).expect("Alloc failed");

        let mut buffer = vec![0u8; size];

        (bytes.len() as u32)
            .write(&mut buffer[0..SLICE_HEADER_SIZE])
            .unwrap();

        buffer[SLICE_HEADER_SIZE..].copy_from_slice(bytes);

        {
            let this = &mut *self;
            let addr = ptr.0.addr;
            this.tx.write(addr, buffer)
        };
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

        let size = buffer.len();
        let ptr = if let Some(ptr) = ptr {
            ptr
        } else {
            Ptr::from_addr(self.alloc_space(size)?).expect("Alloc failed")
        };

        value.write(&mut buffer)?;
        {
            let this = &mut *self;
            let addr = ptr.addr;
            this.tx.write(addr, buffer)
        };
        Ok((ptr, size))
    }

    pub fn update<T: Record>(&mut self, handle: &Handle<T>) -> Result<()> {
        self.write_to_memory(&handle.value, Some(handle.ptr()))?;
        Ok(())
    }

    pub fn reclaim<T>(&mut self, handle: Handle<T>) -> T {
        self.tx.reclaim(handle.addr, handle.size);
        handle.into_inner()
    }

    pub fn finish(self) -> S {
        let Memory {
            tx: mut snapshot,
            mem_info,
        } = self;
        mem_info.update(&mut snapshot);
        snapshot
    }
}

pub struct Ptr<T> {
    addr: Addr,
    _phantom: PhantomData<T>,
}

impl<T> Ptr<T> {
    pub fn from_addr(addr: Addr) -> Option<Self> {
        (addr > 0).then_some(Self {
            addr,
            _phantom: PhantomData::<T>,
        })
    }
}

impl<T> Record for Ptr<T> {
    const SIZE: usize = mem::size_of::<Addr>();

    fn read(data: &[u8]) -> Result<Self> {
        let addr = Addr::from_le_bytes(data.try_into()?);
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

impl<T> PartialEq for Ptr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for Ptr<T> {}

impl<T> NonZeroRecord for SlicePtr<T> {}

#[derive(Record)]
pub struct SlicePtr<T>(Ptr<T>);

impl<T> Clone for SlicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> SlicePtr<T> {
    fn from_addr(addr: Addr) -> Option<Self> {
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

impl<const N: usize, T: Record + Default + Copy> Record for [T; N] {
    const SIZE: usize = T::SIZE * N;

    fn read(data: &[u8]) -> Result<Self> {
        assert!(data.len() == T::SIZE * N);

        let mut result = [T::default(); N];
        for (item, chunk) in result.iter_mut().zip(data.chunks(T::SIZE)) {
            *item = T::read(chunk)?;
        }
        Ok(result)
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        assert!(data.len() == T::SIZE * N);

        for (value, bytes) in self.iter().zip(data.chunks_mut(T::SIZE)) {
            value.write(bytes)?
        }
        Ok(())
    }
}

/// The Record trait is used to serialize and deserialize data structures.
///
/// It requires the implementor to provide the size of the data structure and methods to read and write
/// the data structure from a byte slice.
///
/// The users supposed to use the derive macro to implement this trait. It provides a default implementation
/// for the Record trait for the data structure using simple algorithm to tight pack the fields suing little endian
/// binary format. Enums are serialized using the discriminant value followed by the fields.
///
/// Example:
/// ```ignore
/// use pmem::memory::Record;
/// use pmem_derive::Record;
///
/// #[derive(Record)]
/// struct MyRecord {
///     field1: u32,
///     field2: u64,
/// }
///
/// fn read_my_struct(input: &[u8]) {
///     let my_record = MyRecord::read(&input).unwrap();
/// }
///
/// fn write_my_struct(output: &mut [u8], my_record: &MyRecord) {
///     my_record.write(output).unwrap();
/// }
/// ```
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

impl Record for bool {
    const SIZE: usize = mem::size_of::<Self>();

    fn read(data: &[u8]) -> Result<Self> {
        assert!(data.len() == Self::SIZE);
        Ok(data[0] != 0)
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        assert!(data.len() == Self::SIZE);
        data[0] = *self as u8;
        Ok(())
    }
}

/// Indicates that all zero bytes is not valid state for a record and it is represented
/// as an `Option<T>::None` in type system.
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

/// Stable Rust at the moment doesn't support compile time max(). Used by proc_macro
pub const fn max<const N: usize>(array: [usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i < array.len() {
        if array[i] > max {
            max = array[i];
        }
        i += 1;
    }
    max
}

#[cfg(test)]
mod tests {
    use crate::page::PagePool;

    use super::*;
    use pmem_derive::Record;
    use std::mem;

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
    fn test_max() {
        assert_eq!(max([1, 2, 3]), 3);
        assert_eq!(max([3, 2, 1]), 3);
        assert_eq!(max([]), 0);
    }

    #[test]
    fn read_write_ptr() -> Result<()> {
        let mut mem = Memory::new();

        let value = Value(42);
        let handle = mem.write(value)?;
        let value_copy = mem.read(handle.ptr())?;
        assert_eq!(&value_copy, &*handle);
        Ok(())
    }

    #[test]
    fn read_write_slice() -> Result<()> {
        let mut mem = Memory::new();

        let value: &[u8] = &[0, 1, 2, 3, 4, 5];
        let slice = mem.write_slice(value)?;
        let value_copy = mem.read_slice(slice)?;
        assert_eq!(value_copy, value);
        Ok(())
    }

    #[test]
    fn memory_can_persist_over_commits() -> Result<()> {
        let mut pool = PagePool::default();
        let mut mem = Memory::init(pool.start());

        let mut ptrs = vec![];

        for i in 0..10 {
            let handle = mem.write(Value(i))?;
            ptrs.push(handle.ptr());
            pool.commit(mem.finish()).unwrap();
            mem = Memory::open(pool.start());
        }

        for (i, ptr) in ptrs.into_iter().enumerate() {
            let value = mem.read(ptr)?;
            assert_eq!(value, Value(i as u32));
        }
        Ok(())
    }

    #[test]
    fn check_error() {
        let mem = Memory::new();

        #[derive(Debug, Record)]
        struct Data {
            ptr: Ptr<Data>,
        }

        // reading from made up address
        let value = mem.read(Ptr::<Data>::from_addr(0xFF).unwrap());
        assert!(value.is_err(), "Err should be generated");
    }

    #[test]
    fn reclaim() -> Result<()> {
        let mut mem = Memory::new();

        let handle = mem.write(Value(42))?;
        let ptr = handle.ptr();
        mem.reclaim(handle);
        let result = mem.read(ptr)?;
        // should be zero, because we use zero-fill semantics
        assert_eq!(result, Value(0));
        Ok(())
    }
}
