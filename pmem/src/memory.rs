use crate::volume::{split_addr, Addr, PageNo, PageOffset, TxRead, TxWrite, PAGE_SIZE};
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
use tracing::trace;

/// The size of any pointer in bytes
pub const PTR_SIZE: usize = Ptr::<u8>::SIZE;
pub const NULL_PTR_SIZE: usize = Option::<Ptr<u8>>::SIZE;
const START_ADDR: Addr = 8;

/// Minimum allocation size for the [`Memory::alloc`] method
const MIN_ALLOC_SIZE: usize = 16;

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

    #[error("Slot #{0} is not free")]
    SlotIsNotFree(Addr),

    #[error("Unable to read record {addr}/{type_name}: {err:?}")]
    RecordReadError {
        addr: Addr,
        type_name: &'static str,
        err: Box<Error>,
    },

    #[error("Page error: {0}")]
    Page(#[from] crate::volume::Error),

    #[error("IO Error: {0}")]
    Io(#[from] io::Error),
}

impl From<Error> for io::Error {
    fn from(error: Error) -> Self {
        let kind = match &error {
            Error::DataIntegrity(..) => io::ErrorKind::InvalidData,
            Error::NoSpaceLeft => io::ErrorKind::OutOfMemory,
            Error::NullPointer => io::ErrorKind::InvalidInput,
            Error::SlotIsNotFree(_) => io::ErrorKind::InvalidData,
            Error::UnexpectedVariantCode(..) => io::ErrorKind::InvalidInput,
            Error::RecordReadError { .. } => io::ErrorKind::InvalidData,
            Error::Page(..) => io::ErrorKind::Other,
            Error::Io(e) => e.kind(),
        };
        io::Error::new(kind, error)
    }
}

pub struct Memory<T> {
    tx: T,
    mem_info: MemoryInfo,
}

impl<T: TxRead> TxRead for Memory<T> {
    fn read_to_buf(&self, addr: Addr, buf: &mut [u8]) {
        self.tx.read_to_buf(addr, buf)
    }

    fn valid_range(&self, addr: Addr, len: usize) -> bool {
        self.tx.valid_range(addr, len)
    }
}

impl<T: TxWrite> TxWrite for Memory<T> {
    fn write(&mut self, addr: Addr, bytes: impl Into<Vec<u8>>) {
        self.tx.write(addr, bytes)
    }

    fn reclaim(&mut self, addr: Addr, len: usize) {
        self.tx.reclaim(addr, len)
    }
}

#[derive(Record, Clone)]
pub struct MemoryInfo {
    /// Next address to allocate objects
    next_addr: Addr,

    next_page: PageNo,

    /// Pointer to the first block in the memory that can be reused
    free_list: Option<Ptr<FreeBlock>>,
}

/// Each allocation block has a header with the size of the block
#[derive(Record)]
pub struct AllocatedBlock {
    // Size of of the memory block excluding header size (this structure)
    size: u32,
}

/// Header of a block of memory that was freed using [`Memory::reclaim()`] call.
///
/// [FreeBlock] and [AllocatedBlock] should have the same prefix with the size
/// of an allocated block.
#[derive(Record)]
struct FreeBlock {
    // Size of of the memory block excluding the [`AllocatedBlock`] header
    size: u32,
    next: Option<Ptr<FreeBlock>>,
}

impl MemoryInfo {
    fn update(&self, tx: &mut impl TxWrite) {
        let mut bytes = [0u8; Self::SIZE];
        self.write(&mut bytes)
            .expect("Unable to update memory info block");
        tx.write(START_ADDR, bytes);
    }
}

impl<S: TxRead> Memory<S> {
    pub fn open(tx: S) -> Self {
        let bytes = tx.read(START_ADDR, MemoryInfo::SIZE);
        let mem_info = MemoryInfo::read(&bytes).unwrap();
        Self { tx, mem_info }
    }

    pub fn lookup<T: Record>(&self, ptr: Ptr<T>) -> Result<Handle<T>> {
        read_value(ptr, &self.tx.read(ptr.addr, T::SIZE))
    }

    pub fn read_bytes(&self, ptr: SlicePtr<u8>) -> Result<Cow<[u8]>> {
        let addr = ptr.0.addr;
        let items = self.read_static::<SLICE_HEADER_SIZE>(addr);

        let bytes = u32::from_le_bytes(items) as usize;
        Ok(self.tx.read(addr + SLICE_HEADER_SIZE as Addr, bytes))
    }

    fn read_static<const N: usize>(&self, addr: Addr) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.tx.read(addr, N);
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }

    #[cfg(test)]
    pub fn read<T: Record>(&self, ptr: Ptr<T>) -> Result<T> {
        Ok(self.lookup(ptr)?.into_inner())
    }
}

impl<S: TxWrite> Memory<S> {
    pub fn init(mut tx: S) -> Self {
        let mem_info = MemoryInfo {
            next_addr: 0,
            next_page: 1,
            free_list: None,
        };

        let mut bytes = [0u8; MemoryInfo::SIZE];
        mem_info.write(&mut bytes).unwrap();
        tx.write(START_ADDR, bytes);

        Self::open(tx)
    }

    fn allocate_pages(&mut self, count: usize) -> Result<PageNo> {
        let page_no = self.mem_info.next_page;
        let addr = PAGE_SIZE as Addr * page_no as Addr;
        if !self.valid_range(addr, PAGE_SIZE * count) {
            return Err(Error::NoSpaceLeft);
        }
        trace!("Allocating {} pages at addr: 0x{:x}", count, addr);
        self.mem_info.next_page += count as u32;
        Ok(page_no)
    }

    pub fn alloc_addr(&mut self, size: usize) -> Result<Addr> {
        assert!(size > 0);
        // Trying to find a free block
        let mut free_block = self.mem_info.free_list;
        let mut prev_free_block: Option<Ptr<FreeBlock>> = None;
        while let Some(ptr) = free_block {
            let block = self.lookup(ptr)?;
            if block.size >= size as u32 {
                // Found a block that fits the size. Marking it as allocated
                let allocated = AllocatedBlock { size: block.size };
                self.write_at(Ptr::from_addr(ptr.addr).unwrap(), allocated)?;

                // Removing the block from the free list
                if let Some(prev) = prev_free_block {
                    let mut prev_block = self.lookup(prev)?;
                    prev_block.next = block.next;
                    self.update(&prev_block)?;
                } else {
                    self.mem_info.free_list = block.next;
                }

                // We need to return the address after the header
                return Ok(ptr.addr + AllocatedBlock::SIZE as Addr);
            } else {
                prev_free_block = free_block;
                free_block = block.next;
            }
        }

        // Creating new allocation
        let block_size = (size + AllocatedBlock::SIZE)
            .max(FreeBlock::SIZE)
            .max(MIN_ALLOC_SIZE);
        let addr = self.mem_info.next_addr;
        if !self.tx.valid_range(addr, block_size) {
            return Err(Error::NoSpaceLeft);
        }
        let (start_page, _) = split_addr(addr);
        let (end_page, end_page_offset) = split_addr(addr + block_size as Addr - 1);

        assert!(block_size <= PAGE_SIZE, "Object size is too big");

        let addr = if addr == 0 || start_page != end_page {
            // object doesn't fit in a page. Allocating a new page
            // TODO: we probably need to update last allocated object info, so that it can possibly be reused
            let new_page = self.allocate_pages(required_pages_cnt(block_size))?;
            new_page as Addr * PAGE_SIZE as Addr
        } else {
            addr
        };
        self.mem_info.next_addr = if end_page_offset as usize == PAGE_SIZE - 1 {
            // we're on the end of the page, zeroing next_addr, so a new page will be allocated next time
            0
        } else {
            addr + block_size as Addr
        };

        self.write_at(
            Ptr::from_addr(addr).unwrap(),
            AllocatedBlock { size: size as u32 },
        )?;

        Ok(addr + AllocatedBlock::SIZE as Addr)
    }

    pub fn alloc<T: Record>(&mut self) -> Result<Ptr<T>> {
        let addr = self.alloc_addr(T::SIZE)?;
        Ok(Ptr::from_addr(addr).unwrap())
    }

    pub fn write_at<T: Record>(&mut self, ptr: Ptr<T>, value: T) -> Result<Handle<T>> {
        let ptr = self.write_to_memory(&value, Some(ptr))?;
        let addr = ptr.addr;
        Ok(Handle { addr, value })
    }

    pub fn write<T: Record>(&mut self, value: T) -> Result<Handle<T>> {
        let ptr = self.write_to_memory(&value, None)?;
        let addr = ptr.addr;
        Ok(Handle { addr, value })
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<SlicePtr<u8>> {
        assert!(bytes.len() <= PageOffset::MAX as usize);

        let size = SLICE_HEADER_SIZE + bytes.len();
        let ptr = SlicePtr::from_addr(self.alloc_addr(size)?).expect("Alloc failed");

        let mut buffer = vec![0u8; size];

        (bytes.len() as u32)
            .write(&mut buffer[0..SLICE_HEADER_SIZE])
            .unwrap();

        buffer[SLICE_HEADER_SIZE..].copy_from_slice(bytes);

        let addr = ptr.0.addr;
        let len = buffer.len();
        self.tx.write(addr, buffer);
        trace_debug_memory_written::<&[u8]>(addr, len);

        Ok(ptr)
    }

    /// Writes object to a given address or allocates new memory for an object and writes to it
    /// Returns the pointer and the size of the block
    fn write_to_memory<T: Record>(&mut self, value: &T, ptr: Option<Ptr<T>>) -> Result<Ptr<T>> {
        let mut buffer = vec![0u8; T::SIZE];

        let size = buffer.len();
        let ptr = if let Some(ptr) = ptr {
            ptr
        } else {
            Ptr::from_addr(self.alloc_addr(size)?).expect("Alloc failed")
        };

        value.write(&mut buffer)?;
        {
            let this = &mut *self;
            let addr = ptr.addr;
            this.tx.write(addr, buffer)
        };
        trace_debug_memory_written::<T>(ptr.addr, size);
        Ok(ptr)
    }

    pub fn update<T: Record>(&mut self, handle: &Handle<T>) -> Result<()> {
        self.write_to_memory(&handle.value, Some(handle.ptr()))?;
        Ok(())
    }

    #[allow(private_bounds)]
    pub fn reclaim(&mut self, ptr: impl AsAddr) -> Result<()> {
        self.reclaim_addr(ptr.addr())
    }

    pub fn reclaim_addr(&mut self, addr: Addr) -> Result<()> {
        // Reading allocation size from the header
        let ptr = Ptr::from_addr(addr - AllocatedBlock::SIZE as Addr).unwrap();
        let block = self.lookup::<AllocatedBlock>(ptr)?;

        // Marking block as free
        self.tx.reclaim(ptr.addr, block.size as usize);
        let free_block = FreeBlock {
            size: block.size,
            next: self.mem_info.free_list,
        };
        let ptr = Ptr::from_addr(block.addr).unwrap();
        self.write_at(ptr, free_block)?;

        // Updating free list head
        self.mem_info.free_list = Some(ptr);
        Ok(())
    }

    pub fn finish(self) -> Result<S> {
        let Memory { mut tx, mem_info } = self;
        mem_info.update(&mut tx);
        Ok(tx)
    }
}

fn read_value<T: Record>(ptr: Ptr<T>, bytes: &[u8]) -> Result<Handle<T>> {
    match T::read(bytes) {
        Ok(value) => Ok(Handle {
            addr: ptr.addr,
            value,
        }),
        Err(err) => Err(Error::RecordReadError {
            addr: ptr.addr,
            type_name: type_name::<T>(),
            err: Box::new(err),
        }),
    }
}

fn trace_debug_memory_written<T>(addr: Addr, size: usize) {
    trace!(
        "Writing to addr: {:8} - {:8} type: {}",
        addr,
        addr + size as Addr,
        type_name::<T>(),
    );
}

/// The number of pages required to fit an object of a given size
fn required_pages_cnt(size: usize) -> usize {
    // TODO tests
    size.div_ceil(PAGE_SIZE)
}

pub struct Ptr<T> {
    addr: Addr,
    _phantom: PhantomData<T>,
}

impl<T> Ptr<T> {
    /// TODO add unchecked version
    pub fn from_addr(addr: Addr) -> Option<Self> {
        (addr > 0).then_some(Self {
            addr,
            _phantom: PhantomData::<T>,
        })
    }
}

impl<T: 'static> Record for Ptr<T> {
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

trait AsAddr {
    fn addr(&self) -> Addr;
}

impl<T> AsAddr for Ptr<T> {
    fn addr(&self) -> Addr {
        self.addr
    }
}

impl<T> AsAddr for SlicePtr<T> {
    fn addr(&self) -> Addr {
        self.0.addr
    }
}

/// Need to implement Copy manually, because derive(Copy) is automatically
/// adding `T: Copy` bound in auto-generated implementation
impl<T> Copy for Ptr<T> {}

#[derive(Record)]
pub struct SlicePtr<T: 'static>(Ptr<T>);

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
pub trait Record: Sized + 'static {
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

impl<T: Record> Record for Option<T> {
    // TODO write tests. It is oncorrect if T alignment is more than 1 byte
    const SIZE: usize = T::SIZE + 1;

    fn read(data: &[u8]) -> Result<Self> {
        if data[0] != 0 {
            T::read(&data[1..]).map(Some)
        } else {
            Ok(None)
        }
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        match self {
            Some(value) => {
                data[0] = 1;
                value.write(&mut data[1..])
            }
            None => {
                data.fill(0);
                Ok(())
            }
        }
    }
}

pub struct Handle<T> {
    addr: Addr,
    value: T,
}

impl<T> Handle<T> {
    // TODO remove
    pub fn new(addr: Addr, value: T) -> Self {
        Handle { addr, value }
    }

    pub fn ptr(&self) -> Ptr<T> {
        Ptr {
            addr: self.addr,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn into_inner(self) -> T {
        self.value
    }

    pub fn as_addr_and_ref(&self) -> (Addr, &T) {
        (self.addr, &self.value)
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

impl<T: Debug> Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("addr", &self.addr)
            .field("value", &self.value)
            .finish()
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

/// This wrapper exists primarily to opt out of default implementation of `Record` trait
/// for arrays. We want to have a custom implementation that will write the array directly as slice.
pub struct Blob<const N: usize>([u8; N]);

impl<const N: usize> Record for Blob<N> {
    const SIZE: usize = N;

    fn read(data: &[u8]) -> Result<Self> {
        assert!(data.len() == N);

        let mut result = [0; N];
        result.copy_from_slice(data);
        Ok(Self(result))
    }

    fn write(&self, data: &mut [u8]) -> Result<()> {
        assert!(data.len() == N);

        data.copy_from_slice(&self.0);
        Ok(())
    }
}

impl<const N: usize> Default for Blob<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> Deref for Blob<N> {
    type Target = [u8; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for Blob<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Record)]
pub struct SlotsState<T: Record> {
    /// Index of the next available slot for allocation.
    ///
    /// This value represents the highest slot number that has been allocated + 1.
    /// When allocating new slots, this value is incremented.
    next_slot: Ptr<Slot<T>>,

    /// Head of a free list
    ///
    /// Index of a first slot in a free list. It points to the [`Slot::Free`]
    /// structure.
    free_slot: Option<Ptr<Slot<T>>>,
}

/// A memory management system for fixed-size slots.
///
/// Key features:
/// - O(1) time complexity for allocation, deallocation, and access operations.
/// - Slots can be reused after deallocation, improving memory efficiency.
/// - Each slot is assigned a stable identifier (claim) that remains valid for the lifetime of the allocation.
/// - Slot occupancy is tracked, preventing use-after-free errors by returning None for deallocated slots.
pub struct Slots<T: Record> {
    state: SlotsState<T>,
    _phantom: PhantomData<T>,
}

#[derive(Record)]
#[repr(u8)]
enum Slot<T: Record> {
    /// Contains slot claim of the next free slot
    Free(Option<Ptr<Slot<T>>>) = 0,
    Occupied(T) = 1,
}

impl<T: Record> Slot<T> {
    fn occupied(self) -> Option<T> {
        match self {
            Slot::Occupied(value) => Some(value),
            Slot::Free(_) => None,
        }
    }
}

// TODO SlotMemory should contains its address like PageAllocator
impl<T: Record> Slots<T> {
    const SLOT_SIZE: usize = Slot::<T>::SIZE;
    const VALUE_OFFSET: usize = 1;

    pub fn open(state: SlotsState<T>) -> Self {
        Self {
            state,
            _phantom: PhantomData,
        }
    }

    pub fn init(mem: &mut Memory<impl TxWrite>) -> Result<Self> {
        let page_no = mem.allocate_pages(1)?;
        let state = SlotsState {
            free_slot: None,
            next_slot: Ptr::from_addr(PAGE_SIZE as Addr * page_no as Addr).unwrap(),
        };
        Ok(Self {
            state,
            _phantom: PhantomData,
        })
    }

    pub fn allocate_and_write(
        &mut self,
        mem: &mut Memory<impl TxWrite>,
        value: T,
    ) -> Result<Handle<T>> {
        if let Some(free_slot) = self.state.free_slot {
            // Reusing slot from a free list
            let mut slot_handle = mem.lookup(free_slot)?;
            let Slot::Free(next_free_slot) = *slot_handle else {
                panic!("Slot is not free {:x}", free_slot.addr);
            };
            self.state.free_slot = next_free_slot;

            // Occupying the slot
            *slot_handle = Slot::Occupied(value);
            mem.update(&slot_handle)?;
            let addr = slot_handle.addr + Self::VALUE_OFFSET as Addr;
            let value = slot_handle.into_inner().occupied().unwrap();
            let handle = Handle { addr, value };

            Ok(handle)
        } else {
            let addr = self.state.next_slot.addr;
            let (start_page, _) = split_addr(addr);
            let (end_page, end_offset) = split_addr(addr + Self::SLOT_SIZE as Addr - 1);

            // If we're on a page boundary page is not allocated yet
            if start_page != end_page {
                self.state.next_slot = self.allocate_page(mem)?;
            }
            let addr = self.state.next_slot.addr;

            let handle = Handle {
                addr: self.state.next_slot.addr,
                value: Slot::Occupied(value),
            };
            mem.update(&handle)?;
            self.state.next_slot = Ptr::from_addr(addr + Self::SLOT_SIZE as Addr).unwrap();

            let handle = Handle {
                addr: handle.ptr().addr + Self::VALUE_OFFSET as Addr,
                value: handle.into_inner().occupied().unwrap(),
            };
            if end_offset as Addr == PAGE_SIZE as Addr - 1 {
                self.state.next_slot = self.allocate_page(mem)?;
            }

            Ok(handle)
        }
    }

    pub fn read(&self, tx: &impl TxRead, slot: Ptr<T>) -> Result<Option<Handle<T>>> {
        let slot_addr = slot.addr - Self::VALUE_OFFSET as Addr;
        let ptr = Ptr::<Slot<T>>::from_addr(slot_addr).unwrap();
        let slot = tx.lookup(ptr)?;
        if let Slot::Occupied(value) = slot.into_inner() {
            Ok(Some(Handle {
                addr: slot_addr + Self::VALUE_OFFSET as Addr,
                value,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn free(&mut self, tx: &mut impl TxWrite, handle: Handle<T>) -> Result<()> {
        // Creating ptr to an outer Slot structure
        let addr = handle.ptr().addr - Self::VALUE_OFFSET as Addr;
        let ptr = Ptr::<Slot<T>>::from_addr(addr).unwrap();

        let mut slot_handle = tx.lookup(ptr)?;
        assert!(
            matches!(*slot_handle, Slot::Occupied(_)),
            "Double free at ptr 0x{:x}",
            ptr.addr
        );
        *slot_handle = Slot::Free(self.state.free_slot);
        tx.update(&slot_handle)?;
        self.state.free_slot = Some(slot_handle.ptr());
        Ok(())
    }

    fn allocate_page(&mut self, mem: &mut Memory<impl TxWrite>) -> Result<Ptr<Slot<T>>> {
        assert!(
            Self::SLOT_SIZE <= PAGE_SIZE,
            "Object type it larger that page size",
        );
        let page_no = mem.allocate_pages(1)?;
        let page_addr = page_no as Addr * PAGE_SIZE as Addr;
        Ok(Ptr::from_addr(page_addr).unwrap())
    }

    pub fn finish(self) -> SlotsState<T> {
        self.state
    }
}

pub trait TxReadExt: TxRead {
    fn lookup<T: Record>(&self, ptr: Ptr<T>) -> Result<Handle<T>> {
        let buf = self.read(ptr.addr, T::SIZE);
        read_value(ptr, &buf)
    }

    fn read_bytes(&self, ptr: SlicePtr<u8>) -> Result<Cow<[u8]>> {
        let addr = ptr.0.addr;
        let items = self.read_static::<SLICE_HEADER_SIZE>(addr);

        let bytes = u32::from_le_bytes(items) as usize;
        Ok(self.read(addr + SLICE_HEADER_SIZE as Addr, bytes))
    }

    fn read_static<const N: usize>(&self, addr: Addr) -> [u8; N] {
        let mut ret = [0; N];
        let bytes = self.read(addr, N);
        for (to, from) in ret.iter_mut().zip(bytes.iter()) {
            *to = *from;
        }
        ret
    }
}

impl<S: TxRead> TxReadExt for S {}

pub trait TxWriteExt: TxWrite {
    fn update<T: Record>(&mut self, r: &Handle<T>) -> Result<()> {
        let (addr, value) = r.as_addr_and_ref();
        let mut buffer = vec![0u8; T::SIZE];
        value.write(&mut buffer)?;
        self.write(addr, buffer);
        Ok(())
    }
}

impl<S: TxWrite> TxWriteExt for S {}

/// Virtual Memory Management
///
/// Virtual memory allows the creation of several contiguous memory spaces inside a single transaction.
/// It can be useful to separate independent data into virtual memory spaces that guarantee
/// different clients will not overwrite each other's data.
///
/// ## Mapping
///
/// 1. The virtual address space is divided into fixed-size pages.
/// 2. A two-level page table structure is used for address translation.
/// 3. Each virtual page number is split into two parts (14 bits each), used as indices in the page tables.
///
/// The virtual page number is split in the following way:
/// - 4 leading bits are not used and should be 0
/// - 14 bits are an index into the outer PTE table (root table)
/// - 14 bits are an index into the inner PTE table
///
/// Each entry in the root PTE table is a page number of an inner PTE table. Each entry in the inner PTE table
/// is a physical page number.
mod vmem {
    use super::*;
    use crate::volume::{make_addr, page_segments};
    use std::{array, cell::RefCell, rc::Rc};
    use tracing::{debug, info};

    const INFO_ADDR: Addr = 8;

    #[derive(Record)]
    struct VMemInfo {
        /// next available for allocation page no.
        next_page: PageNo,

        /// The number of virtual spaces. On runtime should be equal or less than `N`.
        ///
        /// It is allowed to open `VMem<N>` as `VMem<N - 1>` (for N > 1)
        spaces: u16,
    }

    #[derive(Record)]
    struct VMemRootPages<const N: usize> {
        /// page numbers for root page entry tables for all virtual spaces
        root_ptes: [PageNo; N],
    }

    pub fn init<const N: usize, T: TxWrite>(mut tx: T) -> Result<[VTx<T>; N]> {
        info!(N, "Initializing VMem");
        let info = VMemInfo {
            spaces: u16::try_from(N).expect("N is too large"),
            // page=0 is reserved, first N pages occupied by root pte pages
            // first available for allocation page is N + 1
            next_page: N as PageNo + 1,
        };
        let root_pages = VMemRootPages::<N> {
            // root_ptes are first N pages started from 1 (eg. [1, 2, 3, ...])
            root_ptes: array::from_fn(|i| i as PageNo + 1),
        };
        tx.update(&Handle::new(INFO_ADDR, info))?;
        tx.update(&Handle::new(INFO_ADDR + VMemInfo::SIZE as Addr, root_pages))?;
        open(tx)
    }

    pub fn open<const N: usize, T: TxRead>(tx: T) -> Result<[VTx<T>; N]> {
        assert!(N > 0);
        let info = tx.lookup(Ptr::<VMemInfo>::from_addr(INFO_ADDR).unwrap())?;
        assert!(
            info.spaces as usize <= N,
            "VMmem configured with {} virtual spaces",
            info.spaces,
        );
        let root_pages = tx.lookup(
            Ptr::<VMemRootPages<N>>::from_addr(INFO_ADDR + VMemInfo::SIZE as Addr).unwrap(),
        )?;
        info!(next_page = info.next_page, N, "Opening VMem");
        for (i, pte) in root_pages.root_ptes.iter().enumerate() {
            assert!(*pte > 0, "Root Page Table number is missing");
            debug!(i, root_pte_page = pte, "Root PTE");
        }

        let tx = Rc::new(RefCell::new(tx));
        let info = Rc::new(RefCell::new(info));

        Ok(array::from_fn(|i| VTx {
            tx: Rc::clone(&tx),
            info: Rc::clone(&info),
            root_pt: root_pages.root_ptes[i],
        }))
    }

    pub fn finish<const N: usize, T: TxWrite>(txs: [VTx<T>; N]) -> Result<T> {
        // We need to drop all transactions except one. After that we should be
        // able to move out of transaction Rc
        let mut txs = txs.into_iter().collect::<Vec<_>>();
        txs.drain(1..);
        let last_vmem = txs.remove(0);

        let mut tx = Rc::into_inner(last_vmem.tx)
            .expect("Unable to take ownership of transaction")
            .into_inner();
        tx.update(&last_vmem.info.borrow())?;
        Ok(tx)
    }

    pub struct VTx<T> {
        /// Global transaction
        tx: Rc<RefCell<T>>,
        info: Rc<RefCell<Handle<VMemInfo>>>,
        root_pt: PageNo,
    }

    impl<T: TxRead> VTx<T> {
        /// Translate virtual page number to a physical one
        fn translate_page(&self, v_page: PageNo) -> Result<Option<PageNo>> {
            let (pt_1_idx, pt_2_idx) = split_pte_idx(v_page);
            let tx = self.tx.borrow();

            let pte_page = tx.lookup(pte_ptr(self.root_pt, pt_1_idx))?;
            let phys_page_no = if *pte_page > 0 {
                *tx.lookup(pte_ptr(*pte_page, pt_2_idx))?
            } else {
                0
            };
            Ok(Some(phys_page_no).filter(|&i| i > 0))
        }

        #[cfg(test)]
        #[allow(unused)]
        fn dump_translation_table(&self) {
            const PTE_ENTRIES: usize = PAGE_SIZE / PageNo::SIZE;
            let tx = self.tx.borrow();
            let info = self.info.borrow();
            let root_ptr =
                Ptr::<[PageNo; PTE_ENTRIES]>::from_addr(self.root_pt as Addr * PAGE_SIZE as Addr)
                    .unwrap();
            let mut v_page = 0;
            let root_pt = tx.lookup(root_ptr).unwrap();
            for pte1 in root_pt.into_iter().take_while(|&i| i > 0) {
                let pt_2_ptr =
                    Ptr::<[PageNo; PTE_ENTRIES]>::from_addr(pte1 as Addr * PAGE_SIZE as Addr)
                        .unwrap();
                let pt_2 = tx.lookup(pt_2_ptr).unwrap();
                for p_page in pt_2.into_iter().take_while(|&i| i > 0) {
                    println!("   {} -> {}", v_page, p_page);
                    v_page += 1;
                }
            }
        }
    }

    /// Splits virtual page no. into 2 PTE tables offsets
    /// - 1st is an offset in an outer (root) PTE table
    /// - 2nd is an offset in an inner PTE table
    ///
    /// Both indices are read from corresponding 14 bits of an virtual page no.
    fn split_pte_idx(v_page: u32) -> (u32, u32) {
        assert_eq!(PAGE_SIZE, 65536, "All PTE calculations are designed around page size of 65536 bytes. Please correct all the computations when changing page size");

        // The size of PTE index in bits. Each PTE table has 2^14 entries.
        const PTE_ENTRY_BITS: usize = 14;
        const PTE_BITS_MASK: u32 = (1 << PTE_ENTRY_BITS) - 1;

        let pte_1_idx = (v_page >> PTE_ENTRY_BITS) & PTE_BITS_MASK;
        let pte_2_idx = v_page & PTE_BITS_MASK;

        // 2 level page mapping sing 14 bits allows us to map map 2^14 * 2^14 = 2^28 entries.
        // Most significant 4 bits must be set to zero
        assert!(
            v_page >> (2 * PTE_ENTRY_BITS) == 0,
            "Leading 4 bits of a virtual page no. should be zero"
        );
        (pte_1_idx, pte_2_idx)
    }

    fn pte_ptr(page_no: PageNo, index: u32) -> Ptr<u32> {
        let pt = page_no as Addr;
        let index = index as Addr;
        let page_addr = pt * PAGE_SIZE as Addr;
        let offset = index * PageNo::SIZE as Addr;
        Ptr::<PageNo>::from_addr(page_addr + offset).unwrap()
    }

    impl<T: TxWrite> VTx<T> {
        /// Translate virtual page number to a physical one and allocates a page
        /// it there is no mapping for a page
        fn translate_allocate_page(&self, v_page: PageNo) -> Result<PageNo> {
            let (pt_1_idx, pt_2_idx) = split_pte_idx(v_page);

            let mut tx = self.tx.borrow_mut();
            let mut pte_1 = tx.lookup(pte_ptr(self.root_pt, pt_1_idx))?;

            if *pte_1 == 0 {
                *pte_1 = self.allocate_physical_page(&tx);
                tx.update(&pte_1)?;
                debug!(page = *pte_1, level = 1, "Allocating new PTE page");
            }

            let mut pte_2 = tx.lookup(pte_ptr(*pte_1, pt_2_idx))?;
            if *pte_2 == 0 {
                *pte_2 = self.allocate_physical_page(&tx);
                debug!(page = *pte_2, level = 2, "Allocating new page");
                tx.update(&pte_2)?;
            }
            Ok(*pte_2)
        }

        fn allocate_physical_page(&self, tx: &T) -> PageNo {
            let mut info = self.info.borrow_mut();
            let page = info.next_page;
            assert!(
                tx.valid_range(PAGE_SIZE as u64 * page as u64, PAGE_SIZE),
                "Out-of-bounds"
            );
            info.next_page += 1;
            page
        }
    }

    impl<T: TxRead> TxRead for VTx<T> {
        fn read_to_buf(&self, v_addr: Addr, buf: &mut [u8]) {
            let segments = page_segments(v_addr, buf.len());
            // dbg!(v_addr, buf.len());
            for (v_addr, range) in segments {
                let (v_page, offset) = split_addr(v_addr);
                // dbg!(v_page, offset);
                if let Some(p_page) = self.translate_page(v_page).unwrap() {
                    dbg!(p_page);
                    let tx = self.tx.borrow();
                    let p_addr = make_addr(p_page, offset);
                    trace!(v_page, v_addr, p_page, p_addr, "Reading");
                    tx.read_to_buf(p_addr, &mut buf[range]);
                }
            }
        }

        fn valid_range(&self, addr: Addr, len: usize) -> bool {
            let tx = self.tx.borrow();
            tx.valid_range(addr, len)
        }
    }

    impl<T: TxWrite> TxWrite for VTx<T> {
        fn write(&mut self, v_addr: Addr, bytes: impl Into<Vec<u8>>) {
            // dbg!(v_addr);
            let bytes = bytes.into();
            for (v_addr, range) in page_segments(v_addr, bytes.len()) {
                let (v_page, offset) = split_addr(v_addr);
                let p_page = self.translate_allocate_page(v_page).unwrap();
                let p_addr = make_addr(p_page, offset);
                trace!(v_page, v_addr, p_page, p_addr, "Writing");
                let mut tx = self.tx.borrow_mut();
                tx.write(p_addr, &bytes[range]);
            }
        }

        fn reclaim(&mut self, v_addr: Addr, len: usize) {
            for (v_addr, range) in page_segments(v_addr, len) {
                let (v_page, offset) = split_addr(v_addr);
                if let Some(p_page) = self.translate_page(v_page).unwrap() {
                    // We only need to reclaim memory if it was allocated
                    let p_addr = make_addr(p_page, offset);
                    trace!(v_page, v_addr, p_page, p_addr, "Reclaiming");
                    let mut tx = self.tx.borrow_mut();
                    tx.reclaim(p_addr, range.len());
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{
            assert_buffers_eq,
            volume::{
                tests::{
                    proptests::{any_snapshot, DB_SIZE},
                    VecTx,
                },
                Volume,
            },
        };
        use proptest::{collection::vec, proptest};

        #[test]
        fn simple_case() -> Result<()> {
            let v = Volume::new_in_memory(10);
            let [mut tx1, mut tx2] = vmem::init(v.start())?;

            tx1.write(0, b"Jekyll");
            tx2.write(0, b"Hide");

            assert_eq!(&*tx1.read(0, 6), b"Jekyll");
            assert_eq!(&*tx2.read(0, 4), b"Hide");
            Ok(())
        }

        #[test]
        fn reopen() -> Result<()> {
            let mut v = Volume::new_in_memory(10);
            let [mut tx1, mut tx2] = vmem::init(v.start())?;

            tx1.write(0, b"Jekyll");
            tx2.write(0, b"Hide");
            v.commit(vmem::finish([tx1, tx2])?).unwrap();

            let [tx1, tx2] = vmem::open(v.snapshot())?;
            assert_eq!(&*tx1.read(0, 6), b"Jekyll");
            assert_eq!(&*tx2.read(0, 4), b"Hide");
            Ok(())
        }

        proptest! {
            #[test]
            fn shadow_write_vmem_2_tx(snapshots in vec((any_snapshot(), any_snapshot()), 0..3)) {
                let mut shadow_a = VecTx(vec![0; DB_SIZE]);
                let mut shadow_b = VecTx(vec![0; DB_SIZE]);
                let mut vol = Volume::with_capacity(5 * DB_SIZE);

                // Initializing vmem in a separate commit
                {
                    let tx = vmem::finish(vmem::init::<2, _>(vol.start())?)?;
                    vol.commit(tx)?;
                }

                for (patches_a, patches_b) in snapshots {
                    let [mut vm_a, mut vm_b] = vmem::open(vol.start())?;

                    for p in patches_a {
                        p.clone().write_to(&mut vm_a);
                        p.write_to(&mut shadow_a);
                    }
                    for p in patches_b {
                        p.clone().write_to(&mut vm_b);
                        p.write_to(&mut shadow_b);
                    }
                    assert_buffers_eq!(vm_a.read(0, DB_SIZE), shadow_a);
                    assert_buffers_eq!(vm_b.read(0, DB_SIZE), shadow_b);

                    vol.commit(vmem::finish([vm_a, vm_b])?)?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::{Transaction, Volume, PAGE_SIZE};
    use pmem_derive::Record;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
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
    fn no_space_left_error_should_be_generated() {
        let (_, mut mem) = create_memory(10);
        let result = mem.allocate_pages(100);
        let Err(Error::NoSpaceLeft) = result else {
            panic!(
                "Err({:?}) should be generated, {:?} instead",
                Error::NoSpaceLeft,
                result
            );
        };
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
    fn memory_can_persist_over_commits() -> Result<()> {
        let mut volume = Volume::default();
        let mut mem = Memory::init(volume.start());

        let mut ptrs = vec![];

        for i in 0..10 {
            let handle = mem.write(Value(i))?;
            ptrs.push(handle.ptr());
            volume.commit(mem.finish()?).unwrap();
            mem = Memory::open(volume.start());
        }

        for (i, ptr) in ptrs.into_iter().enumerate() {
            let value = mem.read(ptr)?;
            assert_eq!(value, Value(i as u32));
        }
        Ok(())
    }

    #[test]
    fn memory_can_reuse_memory() -> Result<()> {
        let volume = Volume::with_capacity(2 * PAGE_SIZE);
        let mut mem = Memory::init(volume.start());
        let mut rng = SmallRng::from_entropy();

        // Each allocation of N bytes will required N + 4 bytes in total (size of [`AllocatedBlock`]).
        // It means that if we will allocate one slot for each possible memory size up to N bytes, we will need
        // (4 + 1) + (4 + 2) + ... + (4 + N) = 4N + N(N + 1) / 2 bytes in total.
        //
        // We need to ensure that this value is less than the size of the volume. This way we can be sure that
        // even if we will allocate slots in most inefficient way possible (first 1 bytes, then 2 bytes and so on),
        // so we can not reuse any of previously allocated slots, we will still have enough space to store all of them.
        //
        // For the volume of size `PAGE_SIZE`:
        // 4N + N(N + 1) / 2 < P, where P = PAGE_SIZE
        // N^2 + 9N - 2P < 0.
        // The only positive solution is:
        // N = (-9 + sqrt(9^2 + 4 * 2 * P)) / 2
        // N = 357.56..
        const MAX_ALLOCATION_SIZE: usize = 357;

        for _ in 0..10_000 {
            // TODO correct computation of max size
            let size = rng.gen_range(1..=MAX_ALLOCATION_SIZE);
            let ptr = mem.alloc_addr(size)?;
            mem.reclaim_addr(ptr)?;
        }

        Ok(())
    }

    #[test]
    fn check_memory_will_not_be_rewritten_when_reuse() -> Result<()> {
        let volume = Volume::with_capacity(PAGE_SIZE * 3);
        let mut mem = Memory::init(volume.start());

        let a = mem.write(1u32)?;
        let b = mem.write(42u32)?;

        let a_addr = a.ptr().addr();
        // Two allocations must be adjacent in memory. Otherwise test is invalid.
        assert_eq!(a_addr + MIN_ALLOC_SIZE as Addr, b.ptr().addr());

        mem.reclaim(a.ptr())?;
        let a = mem.write(3u32)?;

        // Realocating memory for the same type should return the same address.
        assert_eq!(a.ptr().addr(), a_addr);

        // b should not be overwritten.
        let b = mem.read(b.ptr())?;
        assert_eq!(b, 42);

        Ok(())
    }

    #[test]
    fn check_memory_will_not_be_rewritten_when_reuse_with_larger_size() -> Result<()> {
        let volume = Volume::with_capacity(PAGE_SIZE * 3);
        let mut mem = Memory::init(volume.start());

        let a = mem.write(1u32)?;
        let b = mem.write(42u32)?;

        let a_addr = a.ptr().addr();
        // Two allocations must be adjacent in memory. Otherwise test is invalid.
        assert_eq!(a_addr + MIN_ALLOC_SIZE as Addr, b.ptr().addr());

        mem.reclaim(a.ptr())?;

        // Writing twice as large value. It should not overwrite b.
        let a = mem.write(3u64)?;
        assert_ne!(a.ptr().addr(), a_addr);

        let b = mem.read(b.ptr())?;
        assert_eq!(b, 42);

        Ok(())
    }

    #[test]
    fn memory_can_reuse_memory_after_reopen() -> Result<()> {
        let mut volume = Volume::with_capacity(PAGE_SIZE * 3);
        let mut mem = Memory::init(volume.start());

        let ptr_a = mem.alloc::<[u8; PAGE_SIZE - 4]>()?;
        let ptr_b = mem.alloc::<[u8; PAGE_SIZE - 4]>()?;
        mem.reclaim(ptr_a)?;
        mem.reclaim(ptr_b)?;
        assert!(
            mem.mem_info.free_list.is_some(),
            "Free list should not be empty"
        );

        volume.commit(mem.finish()?).unwrap();

        let mut mem = Memory::open(volume.start());
        assert!(
            mem.mem_info.free_list.is_some(),
            "Free list should not be empty"
        );
        mem.alloc::<[u8; PAGE_SIZE - 4]>()?;
        mem.alloc::<[u8; PAGE_SIZE - 4]>()?;

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
        mem.reclaim(ptr)?;
        let result = mem.read(ptr)?;
        // should be zero, because we use zero-fill semantics
        assert_eq!(result, Value(0));
        Ok(())
    }

    #[test]
    fn slot_allocator_simple_allocate() -> Result<()> {
        let (mut tx, mut slots) = create_slot_memory();

        let slot = slots.read(&tx, Ptr::from_addr(13).unwrap())?;
        assert!(slot.is_none());

        let handle_42 = slots.allocate_and_write(&mut tx, 42)?;
        let handle_43 = slots.allocate_and_write(&mut tx, 43)?;

        let slot = slots.read(&tx, handle_42.ptr())?;
        assert_eq!(slot.as_deref(), Some(&42));

        let slot = slots.read(&tx, handle_43.ptr())?;
        assert_eq!(slot.as_deref(), Some(&43));
        Ok(())
    }

    #[test]
    fn slot_page_boundary_allocate() -> Result<()> {
        check_type_correctness::<[u8; 31]>(&[0; 31], &[1; 31])
    }

    fn check_type_correctness<T: Record + Clone + Debug + PartialEq>(
        a_value: &T,
        b_value: &T,
    ) -> Result<()> {
        assert!(
            PAGE_SIZE % Slot::<T>::SIZE == 0,
            "For test to work correctly page size {} should be divisible by the type <{}> size: {}",
            PAGE_SIZE,
            type_name::<T>(),
            Slot::<T>::SIZE
        );
        let (mut tx, mut slots) = create_slot_memory();

        let count = PAGE_SIZE / Slot::<T>::SIZE * 2;

        let expected_values = [a_value.clone(), b_value.clone()]
            .iter()
            .cycle()
            .take(count)
            .cloned()
            .collect::<Vec<_>>();

        let handles = expected_values
            .iter()
            .cloned()
            .map(|val| slots.allocate_and_write(&mut tx, val))
            .collect::<Result<Vec<_>>>()?;

        for (handle, expected) in handles.iter().zip(expected_values.iter()) {
            let value = slots.read(&tx, handle.ptr())?.unwrap();
            assert_eq!(expected, &*value);
        }
        Ok(())
    }

    #[test]
    fn check_slot_sizes() {
        assert_eq!(Option::<Ptr<Slot<[u8; 1]>>>::SIZE, 9);
        assert_eq!(Slot::<[u8; 1]>::SIZE, 10);
    }

    #[test]
    fn slot_can_be_updated() -> Result<()> {
        let (mut tx, mut slots) = create_slot_memory();

        let mut handle = slots.allocate_and_write(&mut tx, 13)?;
        *handle = 42;
        tx.update(&handle)?;

        let value = slots.read(&tx, handle.ptr())?.unwrap();
        assert_eq!(*value, 42);
        Ok(())
    }

    #[test]
    fn slot_memory_can_be_reopen() -> Result<()> {
        let (mut tx, mut slots) = create_slot_memory::<u64>();

        let handle = slots.allocate_and_write(&mut tx, 42)?;

        let slots = Slots::<u64>::open(slots.finish());
        let value = slots.read(&tx, handle.ptr())?.unwrap();
        assert_eq!(*value, 42);
        Ok(())
    }

    #[test]
    fn slot_reallocation() -> Result<()> {
        let (mut tx, mut slots) = create_slot_memory();

        // TODO at the moment we can not relocate slot=0, only >=1
        // therefore need to fill slot=0
        slots.allocate_and_write(&mut tx, 0)?;

        let handle = slots.allocate_and_write(&mut tx, 1)?;
        slots.allocate_and_write(&mut tx, 2)?;
        let handle_ptr = handle.ptr();
        slots.free(&mut tx, handle)?;

        let new_handle = slots.allocate_and_write(&mut tx, 10)?;
        assert_eq!(new_handle.ptr(), handle_ptr);

        let value = slots.read(&tx, new_handle.ptr())?;
        assert_eq!(value.as_deref(), Some(&10));
        Ok(())
    }

    fn create_slot_memory<T: Record>() -> (Memory<Transaction>, Slots<T>) {
        let volume = Volume::new_in_memory(10);
        let tx = volume.start();
        let mut mem = Memory::init(tx);
        let slots = Slots::init(&mut mem).unwrap();

        (mem, slots)
    }

    mod proptests {
        use super::*;
        use prop::collection::vec;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn slot_can_be_reclaimed((items, target_idx) in any_items_and_target()) {
                let (mut tx, mut slots) = create_slot_memory();

                let allocated = items
                    .iter()
                    .copied()
                    .map(|value| slots.allocate_and_write(&mut tx, value))
                    .collect::<Result<Vec<_>>>()?;

                // Reclaiming target
                let target_handle = slots.read(&tx, allocated[target_idx].ptr())?.unwrap();
                slots.free(&mut tx, target_handle)?;

                for (idx, (slot, item)) in allocated.into_iter().zip(items).enumerate() {
                    let slot = slots.read(&tx, slot.ptr())?;
                    if idx == target_idx {
                        prop_assert_eq!(slot.as_deref(), None);
                    } else {
                        prop_assert_eq!(slot.as_deref(), Some(&item));
                    }
                }
            }

            #[test]
            fn slot_will_not_be_removed_until_free((items, target_idx) in any_items_and_target()) {
                let (mut tx, mut slots) = create_slot_memory();

                let target = items[target_idx];
                let mut allocated = items
                    .into_iter()
                    .map(|value| slots.allocate_and_write(&mut tx, value))
                    .collect::<Result<Vec<_>>>()?;

                // Reclaiming all but target
                let target_slot = allocated.remove(target_idx);
                for item in allocated {
                    slots.free(&mut tx, item)?;
                }

                let target_slot = slots.read(&tx, target_slot.ptr())?;
                prop_assert_eq!(target_slot.as_deref(), Some(&target));
            }

            #[test]
            fn memory_allocate_free(ops in vec(any_allocate_or_free(), 255)) {
                let volume = Volume::with_capacity(PAGE_SIZE * 10);
                let mut mem = Memory::init(volume.start());
                let mut allocations = vec![];

                for (id, op) in ops.into_iter().enumerate() {
                    let id = (id % 256) as u8;
                    match op {
                        AllocateOrFree::Allocate(size) => {
                            let addr = mem.alloc_addr(size)?;
                            allocations.push((id, addr, size));
                            TxWrite::write(&mut mem, addr, vec![id; size]);
                        }
                        AllocateOrFree::Free(idx) => {
                            if !allocations.is_empty() {
                                let (id, addr, size) = allocations.remove(idx % allocations.len());
                                let data = TxRead::read(&mem, addr, size);
                                let expected = vec![id; size];
                                prop_assert_eq!(data, expected);
                            }
                        }
                    }
                }

                // Checking that all leftover allocations are still there and have correct data
                for (id, addr, size) in allocations {
                    let data = TxRead::read(&mem, addr, size);
                    let expected = vec![id; size];
                    prop_assert_eq!(data, expected);
                }
            }
        }

        fn any_allocate_or_free() -> impl Strategy<Value = AllocateOrFree> {
            prop_oneof![
                (1usize..15).prop_map(AllocateOrFree::Allocate),
                // The concrete index is not important, because it will be taken modulo items.len()
                (0usize..100).prop_map(AllocateOrFree::Free)
            ]
        }

        #[derive(Debug, Clone, Copy)]
        enum AllocateOrFree {
            /// The size of bytes
            Allocate(usize),
            /// the index of the element to free (% items.len())
            Free(usize),
        }

        /// Returns a vectors of random ites to allocate and random index of one (target) element
        fn any_items_and_target() -> impl Strategy<Value = (Vec<u64>, usize)> {
            (1usize..10).prop_flat_map(|size| {
                let allocations = (1u64..=size as u64).collect::<Vec<_>>();
                (Just(allocations), 0..size)
            })
        }
    }

    fn create_memory(pages: u32) -> (Volume, Memory<Transaction>) {
        let volume = Volume::new_in_memory(pages);
        let mem = Memory::init(volume.start());
        (volume, mem)
    }

    impl Memory<Transaction> {
        fn new() -> Self {
            let volume = Volume::default();
            Self::init(volume.start())
        }
    }
}
