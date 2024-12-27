//! Virtual Memory Management
//!
//! Virtual memory allows the creation of several contiguous memory spaces inside a single transaction.
//! It can be useful to separate independent data into virtual memory spaces that guarantee
//! different clients will not overwrite each other's data.
//!
//! ## Mapping
//!
//! 1. The virtual address space is divided into fixed-size pages.
//! 2. A two-level page table structure is used for address translation.
//! 3. Each virtual page number is split into two parts (14 bits each), used as indices in the page tables.
//!
//! The virtual page number is split in the following way:
//! - 4 leading bits are not used and should be 0
//! - 14 bits are an index into the outer PTE table (root table)
//! - 14 bits are an index into the inner PTE table
//!
//! Each entry in the root PTE table is a page number of an inner PTE table. Each entry in the inner PTE table
//! is a physical page number.
//!
//! ## Virtual Memory Example
//!
//! This example demonstrates how to use the vmem package to create
//! and manipulate multiple virtual memory spaces within a single transaction.
//!
//! ```
//! use pmem::memory::vmem;
//! use pmem::volume::{Volume, TxRead, TxWrite};
//!
//! // Create a new volume with 10 pages
//! let mut volume = Volume::with_capacity(1024 * 1024);
//!
//! // Initialize two virtual memory spaces
//! let [mut vm1, mut vm2] = vmem::init(volume.start()).unwrap();
//!
//! // Write data to the first virtual memory space
//! vm1.write(0, b"Hello from VM1");
//!
//! // Write data to the second virtual memory space
//! vm2.write(0, b"Greetings from VM2");
//!
//! // Read data from both virtual memory spaces
//! let data1 = vm1.read(0, 14);
//! let data2 = vm2.read(0, 18);
//!
//! assert_eq!(&*data1, b"Hello from VM1");
//! assert_eq!(&*data2, b"Greetings from VM2");
//!
//! // Commit changes back to Volume
//! let tx = vmem::finish([vm1, vm2]).unwrap();
//! volume.commit(tx).unwrap();
//!
//! // Reopen the virtual memory spaces
//! let [vm1, vm2] = vmem::open(volume.snapshot()).unwrap();
//!
//! // Verify data persists after reopening
//! let data1 = vm1.read(0, 14);
//! let data2 = vm2.read(0, 18);
//!
//! assert_eq!(&*data1, b"Hello from VM1");
//! assert_eq!(&*data2, b"Greetings from VM2");
//! ```

use crate::{
    memory::{Result, TxReadExt, TxWriteExt},
    volume::{
        make_addr, page_segments, split_addr, Addr, PageNo, PageOffset, TxRead, TxWrite, PAGE_SIZE,
    },
    Handle, Ptr, Record,
};
use pmem_derive::Record;
use std::{array, cell::RefCell, rc::Rc};
use tracing::{debug, info, trace};

const INFO_ADDR: Addr = 8;

#[derive(Record)]
struct GlobalInfo {
    /// next available for allocation page no.
    next_page: PageNo,

    /// The number of virtual spaces. On runtime should be equal or less than `N`.
    ///
    /// It is allowed to open `VMem<N>` as `VMem<N - 1>` (for N > 1)
    spaces: u16,
}

/// page numbers for root page entry tables for all virtual spaces
#[derive(Record)]
struct RootTranslationTables<const N: usize>([PageNo; N]);

pub fn init<const N: usize, T: TxWrite>(mut tx: T) -> Result<[VTx<T>; N]> {
    info!(N, "Initializing VMem");
    let info = GlobalInfo {
        spaces: u16::try_from(N).expect("N is too large"),
        // page=0 is reserved, first N pages occupied by root pte pages
        // first available for allocation page is N + 1
        next_page: N as PageNo + 1,
    };
    // root_ptes are first N pages started from 1 (eg. [1, 2, 3, ...])
    let root_pages = RootTranslationTables::<N>(array::from_fn(|i| i as PageNo + 1));
    tx.update(&Handle::new(INFO_ADDR, info))?;
    tx.update(&Handle::new(
        INFO_ADDR + GlobalInfo::SIZE as Addr,
        root_pages,
    ))?;
    open(tx)
}

pub fn open<const N: usize, T: TxRead>(tx: T) -> Result<[VTx<T>; N]> {
    assert!(N > 0);
    let info = tx.lookup(Ptr::<GlobalInfo>::from_addr(INFO_ADDR).unwrap())?;
    assert!(
        info.spaces as usize <= N,
        "VMmem configured with {} virtual spaces",
        info.spaces,
    );
    let root_pages = tx.lookup(
        Ptr::<RootTranslationTables<N>>::from_addr(INFO_ADDR + GlobalInfo::SIZE as Addr).unwrap(),
    )?;
    info!(next_page = info.next_page, N, "Opening VMem");
    for (i, pte) in root_pages.0.iter().enumerate() {
        assert!(*pte > 0, "Root Page Table number is missing");
        debug!(i, root_pte_page = pte, "Root PTE");
    }

    let tx = Rc::new(RefCell::new(tx));
    let info = Rc::new(RefCell::new(info));

    Ok(array::from_fn(|i| VTx {
        tx: Rc::clone(&tx),
        info: Rc::clone(&info),
        root_pt: root_pages.0[i],
    }))
}

pub fn finish<const N: usize, T: TxWrite>(txs: [VTx<T>; N]) -> Result<T> {
    // We need to drop all transactions except one. After that we should be
    // able to move global transaction out of Rc
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
    info: Rc<RefCell<Handle<GlobalInfo>>>,
    root_pt: PageNo,
}

impl<T: TxRead> VTx<T> {
    /// Translate virtual page number to a physical one
    fn translate_page(&self, tx: &T, v_page: PageNo) -> Option<PageNo> {
        let (pt_1_idx, pt_2_idx) = split_pte_idx(v_page);

        let pte_page = tx
            .lookup(pte_ptr(self.root_pt, pt_1_idx))
            .expect("Unable to read PTE");
        let phys_page_no = if *pte_page > 0 {
            *tx.lookup(pte_ptr(*pte_page, pt_2_idx))
                .expect("Unable to read PTE")
        } else {
            0
        };
        Some(phys_page_no).filter(|&i| i > 0)
    }

    /// Diagnostic method that prints translation table for a current virtual transaction
    #[cfg(test)]
    #[allow(unused)]
    fn dump_translation_table(&self) {
        const PTE_ENTRIES: usize = PAGE_SIZE / PageNo::SIZE;
        let tx = self.tx.borrow();
        let info = self.info.borrow();
        let root_ptr = Ptr::<[PageNo; PTE_ENTRIES]>::from_page_offset(self.root_pt, 0).unwrap();
        let mut v_page = 0;
        let root_pt = tx.lookup(root_ptr).unwrap();
        for pte1 in root_pt.into_iter().take_while(|&i| i > 0) {
            let pt_2_ptr = Ptr::<[PageNo; PTE_ENTRIES]>::from_page_offset(pte1, 0).unwrap();
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
    let offset = index as PageOffset * PageNo::SIZE as PageOffset;
    Ptr::<PageNo>::from_page_offset(page_no, offset).unwrap()
}

impl<T: TxWrite> VTx<T> {
    /// Translate virtual page number to a physical one and allocates a page
    /// it there is no mapping for a page
    fn translate_allocate_page(&self, v_page: PageNo) -> PageNo {
        let (pt_1_idx, pt_2_idx) = split_pte_idx(v_page);

        let mut tx = self.tx.borrow_mut();
        let mut pte_1 = tx
            .lookup(pte_ptr(self.root_pt, pt_1_idx))
            .expect("Unable to read PTE");

        if *pte_1 == 0 {
            *pte_1 = self.allocate_physical_page(&tx);
            tx.update(&pte_1).expect("Unable to update PTE");
            debug!(page = *pte_1, level = 1, "Allocating new PTE page");
        }

        let mut pte_2 = tx
            .lookup(pte_ptr(*pte_1, pt_2_idx))
            .expect("Unable to read PTE");
        if *pte_2 == 0 {
            *pte_2 = self.allocate_physical_page(&tx);
            debug!(page = *pte_2, level = 2, "Allocating new page");
            tx.update(&pte_2).expect("Unable to update PTE");
        }
        *pte_2
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
        let tx = self.tx.borrow();
        for (v_addr, range) in segments {
            let (v_page, offset) = split_addr(v_addr);
            if let Some(p_page) = self.translate_page(&tx, v_page) {
                let p_addr = make_addr(p_page, offset);
                trace!(v_page, v_addr, p_page, p_addr, "Reading");
                tx.read_to_buf(p_addr, &mut buf[range]);
            }
        }
    }

    fn valid_range(&self, addr: Addr, len: usize) -> bool {
        self.tx.borrow().valid_range(addr, len)
    }
}

impl<T: TxWrite> TxWrite for VTx<T> {
    fn write(&mut self, v_addr: Addr, bytes: impl Into<Vec<u8>>) {
        // dbg!(v_addr);
        let bytes = bytes.into();
        for (v_addr, range) in page_segments(v_addr, bytes.len()) {
            let (v_page, offset) = split_addr(v_addr);
            let p_page = self.translate_allocate_page(v_page);
            let p_addr = make_addr(p_page, offset);
            trace!(v_page, v_addr, p_page, p_addr, "Writing");
            let mut tx = self.tx.borrow_mut();
            tx.write(p_addr, &bytes[range]);
        }
    }

    fn reclaim(&mut self, v_addr: Addr, len: usize) {
        let mut tx = self.tx.borrow_mut();
        for (v_addr, range) in page_segments(v_addr, len) {
            let (v_page, offset) = split_addr(v_addr);
            if let Some(p_page) = self.translate_page(&tx, v_page) {
                // We only need to reclaim memory if it was allocated
                let p_addr = make_addr(p_page, offset);
                trace!(v_page, v_addr, p_page, p_addr, "Reclaiming");
                tx.reclaim(p_addr, range.len());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_buffers_eq, vmem,
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
