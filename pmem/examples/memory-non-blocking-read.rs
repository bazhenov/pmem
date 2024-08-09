//! This test is exploring the algorithm of sharing a buffer between multiple threads
//! without using locks. The buffer is updated by one thread only (the writer), but may be read by
//! multiple threads (readers).
//!
//! Consistency is guaranteed by maintaining a list of inverse (restoring) patches that allows the reader to
//! fix the buffer that is being read without any locks. The inverse patch allows to restore previous
//! versions of the buffer. The writer thread saves the inverse patch before updating the buffer. The reader
//! thread reads the buffer (possibly inconsistent), and then uses the inverse patches to restore the buffer
//! at a specific version (LSN).
//!
//! Buffer update protocol:
//!
//! # Writing side
//! 1. generate new LSN, but not update it yet
//! 2. based on new data create inverse patches with new LSN and publish them atomically
//! 3. update the buffer itself with the new data
//! 4. update the LSN atomically
//!
//! # Reading side
//! 1. read the last LSN
//! 2. read the latest buffer, which may be inconsistent
//! 3. read the last delta reference
//! 4. updates all the delta patches that are greater than LSN from step (1).
//!
//! Happens-before relationships:
//! 1. patch.store -> buffer.store -> lsn.store (program order writer side)
//! 2. lsn.load -> buffer.load -> patch.load (program order reader side)
//! 3. lsn.store -> lsn.load (atomic store/load)
//!
//! The main consistency rule is patch.store -> patch.load. Which means that the reader will
//! always see the patches required to restore requested LSN, even if the buffer is inconsistent.
//! Form (1), (2) and (3) we can see that this rule is always satisfied.
//!
//! Thus, the reader will always see the patches required to restore requested LSN content and is able to
//! fix possibly inconsistent buffer.

use arc_swap::ArcSwap;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    cell::{Cell, UnsafeCell},
    mem,
    sync::{
        atomic::{fence, AtomicU64, Ordering},
        Arc,
    },
    thread,
};

// Size of a buffer for the test
const SIZE: usize = 1 << 16;

// Number of threads for the test (readers)
const THREADS: usize = 4;

const ITERATIONS: usize = 1_000_000;

struct Buffer {
    lsn: AtomicU64,
    data: Box<Cell<[u8]>>,
    snapshots: ArcSwap<Delta>,
}

unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}

impl Buffer {
    /// Write patches to the buffer
    ///
    /// Each patch is a tuple of offset and data to be written to the buffer
    fn write(self: &Arc<Self>, patches: Vec<(usize, Vec<u8>)>) {
        let d = self.data.as_ptr() as *const UnsafeCell<[u8]>;
        let b: &mut [u8] = unsafe { &mut *UnsafeCell::raw_get(d) };

        let lsn = self.lsn.load(Ordering::SeqCst);
        let new_lsn = lsn + 1;

        let mut inverse_patch = vec![];
        for (offset, patch) in &patches {
            let offset = *offset;
            let len = patch.len();
            let mut inverse = vec![0; len];
            inverse.copy_from_slice(&b[offset..offset + len]);
            inverse_patch.push((offset, inverse));
        }

        let new_snapshot = Arc::new(Delta {
            lsn,
            patches: inverse_patch,
            next: Some(self.snapshots.load_full()),
        });
        self.snapshots.swap(new_snapshot);

        fence(Ordering::Acquire);
        for (offset, patch) in patches {
            let len = patch.len();
            b[offset..offset + len].copy_from_slice(&patch);
        }
        fence(Ordering::Release);
        self.lsn.store(new_lsn, Ordering::SeqCst);
    }

    fn read(self: &Arc<Self>, buf: &mut [u8], offset: usize) -> u64 {
        let lsn = self.lsn.load(Ordering::SeqCst);

        let src: &[u8] = unsafe { mem::transmute(self.data.as_slice_of_cells()) };
        buf.copy_from_slice(&src[offset..offset + buf.len()]);

        let mut delta = Some(self.snapshots.load_full());
        while let Some(d) = delta {
            if d.lsn < lsn {
                break;
            }

            for (offset, patch) in &d.patches {
                let offset = *offset;
                let len = patch.len();
                buf[offset..offset + len].copy_from_slice(patch);
            }

            delta = {
                #[allow(clippy::assigning_clones)]
                d.next.clone()
            };
        }
        lsn
    }
}

struct Delta {
    lsn: u64,
    patches: Vec<(usize, Vec<u8>)>,
    next: Option<Arc<Self>>,
}

impl Drop for Delta {
    /// Custom drop logic is necessary here to prevent a stack overflow that could
    /// occur due to recursive drop calls on a long chain of `Arc` references to base snapshot.
    /// Each `Arc` decrement could potentially trigger the drop of another `Arc` in the chain,
    /// leading to deep recursion.
    ///
    /// By explicitly unwrapping and handling the inner `Arc` references, we ensure that the drop sequence
    /// is performed without any recursion
    fn drop(&mut self) {
        let mut next_base = self.next.take();
        while let Some(base) = next_base {
            next_base = Arc::try_unwrap(base)
                .map(|mut base| base.next.take())
                .unwrap_or(None);
        }
    }
}

fn main() {
    let mut state = [0u8; SIZE];
    let buf = Arc::new(Buffer {
        data: Box::new(Cell::new(state)),
        lsn: AtomicU64::new(0),
        snapshots: ArcSwap::from_pointee(Delta {
            lsn: 0,
            patches: vec![],
            next: None,
        }),
    });

    // Spawning threads that reads buffer and checks consistency
    let mut handles = vec![];
    for _ in 0..THREADS {
        let buf = Arc::clone(&buf);
        let handle = thread::spawn(move || {
            // By using number of iterations as max_lsn we make sure that
            // readers are able to progress and will see the most up to date
            // version eventually
            let max_lsn = ITERATIONS as u64;
            check_read_consistency(buf, max_lsn);
        });
        handles.push(handle);
    }

    // Updating buffer with random patches but keeping the sum of the buffer to be 0
    let mut rnd = SmallRng::from_entropy();
    for _ in 0..ITERATIONS {
        let mut patches = random_patches(&mut rnd, SIZE, 40);

        // mirroring changes on the local buffer to be able to calculate the compensator
        // to make the sum of the buffer to be 0
        for (offset, patch) in &patches {
            let len = patch.len();
            state[*offset..*offset + len].copy_from_slice(patch);
        }
        let sum = slice_wrapping_sum(&state);
        // The value that will make the sum of the buffer to be exactly 0
        let compensator = 255 - sum.wrapping_sub(state[0]).wrapping_sub(1);
        state[0] = compensator;
        patches.push((0, vec![compensator]));

        assert_eq!(slice_wrapping_sum(&state), 0);
        buf.write(patches);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Ok");
}

fn check_read_consistency(buf: Arc<Buffer>, max_lsn: u64) {
    let mut b = vec![0u8; SIZE];
    let mut read_lsn = 0;
    while read_lsn < max_lsn {
        let lsn = buf.read(&mut b, 0);
        assert!(
            lsn >= read_lsn,
            "Memory ordering violated. We've got stale LSN (latest LSN: {}, lsn: {})",
            read_lsn,
            lsn
        );
        read_lsn = lsn;
        let sum = slice_wrapping_sum(&b);
        assert_eq!(sum, 0, "Mismatch found. LSN: {}, data: {:?}", read_lsn, b);
    }
}

fn slice_wrapping_sum(initial_state: &[u8]) -> u8 {
    initial_state
        .iter()
        .copied()
        .reduce(|a, b| a.wrapping_add(b))
        .unwrap()
}

fn random_patches(rng: &mut SmallRng, size: usize, count: usize) -> Vec<(usize, Vec<u8>)> {
    let mut patches = vec![];
    for _ in 0..count {
        let len = rng.gen_range(1..100);
        let offset = rng.gen_range(0..=size - len);
        let mut patch = vec![0; len];
        rng.fill(&mut patch[..]);
        patches.push((offset, patch));
    }
    patches
}
