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
const SIZE: usize = 10;

/// Buffer update protocol:
///
/// # Writing side
/// 1. generate new LSN, but not update it yet
/// 2. create new inverse patch with new LSN and push it to the list
/// 3. update the buffer itself with the new data
/// 4. update the lsn
///
/// # Reading side
/// 1. read the last lsn and save a copy to snapshot
/// 2. read the latest buffer, which may be inconsistent
/// 3. read the last delta reference
/// 4. updates all the delta patches that are greater than LSN from step (1).
///
/// Happens-before relationships:
/// 1. patch.store -> buffer.store -> lsn.store (program order writer side)
/// 2. lsn.load -> buffer.load -> patch.load (program order reader side)
/// 3. lsn.store -> lsn.load (atomic store/load)
///
/// The main consistency rule is patch.store -> patch.load. Which means that the reader will
/// always see the patches required to restore requested LSN, even if the buffer is inconsistent.
/// Form (1), (2) and (3) we can see that this rule is always satisfied.
///
/// Thus, the reader will always see the patches required to restore requested LSN.
struct Buf {
    lsn: AtomicU64,
    data: Box<Cell<[u8]>>,
    snapshots: ArcSwap<Delta>,
}

unsafe impl Sync for Buf {}
unsafe impl Send for Buf {}

impl Buf {
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
                buf[offset..offset + len].copy_from_slice(&patch);
            }
            delta = d.next.clone();
        }
        lsn
    }
}

struct Delta {
    lsn: u64,
    patches: Vec<(usize, Vec<u8>)>,
    next: Option<Arc<Self>>,
}

#[test]
fn check_mem() {
    for _ in 0..1000 {
        run();
    }
}

fn run() {
    let mut state = [0u8; SIZE];
    let buf = Arc::new(Buf {
        data: Box::new(Cell::new(state.clone())),
        lsn: AtomicU64::new(0),
        snapshots: ArcSwap::from_pointee(Delta {
            lsn: 0,
            patches: vec![],
            next: None,
        }),
    });

    let iterations = 1000;
    // Spawning threads that reads buffer and checks consistency
    let mut handles = vec![];
    for _ in 0..4 {
        let buf = buf.clone();
        let handle = thread::spawn(move || {
            check_read_consistency(buf, iterations);
        });
        handles.push(handle);
    }

    // Updating buffer with random patches but keeping the sum of the buffer to be 0
    let mut rnd = SmallRng::from_entropy();
    for _ in 0..iterations {
        let mut patches = random_patches(&mut rnd, SIZE, 4);

        // mirroring changes on the local buffer to be able to calculate the compensator
        // to make the sum of the buffer to be 0
        for (offset, patch) in &patches {
            let len = patch.len();
            state[*offset..*offset + len].copy_from_slice(&patch);
        }
        let sum = slice_wrapping_sum(&state);
        // The value that will make the sum of the buffer to be exactly 0
        let compensator = 255 - sum.wrapping_sub(state[0]).wrapping_sub(1);
        state[0] = compensator;
        patches.push((0, vec![compensator]));

        debug_assert_eq!(slice_wrapping_sum(&state), 0);
        buf.write(patches);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn check_read_consistency(buf: Arc<Buf>, max_lsn: u64) {
    let mut b = vec![0u8; SIZE];
    let mut read_lsn = 0;
    while read_lsn < max_lsn {
        read_lsn = buf.read(&mut b, 0);
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
        let len = rng.gen_range(1..=size);
        let offset = rng.gen_range(0..=size - len);
        let mut patch = vec![0; len];
        rng.fill(&mut patch[..]);
        patches.push((offset, patch));
    }
    patches
}
