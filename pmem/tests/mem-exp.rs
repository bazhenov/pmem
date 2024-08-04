use arc_swap::ArcSwap;
use rand::{rngs::SmallRng, Rng};
use std::{
    collections::HashSet,
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
};
use sync_unsafe_cell::SyncUnsafeCell;

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
#[derive(Clone)]
struct Buf {
    lsn: Arc<AtomicU64>,
    data: Arc<SyncUnsafeCell<[u8; SIZE]>>,
    snapshots: Arc<ArcSwap<Delta>>,
}

impl Buf {
    fn commit_patches(&mut self, patches: Vec<(usize, Vec<u8>)>) {
        let b: *mut [u8; SIZE] = self.data.get();

        let lsn = self.lsn.load(Ordering::SeqCst);
        let new_lsn = lsn + 1;

        let mut inverse_patch = vec![];
        for (offset, patch) in &patches {
            let offset = *offset;
            let len = patch.len();
            let mut inverse = vec![0; len];
            inverse.copy_from_slice(unsafe { &(*b)[offset..offset + len] });
            inverse_patch.push((offset, inverse));
        }

        let new_snapshot = Arc::new(Delta {
            lsn,
            patches: inverse_patch,
            next: Some(self.snapshots.load_full()),
        });
        self.snapshots.swap(new_snapshot);

        for (offset, patch) in patches {
            let len = patch.len();
            unsafe { (*b)[offset..offset + len].copy_from_slice(&patch) };
        }
        self.lsn.store(new_lsn, Ordering::SeqCst);
    }

    fn snapshot(&self) -> Snapshot {
        let data = self.data.clone();
        let lsn = self.lsn.load(Ordering::SeqCst);
        let snapshots = Arc::clone(&self.snapshots);
        Snapshot {
            lsn,
            data,
            snapshots,
        }
    }
}

struct Snapshot {
    lsn: u64,
    data: Arc<SyncUnsafeCell<[u8; SIZE]>>,
    snapshots: Arc<ArcSwap<Delta>>,
}

impl Snapshot {
    fn as_slice(&self) -> &[u8] {
        unsafe { &*self.data.get() }
    }
}

struct Delta {
    lsn: u64,
    patches: Vec<(usize, Vec<u8>)>,
    next: Option<Arc<Self>>,
}

#[test]
fn check_mem() {
    for _ in 0..10000 {
        do_run();
    }
}

fn do_run() {
    let mut buf = Buf {
        data: Arc::new([0x0; SIZE].into()),
        lsn: Arc::new(AtomicU64::new(0)),
        snapshots: Arc::new(ArcSwap::from_pointee(Delta {
            lsn: 0,
            patches: vec![],
            next: None,
        })),
    };

    let handle = {
        let buf_handle = buf.clone();
        thread::spawn(move || {
            let mut lsn = 0;
            while lsn < 255 {
                let mut b = vec![0u8; SIZE];
                let s = buf_handle.snapshot();
                lsn = s.lsn;
                b.copy_from_slice(s.as_slice());
                let mut delta = Some(s.snapshots.load_full());
                let last_snapshot_lsn = delta.as_ref().unwrap().lsn;
                // println!("Latest snapshot: {}", s.lsn);

                let mut snapshots_applied = 0;
                while let Some(d) = delta {
                    // println!("Applying LSN: {} (patches: {})", d.lsn, d.patches.len());
                    if d.lsn < s.lsn {
                        break;
                    }

                    for (offset, patch) in &d.patches {
                        let offset = *offset;
                        let len = patch.len();
                        b[offset..offset + len].copy_from_slice(&patch);
                    }
                    delta = d.next.clone();
                    snapshots_applied += 1;
                }

                let all_ok = b.iter().all(|x| *x == s.lsn as u8);
                if !all_ok {
                    panic!(
                        "LSN :{} (last snapshot: {}, applied: {}), data: {:?}",
                        s.lsn, last_snapshot_lsn, snapshots_applied, b,
                    );
                }
            }
        })
    };

    for i in 1..=255 {
        buf.commit_patches(vec![(0, vec![i; 10])]);
    }

    handle.join().unwrap();
}

fn random_segments(rng: &mut SmallRng, size: usize) -> Vec<Range<usize>> {
    use rand::seq::SliceRandom;

    let mut points = HashSet::new();
    let ranges_cnt = (size / 10).max(1).min(10);
    while points.len() < ranges_cnt - 1 {
        points.insert(rng.gen_range(1..size - 1));
    }
    points.insert(0);
    points.insert(size);

    let mut split_points = points.into_iter().collect::<Vec<_>>();
    split_points.sort();

    let mut ranges = Vec::new();
    for pair in split_points.windows(2) {
        let [from, to] = [pair[0], pair[1]];
        ranges.push(from..to);
    }
    ranges.shuffle(rng);

    ranges
}
