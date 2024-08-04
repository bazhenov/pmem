use arc_swap::ArcSwap;
use rand::{rngs::SmallRng, Rng};
use std::{
    collections::HashSet,
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use sync_unsafe_cell::SyncUnsafeCell;

const SIZE: usize = 10;

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
        // let patches = vec![(0, vec![new_lsn as u8; SIZE])];

        // println!("COMMIT LSN: {}, content: {:?}", new_lsn, &patches[0]);

        let mut inverse_patches = vec![];
        for (offset, patch) in &patches {
            let offset = *offset;
            let len = patch.len();
            let mut inverse = vec![0; len];
            inverse.copy_from_slice(unsafe { &(*b)[offset..offset + len] });
            inverse_patches.push((offset, inverse));
        }

        let new_snapshot = Arc::new(Delta {
            lsn,
            patches: inverse_patches,
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
    for _ in 0..1000 {
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
            std::thread::spawn(move || {
                for _ in 0..10000 {
                    let mut b = vec![0u8; SIZE];
                    let s = buf_handle.snapshot();
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

        for i in 1..255 {
            buf.commit_patches(vec![(0, vec![i; 10])]);
        }

        handle.join().unwrap();
    }
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
