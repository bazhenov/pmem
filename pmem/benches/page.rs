use pmem::volume::{Addr, TxRead, TxWrite, Volume};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{hint::black_box, ops::Range};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, MeasurementSettings,
    Sampler,
};

const DB_SIZE: usize = 100 * 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("arbitrary_read_1k", arbitrary_read_1k),
        benchmark_fn("arbitrary_write_1k", arbitrary_write_1k),
        benchmark_fn("write_commit_1k", write_commit_1k),
        benchmark_fn("repeatable_read_tx_1k", repeatable_read_tx_1k),
    ]
}

fn arbitrary_read_1k(b: Bencher) -> Box<dyn Sampler> {
    const PATCH_SIZE: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mem = generate_mem(&mut rng);
    b.iter(move || {
        let tx = mem.start();

        let addr = random_addr::<PATCH_SIZE>(&mut rng, 0..DB_SIZE);
        let _ = black_box(tx.read(addr as Addr, PATCH_SIZE));
    })
}

fn arbitrary_write_1k(b: Bencher) -> Box<dyn Sampler> {
    const PATCH_SIZE: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut patch = vec![0u8; PATCH_SIZE];
    rng.fill(&mut patch[..]);

    let mem = Volume::with_capacity(DB_SIZE);

    b.iter(move || {
        const PATCHES_COUNT: usize = 10;
        let mut tx = mem.start();

        for _ in 0..PATCHES_COUNT {
            // Here we use 0..PATCH_SIZE * PATCHES_COUNT to ensure that the patches are competing
            // for the same memory regions.
            let addr = random_addr::<PATCH_SIZE>(&mut rng, 0..PATCH_SIZE * PATCHES_COUNT);
            tx.write(addr as Addr, patch.as_slice());
        }
    })
}

fn write_commit_1k(b: Bencher) -> Box<dyn Sampler> {
    const PATCH_SIZE: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut patch = vec![0u8; PATCH_SIZE];
    rng.fill(&mut patch[..]);

    let mut mem = Volume::with_capacity(DB_SIZE);

    b.iter(move || {
        let addr = random_addr::<PATCH_SIZE>(&mut rng, 0..DB_SIZE);
        let mut tx = mem.start();
        tx.write(addr as Addr, patch.as_slice());
        mem.commit(tx).unwrap();
    })
}

/// This benchmark measures the time it takes for a repeated read operation to read through the undo log.
fn repeatable_read_tx_1k(b: Bencher) -> Box<dyn Sampler> {
    const PATCH_SIZE: usize = 1024;
    const TRANSACTIONS: usize = 1000;

    let mut mem = Volume::with_capacity(PATCH_SIZE);
    // We must take snapshot before the transactions commit, otherwise the undo log of a snapshot will be empty.
    let s = mem.snapshot();
    for _ in 0..TRANSACTIONS {
        let mut tx = mem.start();
        tx.write(0, [1; PATCH_SIZE]);
        mem.commit(tx).unwrap();
    }

    b.iter(move || {
        let _ = black_box(s.read(0, PATCH_SIZE));
    })
}

fn random_addr<const N: usize>(rng: &mut SmallRng, mut range: Range<usize>) -> usize {
    range.end -= N;
    rng.gen_range(range)
}

fn generate_mem(rng: &mut SmallRng) -> Volume {
    const TRANSACTIONS: usize = 100;
    const PATCHES: usize = 100;
    const PATCH_SIZE: usize = 1024;

    let mut patch = vec![0u8; PATCH_SIZE];
    rng.fill(&mut patch[..]);

    let mut mem = Volume::with_capacity(DB_SIZE);
    for _ in 0..TRANSACTIONS {
        let mut tx = mem.start();
        for _ in 0..PATCHES {
            let addr = random_addr::<PATCH_SIZE>(rng, 0..DB_SIZE);
            tx.write(addr as Addr, patch.as_slice());
        }
        mem.commit(tx).unwrap();
    }
    mem
}

tango_benchmarks!(page_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
