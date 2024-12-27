use pmem::{
    vmem::{self, VTx},
    volume::{Addr, Transaction, TxRead, TxWrite, Volume},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{hint::black_box, ops::Range};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, ErasedSampler, IntoBenchmarks,
    MeasurementSettings,
};

const DB_SIZE: usize = 100 * 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("arbitrary_read_1k", arbitrary_read_1k),
        benchmark_fn("arbitrary_write_1k", arbitrary_write_1k),
        benchmark_fn("write_commit_1k", write_commit_1k),
        benchmark_fn("repeatable_read_tx_1k", repeatable_read_tx_1k),
        benchmark_fn("vmem_read_1k", vmem_read_1k),
    ]
}

fn arbitrary_read_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    const READ_LEN: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let vol = generate_volume(&mut rng);
    b.iter(move || {
        let tx = vol.start();

        let addr = random_addr(&mut rng, 0..DB_SIZE, READ_LEN);
        let _ = black_box(tx.read(addr as Addr, READ_LEN));
    })
}

fn arbitrary_write_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    const WRITE_LEN: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut patch = [0u8; WRITE_LEN];
    rng.fill(&mut patch[..]);

    let vol = Volume::with_capacity(DB_SIZE);

    b.iter(move || {
        const PATCHES_COUNT: usize = 10;
        let mut tx = vol.start();

        for _ in 0..PATCHES_COUNT {
            // Here we use 0..PATCH_SIZE * PATCHES_COUNT to ensure that the patches are competing
            // for the same memory regions.
            let addr = random_addr(&mut rng, 0..WRITE_LEN * PATCHES_COUNT, WRITE_LEN);
            tx.write(addr as Addr, patch.as_slice());
        }
    })
}

fn write_commit_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    const WRITE_LEN: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut patch = [0u8; WRITE_LEN];
    rng.fill(&mut patch[..]);
    let mut vol = Volume::with_capacity(DB_SIZE);

    b.iter(move || {
        let addr = random_addr(&mut rng, 0..DB_SIZE, WRITE_LEN);
        let mut tx = vol.start();
        tx.write(addr as Addr, patch.as_slice());
        vol.commit(tx)
    })
}

/// This benchmark measures the time it takes for a repeated read operation to read through the undo log.
fn repeatable_read_tx_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    const READ_LEN: usize = 1024;
    const TRANSACTIONS: usize = 1000;

    let mut vol = Volume::with_capacity(READ_LEN);
    // We must take snapshot before the transactions commit, otherwise the undo log of a snapshot will be empty.
    let s = vol.snapshot();
    for _ in 0..TRANSACTIONS {
        let mut tx = vol.start();
        tx.write(0, [1; READ_LEN]);
        vol.commit(tx).unwrap();
    }

    b.iter(move || {
        let _ = black_box(s.read(0, READ_LEN));
    })
}

fn vmem_read_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    const READ_LEN: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let (mut vol, txs) = generate_vmem::<2>(&mut rng);
    let tx = vmem::finish(txs).unwrap();
    vol.commit(tx).unwrap();
    let txs = vmem::open::<2, _>(vol.snapshot()).unwrap();

    b.iter(move || {
        let tx = &txs[0];
        let addr = random_addr(&mut rng, 0..DB_SIZE, READ_LEN);
        let _ = black_box(tx.read(addr as Addr, READ_LEN));
    })
}

fn random_addr(rng: &mut SmallRng, mut range: Range<usize>, len: usize) -> usize {
    range.end -= len;
    rng.gen_range(range)
}

fn generate_volume(rng: &mut SmallRng) -> Volume {
    const TRANSACTIONS: usize = 100;
    const PATCH_SIZE: usize = 1024;
    const PATCHES: usize = 100;

    let mut patch = [0u8; PATCH_SIZE];
    rng.fill(&mut patch[..]);

    let mut vol = Volume::with_capacity(DB_SIZE);
    for _ in 0..TRANSACTIONS {
        let mut tx = vol.start();
        write_repeatedly(rng, &mut tx, &patch, PATCHES);
        vol.commit(tx).unwrap();
    }
    vol
}

fn generate_vmem<const N: usize>(rng: &mut SmallRng) -> (Volume, [VTx<Transaction>; N]) {
    const PATCH_SIZE: usize = 1024;
    const PATCHES: usize = 100;

    let mut patch = [0u8; PATCH_SIZE];
    rng.fill(&mut patch[..]);

    let vol = Volume::with_capacity(DB_SIZE);
    let mut txs = vmem::init::<N, _>(vol.start()).unwrap();
    for tx in txs.as_mut() {
        write_repeatedly(rng, tx, &patch, PATCHES);
    }
    (vol, txs)
}

fn write_repeatedly(rng: &mut SmallRng, tx: &mut impl TxWrite, buf: &[u8], times: usize) {
    for _ in 0..times {
        let addr = random_addr(rng, 0..DB_SIZE, buf.len());
        tx.write(addr as Addr, buf);
    }
}

tango_benchmarks!(page_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
