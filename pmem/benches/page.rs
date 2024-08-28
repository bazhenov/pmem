use pmem::page::{Addr, PagePool, TxRead, TxWrite, PAGE_SIZE};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{hint::black_box, ops::Range};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, MeasurementSettings,
    Sampler,
};

const DB_SIZE: usize = 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("arbitrary_read", arbitrary_read),
        benchmark_fn("arbitrary_write", arbitrary_write),
        benchmark_fn("write_commit", write_commit),
    ]
}

fn arbitrary_read(b: Bencher) -> Box<dyn Sampler> {
    let mut rng = SmallRng::seed_from_u64(b.seed);
    let mem = generate_mem(&mut rng);
    b.iter(move || {
        let (addr, len) = random_segment(&mut rng, 0..DB_SIZE);
        let tx = mem.start();
        let _ = black_box(tx.read(addr as Addr, len));
    })
}

fn arbitrary_write(b: Bencher) -> Box<dyn Sampler> {
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut buffer = [0u8; DB_SIZE];
    rng.fill(&mut buffer[..]);

    let mem = PagePool::new_in_memory((DB_SIZE / PAGE_SIZE + 1) as u32);
    let mut tx = mem.start();
    b.iter(move || {
        let (addr, len) = random_segment(&mut rng, 0..DB_SIZE);
        tx.write(addr as Addr, &buffer[..len]);
    })
}

fn write_commit(b: Bencher) -> Box<dyn Sampler> {
    let mut rng = SmallRng::seed_from_u64(b.seed);

    let mut buffer = [0u8; DB_SIZE];
    rng.fill(&mut buffer[..]);

    let mut mem = PagePool::new_in_memory((DB_SIZE / PAGE_SIZE + 1) as u32);

    b.iter(move || {
        let (addr, len) = random_segment(&mut rng, 0..DB_SIZE);
        let mut tx = mem.start();
        tx.write(addr as Addr, &buffer[..len]);
        mem.commit(tx).unwrap();
    })
}

fn random_segment(rng: &mut SmallRng, mut range: Range<usize>) -> (usize, usize) {
    let len = rng.gen_range(1..1024);
    range.end -= len;
    let addr = rng.gen_range(range);
    (addr, len)
}

fn generate_mem(rng: &mut SmallRng) -> PagePool {
    const TRANSACTIONS: usize = 100;
    const PATCHES: usize = 1000;

    let mut buffer = [0u8; DB_SIZE];
    rng.fill(&mut buffer[..]);

    let mut mem = PagePool::new_in_memory((DB_SIZE / PAGE_SIZE + 1) as u32);
    for _ in 0..TRANSACTIONS {
        let mut tx = mem.start();
        for _ in 0..PATCHES {
            let (addr, len) = random_segment(rng, 0..DB_SIZE);
            tx.write(addr as Addr, &buffer[..len]);
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
