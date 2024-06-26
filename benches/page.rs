use pmem::{
    page::{PagePool, PAGE_SIZE},
    Addr,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::hint::black_box;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, Sampler};

const DB_SIZE: usize = 10 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("arbitrary_write", bench_arbitrary_write)]
}

fn bench_arbitrary_write(b: Bencher) -> Box<dyn Sampler> {
    let mut rng = SmallRng::seed_from_u64(b.seed);
    let mem = generate_mem(&mut rng);
    b.iter(move || {
        let len = rng.gen_range(1..1024);
        let addr = rng.gen_range(0..DB_SIZE - len);

        let snapshot = mem.snapshot();
        let _ = black_box(snapshot.read(addr as Addr, len)).unwrap();
    })
}

fn generate_mem(rng: &mut SmallRng) -> PagePool {
    const SNAPSHOTS: usize = 100;
    const PATCHES: usize = 10;

    let mut buffer = [0u8; DB_SIZE];
    rng.fill(&mut buffer[..]);

    let mut mem = PagePool::new(DB_SIZE / PAGE_SIZE + 1);
    for _ in 0..SNAPSHOTS {
        let mut snapshot = mem.snapshot();
        for _ in 0..PATCHES {
            let len = rng.gen_range(1..DB_SIZE);
            let addr = rng.gen_range(0..DB_SIZE - len);
            snapshot.write(addr as Addr, &buffer[..len]);
        }
        mem.commit(snapshot);
    }
    mem
}

tango_benchmarks!(page_benchmarks());
tango_main!();
