use pmem::{
    volume::{Volume, PAGE_SIZE},
    Memory,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, ErasedSampler, IntoBenchmarks,
    MeasurementSettings,
};
const VOLUME_SIZE: usize = 100 * 1024 * 1024;

fn memory_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("alloc_dealloc_fixed_1k", alloc_dealloc_fixed_1k),
        benchmark_fn("alloc_dealloc_random", alloc_dealloc_random),
    ]
}

fn alloc_dealloc_fixed_1k(b: Bencher) -> Box<dyn ErasedSampler> {
    let volume = Volume::with_capacity(VOLUME_SIZE);
    let mut mem = Memory::init(volume.start());

    b.iter(move || {
        let ptr = mem.alloc::<[u8; 1024]>().expect("Unable allocate memory");
        mem.reclaim(ptr).expect("Unable to reclaim memory");
    })
}

fn alloc_dealloc_random(b: Bencher) -> Box<dyn ErasedSampler> {
    let volume = Volume::with_capacity(PAGE_SIZE);
    let mut mem = Memory::init(volume.start());
    let mut rng = SmallRng::seed_from_u64(b.seed);

    b.iter(move || {
        // see [`memory_can_reuse_memory()`] test for this calculation
        let size = rng.gen_range(1..=357);

        let addr = mem.alloc_addr(size).expect("Unable allocate memory");
        mem.reclaim_addr(addr).expect("Unable to reclaim memory");
    })
}

tango_benchmarks!(memory_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
