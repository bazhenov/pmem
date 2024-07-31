use pmem::{
    page::{PagePool, PAGE_SIZE},
    Memory, Storable,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rfs::{FileMeta, Filesystem};
use std::io::{Seek, SeekFrom, Write};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, MeasurementSettings,
    Sampler,
};

const BLOCK: [u8; 4096] = [0xAA; 4096];
const FILE_SIZE: u64 = 4 * 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("bench_4k_write", bench_4k_write),
        benchmark_fn("bench_random_4k_write", bench_random_4k_write),
    ]
}

fn bench_4k_write(b: Bencher) -> Box<dyn Sampler> {
    let mut fs = create_fs();
    let root = fs.get_root().unwrap();
    let file = fs.create_file(&root, "file").unwrap();
    b.iter(move || {
        let mut file = fs.open_file(&file).unwrap();
        file.write_all(&BLOCK).unwrap();
        file.flush().unwrap();
    })
}

fn bench_random_4k_write(b: Bencher) -> Box<dyn Sampler> {
    let mut fs = create_fs();
    let mut rand = SmallRng::seed_from_u64(b.seed);
    let meta = create_test_file(&mut fs, FILE_SIZE);

    b.iter(move || {
        let pos = rand.gen_range(0u64..FILE_SIZE - (BLOCK.len() as u64));
        let mut file = fs.open_file(&meta).unwrap();

        file.seek(SeekFrom::Start(pos)).unwrap();
        file.write_all(&BLOCK).unwrap();

        file.flush().unwrap();
    })
}

fn create_test_file(fs: &mut Filesystem, file_size: u64) -> FileMeta {
    let root = fs.get_root().unwrap();
    let meta = fs.create_file(&root, "file").unwrap();

    let mut file = fs.open_file(&meta).unwrap();

    file.seek(SeekFrom::Start(file_size - 1)).unwrap();
    for _ in 0..file_size / BLOCK.len() as u64 {
        file.write_all(&BLOCK).unwrap();
    }

    file.flush().unwrap();

    meta
}

fn create_fs() -> Filesystem {
    let mem = Memory::new(PagePool::new(2usize.pow(32) / PAGE_SIZE)); // 4GiB
    Filesystem::allocate(mem.start())
}

tango_benchmarks!(page_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
