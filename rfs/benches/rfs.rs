use pmem::{
    page::{PagePool, PAGE_SIZE},
    Memory, Storable,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rfs::Filesystem;
use std::{
    cell::RefCell,
    io::{Seek, SeekFrom, Write},
};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, MeasurementSettings,
    Sampler,
};

const BLOCK: [u8; 4096] = [0; 4096];

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("bench_4k_write", bench_4k_write),
        benchmark_fn("bench_random_4k_write", bench_random_4k_write),
    ]
}

fn bench_4k_write(b: Bencher) -> Box<dyn Sampler> {
    let fs = RefCell::new(create_fs());
    let root = fs.borrow().get_root().unwrap();
    let file = fs.borrow_mut().create_file(&root, "file").unwrap();

    b.iter(move || {
        let mut fs_borrow = fs.borrow_mut();
        let mut file = fs_borrow.open_file(&file).unwrap();
        file.write(&BLOCK).unwrap();
        file.flush().unwrap();
    })
}

fn bench_random_4k_write(b: Bencher) -> Box<dyn Sampler> {
    let fs = RefCell::new(create_fs());
    let root = fs.borrow().get_root().unwrap();
    let meta = fs.borrow_mut().create_file(&root, "file").unwrap();
    let file_size = 4 * 1024 * 1024; // 4MiB
    {
        let mut fs_borrow = fs.borrow_mut();
        let mut file = fs_borrow.open_file(&meta).unwrap();
        file.seek(SeekFrom::Start(file_size - 1)).unwrap();
        file.write(&[0]).unwrap();
        file.flush().unwrap();
    }

    let mut rand = SmallRng::seed_from_u64(b.seed);
    b.iter(move || {
        let pos = rand.gen_range(0u64..file_size - (BLOCK.len() as u64));
        let mut fs_borrow = fs.borrow_mut();
        let mut file = fs_borrow.open_file(&meta).unwrap();
        // for _ in 0..1024 {
        file.seek(SeekFrom::Start(pos)).unwrap();
        file.write(&BLOCK).unwrap();
        // }
        file.flush().unwrap();
    })
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
