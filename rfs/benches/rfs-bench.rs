use pmem::volume::{TxWrite, Volume};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rfs::{FileMeta, Filesystem};
use std::io::{Read, Seek, SeekFrom, Write};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, ErasedSampler, IntoBenchmarks,
    MeasurementSettings,
};

const BLOCK: [u8; 4096] = [0xAA; 4096];
const FILE_SIZE: u64 = 4 * 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("streaming_read_4m", streaming_read_4m),
        benchmark_fn("streaming_write_4m", streaming_write_4m),
        benchmark_fn("random_read_4k", random_read_4k),
        benchmark_fn("random_write_4k", random_write_4k),
        benchmark_fn("create_100_empty_files", create_100_empty_files),
        benchmark_fn("navigate_directories", navigate_directories),
    ]
}

fn streaming_write_4m(b: Bencher) -> Box<dyn ErasedSampler> {
    let mut fs = create_fs();
    let root = fs.get_root().unwrap();
    let file = fs.create_file(&root, "file").unwrap();
    b.iter(move || {
        let mut file = fs.open_file(&file).unwrap();

        for _ in 0..FILE_SIZE / BLOCK.len() as u64 {
            file.write_all(&BLOCK).unwrap();
        }

        file.flush().unwrap();
    })
}

fn streaming_read_4m(b: Bencher) -> Box<dyn ErasedSampler> {
    let mut fs = create_fs();
    let meta = create_test_file(&mut fs, FILE_SIZE);
    let mut buf = vec![0u8; BLOCK.len()];

    b.iter(move || {
        let mut file = fs.open_file(&meta).unwrap();
        while file.read(&mut buf).unwrap() > 0 {}
    })
}

fn random_write_4k(b: Bencher) -> Box<dyn ErasedSampler> {
    let mut fs = create_fs();
    let mut rand = SmallRng::seed_from_u64(b.seed);
    let meta = create_test_file(&mut fs, FILE_SIZE);

    b.iter(move || {
        let mut file = fs.open_file(&meta).unwrap();

        let pos = rand.gen_range(0u64..FILE_SIZE - (BLOCK.len() as u64));
        file.seek(SeekFrom::Start(pos)).unwrap();
        file.write_all(&BLOCK).unwrap();

        file.flush().unwrap();
    })
}

fn random_read_4k(b: Bencher) -> Box<dyn ErasedSampler> {
    let mut fs = create_fs();
    let mut rand = SmallRng::seed_from_u64(b.seed);
    let meta = create_test_file(&mut fs, FILE_SIZE);
    let mut buf = vec![0u8; BLOCK.len()];

    b.iter(move || {
        let mut file = fs.open_file(&meta).unwrap();

        let pos = rand.gen_range(0u64..FILE_SIZE - (BLOCK.len() as u64));
        file.seek(SeekFrom::Start(pos)).unwrap();
        file.read_exact(&mut buf).unwrap();

        file.flush().unwrap();
    })
}

fn create_100_empty_files(b: Bencher) -> Box<dyn ErasedSampler> {
    b.iter(move || {
        let mut fs = create_fs();
        let root = fs.get_root().unwrap();
        for file_no in 0..100 {
            let file = format!("file{}", file_no);
            fs.create_file(&root, &file).unwrap();
        }
    })
}

fn create_test_file(fs: &mut Filesystem<impl TxWrite>, file_size: u64) -> FileMeta {
    let root = fs.get_root().unwrap();
    let meta = fs.create_file(&root, "file").unwrap();

    let mut file = fs.open_file(&meta).unwrap();

    for _ in 0..file_size / BLOCK.len() as u64 {
        file.write_all(&BLOCK).unwrap();
    }

    file.flush().unwrap();

    meta
}

fn navigate_directories(b: Bencher) -> Box<dyn ErasedSampler> {
    // TODO: How to create this setup only once?
    let mut rnd = SmallRng::seed_from_u64(b.seed);
    let mut fs = create_fs();

    let root = fs.get_root().unwrap();
    const DIRECTORIES: usize = 100;
    for idx in 0..DIRECTORIES {
        fs.create_dir(&root, format!("dir{}", idx)).unwrap();
    }

    b.iter(move || {
        let name = format!("dir{}", rnd.gen_range(0..DIRECTORIES));
        fs.lookup(&root, name).unwrap()
    })
}

fn create_fs() -> Filesystem<impl TxWrite> {
    let volume = Volume::with_capacity(2usize.pow(32)); // 4GiB
    Filesystem::allocate(volume.start())
}

tango_benchmarks!(page_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
