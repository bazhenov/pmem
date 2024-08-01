use pmem::{
    page::{PagePool, PAGE_SIZE},
    Memory,
};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rfs::{FileMeta, Filesystem};
use std::io::{Read, Seek, SeekFrom, Write};
use tango_bench::{
    benchmark_fn, tango_benchmarks, tango_main, Bencher, IntoBenchmarks, MeasurementSettings,
    Sampler,
};

const BLOCK: [u8; 4096] = [0xAA; 4096];
const FILE_SIZE: u64 = 4 * 1024 * 1024;

fn page_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("streaming_read_4m", streaming_read_4m),
        benchmark_fn("streaming_write_4m", streaming_write_4m),
        benchmark_fn("random_read_4k", random_read_4k),
        benchmark_fn("random_write_4k", random_write_4k),
        benchmark_fn("create_empty_file", create_empty_file),
        benchmark_fn("navigate_directories", navigate_directories),
    ]
}

fn streaming_write_4m(b: Bencher) -> Box<dyn Sampler> {
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

fn streaming_read_4m(b: Bencher) -> Box<dyn Sampler> {
    let mut fs = create_fs();
    let meta = create_test_file(&mut fs, FILE_SIZE);
    let mut buf = vec![0u8; BLOCK.len()];

    b.iter(move || {
        let mut file = fs.open_file(&meta).unwrap();
        while file.read(&mut buf).unwrap() > 0 {}
    })
}

fn random_write_4k(b: Bencher) -> Box<dyn Sampler> {
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

fn random_read_4k(b: Bencher) -> Box<dyn Sampler> {
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

fn create_empty_file(b: Bencher) -> Box<dyn Sampler> {
    let mut fs = create_fs();
    let root = fs.get_root().unwrap();
    let mut file_no = 0;
    b.iter(move || {
        file_no += 1;
        let file = format!("file{}", file_no);
        fs.create_file(&root, &file).unwrap()
    })
}

fn create_test_file(fs: &mut Filesystem, file_size: u64) -> FileMeta {
    let root = fs.get_root().unwrap();
    let meta = fs.create_file(&root, "file").unwrap();

    let mut file = fs.open_file(&meta).unwrap();

    for _ in 0..file_size / BLOCK.len() as u64 {
        file.write_all(&BLOCK).unwrap();
    }

    file.flush().unwrap();

    meta
}

fn navigate_directories(b: Bencher) -> Box<dyn Sampler> {
    let mut rnd = SmallRng::seed_from_u64(b.seed);
    let mut fs = create_fs();

    let tree_height = 5;
    let child_names = [
        "dir1", "dir2", "dir3", "dir4", "dir5", "dir6", "dir7", "dir8", "dir9", "dir10",
    ];

    let root = fs.get_root().unwrap();
    create_deep_tree(&mut fs, &root, tree_height, &child_names);

    b.iter(move || {
        let mut dir = fs.get_root().unwrap();
        for _ in 0..tree_height {
            let name = child_names.choose(&mut rnd).unwrap();
            dir = fs.lookup(&dir, name).unwrap();
        }
    })
}

fn create_deep_tree(fs: &mut Filesystem, parent: &FileMeta, level: u64, names: &[&str]) {
    if level == 0 {
        return;
    }
    for name in names {
        let dir = fs.create_dir(&parent, &name).unwrap();
        create_deep_tree(fs, &dir, level - 1, names);
    }
}

fn create_fs() -> Filesystem {
    let mem = Memory::new(PagePool::new(2usize.pow(32) / PAGE_SIZE)); // 4GiB
    Filesystem::allocate(mem)
}

tango_benchmarks!(page_benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: 1000,
    ..Default::default()
});
