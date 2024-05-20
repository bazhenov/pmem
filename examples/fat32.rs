use fatfs::{format_volume, FileSystem, FormatVolumeOptions, FsOptions};
use pmem::{
    page::{PagePool, Snapshot, PAGE_SIZE},
    Addr,
};
use std::io::{self, Read, Seek, SeekFrom, Write};

fn main() {
    let page_count = 100;
    let mut pool = PagePool::new(page_count);
    {
        // Formatting file system
        let mut layer = FsLayer::new(&pool, page_count);
        format_volume(&mut layer, FormatVolumeOptions::new()).unwrap();
        pool.commit(layer.snapshot);
    }

    let mut layer = FsLayer::new(&pool, page_count);
    let fs = FileSystem::new(&mut layer, FsOptions::new()).unwrap();
    let root_dir = fs.root_dir();
    let file_name = "hello.txt";
    let content = b"Hello World!";
    {
        // Writing file
        root_dir
            .create_file(file_name)
            .unwrap()
            .write_all(content)
            .unwrap();
    }
    {
        // Reading file
        let mut buf = vec![];
        root_dir
            .open_file(file_name)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(content.as_ref(), buf);
    }
}

struct FsLayer {
    snapshot: Snapshot,
    addr: u64,
    size: usize,
}

impl FsLayer {
    fn new(pool: &PagePool, page_count: usize) -> Self {
        let snapshot = pool.snapshot();
        Self {
            snapshot,
            addr: 0,
            size: page_count * PAGE_SIZE,
        }
    }
}

impl Seek for &mut FsLayer {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.addr = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(pos) => (self.size as i64 + pos) as u64,
            SeekFrom::Current(pos) => (self.addr as i64 + pos) as u64,
        };
        Ok(self.addr)
    }
}

impl Read for &mut FsLayer {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let result = self.snapshot.read(self.addr as Addr, buf.len()).unwrap();
        buf.copy_from_slice(result.as_ref());

        self.addr += buf.len() as u64;

        Ok(buf.len())
    }
}

impl Write for &mut FsLayer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.snapshot.write(self.addr as Addr, buf);
        self.addr += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
