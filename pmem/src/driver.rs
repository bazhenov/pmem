use crate::page::{PageNo, PAGE_SIZE};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub trait PageDriver: Send {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<Option<&[u8; PAGE_SIZE]>>;
    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]>;

    fn flush(&mut self) -> io::Result<()>;
}

#[derive(Default)]
pub struct MemoryDriver {
    pages: BTreeMap<PageNo, Box<[u8; PAGE_SIZE]>>,
}

impl PageDriver for MemoryDriver {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<Option<&[u8; PAGE_SIZE]>> {
        Ok(self.pages.get(&page_no).map(|p| p.as_ref()))
    }

    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]> {
        Ok(self
            .pages
            .entry(page_no)
            .or_insert_with(|| Box::new([0; PAGE_SIZE])))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct FileDriver {
    pages: BTreeMap<PageNo, Box<[u8; PAGE_SIZE]>>,
    file: std::fs::File,
}

impl FileDriver {
    pub fn new(file: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(file)?;
        let pages = BTreeMap::new();
        Ok(Self { file, pages })
    }

    fn ensure_loaded(&mut self, page_no: PageNo) -> io::Result<()> {
        if self.pages.contains_key(&page_no) {
            return Ok(());
        }
        let page_offset = page_no as u64 * PAGE_SIZE as u64;
        let expected_size = (page_no as u64 + 1) * (PAGE_SIZE as u64);
        if self.file.metadata()?.len() < expected_size {
            self.file.set_len(expected_size)?;
        }
        self.file.seek(SeekFrom::Start(page_offset))?;
        let mut page = Box::new([0; PAGE_SIZE]);
        self.file.read_exact(page.as_mut())?;
        self.pages.insert(page_no, page);
        Ok(())
    }
}

impl PageDriver for FileDriver {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<Option<&[u8; PAGE_SIZE]>> {
        self.ensure_loaded(page_no)?;
        Ok(self.pages.get(&page_no).map(|p| p.as_ref()))
    }

    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]> {
        self.ensure_loaded(page_no)?;
        Ok(self
            .pages
            .entry(page_no)
            .or_insert_with(|| Box::new([0; PAGE_SIZE])))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_read_and_write_page() -> io::Result<()> {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_file = temp_dir.path().join("test.db");
        let mut driver = FileDriver::new(temp_file)?;
        let page_no = 0;
        let page = driver.read_page_mut(page_no)?;
        page[0] = 42;
        driver.flush()?;
        let page = driver.read_page(page_no)?.unwrap();
        assert_eq!(page[0], 42);
        Ok(())
    }
}
