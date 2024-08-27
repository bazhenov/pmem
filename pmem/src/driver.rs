use crate::page::{PageNo, PAGE_SIZE};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub trait PageDriver: Send {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<Option<&[u8; PAGE_SIZE]>>;
    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]>;

    fn flush(&mut self) -> io::Result<()>;
}

struct Page {
    data: Box<[u8; PAGE_SIZE]>,
    dirty: bool,
}

impl Page {
    fn new() -> Self {
        Self {
            data: Box::new([0; PAGE_SIZE]),
            dirty: false,
        }
    }
}

pub struct FileDriver<T> {
    pages: BTreeMap<PageNo, Page>,
    file: T,
}

impl FileDriver<File> {
    pub fn from_file(file: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(file)?;
        Ok(Self::new(file))
    }
}

impl FileDriver<Cursor<Vec<u8>>> {
    pub fn in_memory() -> Self {
        Self::new(Cursor::new(vec![]))
    }
}

impl<T> FileDriver<T> {
    pub fn into_inner(self) -> T {
        self.file
    }
}

impl<T: Read + Write + Seek> FileDriver<T> {
    pub fn new(file: T) -> Self {
        Self {
            pages: BTreeMap::new(),
            file,
        }
    }

    fn ensure_loaded(&mut self, page_no: PageNo) -> io::Result<()> {
        let page = self.pages.entry(page_no).or_insert_with(Page::new);
        let expected_size = (page_no as u64 + 1) * (PAGE_SIZE as u64);
        if stream_len(&mut self.file)? >= expected_size {
            let page_offset = page_no as u64 * PAGE_SIZE as u64;
            self.file.seek(SeekFrom::Start(page_offset))?;
            self.file.read_exact(page.data.as_mut())?;
        }
        Ok(())
    }
}

impl<T: Read + Write + Seek + Send> PageDriver for FileDriver<T> {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<Option<&[u8; PAGE_SIZE]>> {
        self.ensure_loaded(page_no)?;
        Ok(self.pages.get(&page_no).map(|p| p.data.as_ref()))
    }

    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]> {
        self.ensure_loaded(page_no)?;
        let page = self.pages.entry(page_no).or_insert_with(Page::new);
        page.dirty = true;
        Ok(page.data.as_mut())
    }

    fn flush(&mut self) -> io::Result<()> {
        for (page_no, page) in self.pages.iter() {
            if page.dirty {
                write_page(&mut self.file, *page_no, page)?;
            }
        }
        self.file.flush()
    }
}

fn stream_len<T: Seek>(file: &mut T) -> io::Result<u64> {
    let current_pos = file.stream_position()?;
    let end_pos = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(current_pos))?;
    Ok(end_pos)
}

fn write_page<T: Write + Seek>(file: &mut T, page_no: PageNo, page: &Page) -> io::Result<()> {
    let page_offset = page_no as u64 * PAGE_SIZE as u64;
    file.seek(SeekFrom::Start(page_offset))?;
    file.write_all(page.data.as_ref())?;

    Ok(())
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;

    #[test]
    fn can_read_and_write_page() -> io::Result<()> {
        let mut driver = FileDriver::in_memory();
        let page_no = 0;
        let page = driver.read_page_mut(page_no)?;
        page[0] = 42;
        driver.flush()?;

        let mut driver = FileDriver::new(driver.into_inner());
        let page_no = 0;
        let page = driver.read_page(page_no)?.unwrap();
        assert_eq!(page[0], 42);
        Ok(())
    }
}
