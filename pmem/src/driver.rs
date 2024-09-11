use crate::volume::{PageNo, LSN, PAGE_SIZE};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    mem,
    path::Path,
    sync::Mutex,
};

pub trait PageDriver: Send + Sync {
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>>;
    fn write_page(&self, page_no: PageNo, page: &[u8; PAGE_SIZE], lsn: LSN) -> io::Result<()>;

    fn flush(&self) -> io::Result<()>;
}

type PageAndLsn = (LSN, Box<[u8; PAGE_SIZE]>);

pub struct NoDriver {
    pages: Mutex<BTreeMap<PageNo, PageAndLsn>>,
}

impl Default for NoDriver {
    fn default() -> Self {
        Self {
            pages: Mutex::new(BTreeMap::new()),
        }
    }
}

impl PageDriver for NoDriver {
    fn read_page(&self, page_no: PageNo, buf: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        let pages = self.pages.lock().unwrap();
        if let Some((lsn, page)) = pages.get(&page_no) {
            buf.copy_from_slice(page.as_slice());
            Ok(Some(*lsn))
        } else {
            Ok(None)
        }
    }

    fn write_page(&self, page_no: PageNo, page: &[u8; PAGE_SIZE], lsn: LSN) -> io::Result<()> {
        let mut pages = self.pages.lock().unwrap();
        pages.insert(page_no, (lsn, Box::new(*page)));
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

pub struct FileDriver {
    file: Mutex<File>,
}

impl FileDriver {
    pub fn from_file(file: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(file)?;
        Ok(Self::new(file))
    }

    pub fn new(file: File) -> Self {
        Self {
            file: Mutex::new(file),
        }
    }

    pub fn into_inner(self) -> File {
        self.file.into_inner().unwrap()
    }
}

const BLOCK_SIZE: u64 = PAGE_SIZE as u64 + mem::size_of::<LSN>() as u64;

impl PageDriver for FileDriver {
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        let expected_size = (page_no as u64 + 1) * BLOCK_SIZE;
        let mut lsn_bytes = [0u8; mem::size_of::<LSN>()];
        let mut file = self.file.lock().unwrap();
        if stream_len(&mut *file)? >= expected_size {
            let page_offset = page_no as u64 * BLOCK_SIZE;
            file.seek(SeekFrom::Start(page_offset))?;
            file.read_exact(&mut lsn_bytes)?;
            file.read_exact(page)?;
            Ok(Some(LSN::from_le_bytes(lsn_bytes)))
        } else {
            Ok(None)
        }
    }

    fn write_page(&self, page_no: PageNo, page: &[u8; PAGE_SIZE], lsn: LSN) -> io::Result<()> {
        let page_offset = page_no as u64 * BLOCK_SIZE;
        let mut file = self.file.lock().unwrap();

        file.seek(SeekFrom::Start(page_offset))?;

        let lsn_bytes = lsn.to_le_bytes();
        file.write_all(&lsn_bytes)?;
        file.write_all(page)?;

        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        self.file.lock().unwrap().flush()
    }
}

fn stream_len<T: Seek>(file: &mut T) -> io::Result<u64> {
    let current_pos = file.stream_position()?;
    let end_pos = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(current_pos))?;
    Ok(end_pos)
}

#[cfg(test)]
pub struct TestPageDriver {
    pub pages: Vec<(LSN, [u8; PAGE_SIZE])>,
}

#[cfg(test)]
impl PageDriver for TestPageDriver {
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        if let Some((lsn, data)) = self.pages.get(page_no as usize) {
            page.copy_from_slice(data);
            Ok(Some(*lsn))
        } else {
            Ok(None)
        }
    }

    fn write_page(&self, _page_no: PageNo, _page: &[u8; PAGE_SIZE], _lsn: LSN) -> io::Result<()> {
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use tempfile;

    #[test]
    fn can_read_and_write_page() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("test.db");

        let pages = [
            (10 as LSN, [1; PAGE_SIZE]), // LSN + Page Content
            (20, [2; PAGE_SIZE]),
            (30, [3; PAGE_SIZE]),
        ];

        let driver = FileDriver::from_file(file)?;

        for (page_no, (lsn, page)) in pages.iter().enumerate() {
            driver.write_page(page_no as PageNo, page, *lsn)?;
        }

        driver.flush()?;

        let driver = FileDriver::new(driver.into_inner());

        let mut page_copy = [0; PAGE_SIZE];
        for (page_no, (expected_lsn, page)) in pages.iter().enumerate() {
            let lsn = driver
                .read_page(page_no as PageNo, &mut page_copy)?
                .unwrap();
            assert_eq!(lsn, *expected_lsn);
            assert_eq!(page, &page_copy);
        }

        Ok(())
    }
}
