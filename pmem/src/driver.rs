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
    fn write_pages(&self, pages: &[(PageNo, &[u8; PAGE_SIZE])], lsn: LSN) -> io::Result<()>;

    fn current_lsn(&self) -> LSN;
}

type PageAndLsn = (LSN, Box<[u8; PAGE_SIZE]>);

pub struct NoDriver {
    pages: Mutex<BTreeMap<PageNo, PageAndLsn>>,
    lsn: LSN,
}

impl Default for NoDriver {
    fn default() -> Self {
        Self {
            pages: Mutex::new(BTreeMap::new()),
            lsn: 0,
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

    fn write_pages(
        &self,
        pages_to_write: &[(PageNo, &[u8; PAGE_SIZE])],
        lsn: LSN,
    ) -> io::Result<()> {
        let mut pages = self.pages.lock().unwrap();
        for (page_no, data) in pages_to_write {
            pages.insert(*page_no, (lsn, Box::new(**data)));
        }

        Ok(())
    }

    fn current_lsn(&self) -> LSN {
        self.lsn
    }
}

pub struct FileDriver {
    file: Mutex<(File, LSN)>,
}

impl FileDriver {
    pub fn from_file(file: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(file)?;
        Self::new(file)
    }

    pub fn new(mut file: File) -> io::Result<Self> {
        let mut lsn_bytes = [0; mem::size_of::<LSN>()];
        let lsn = if stream_len(&mut file)? >= mem::size_of::<LSN>() as u64 {
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut lsn_bytes)?;
            LSN::from_le_bytes(lsn_bytes)
        } else {
            0
        };

        Ok(Self {
            file: Mutex::new((file, lsn)),
        })
    }

    pub fn into_inner(self) -> File {
        self.file.into_inner().unwrap().0
    }
}

impl PageDriver for FileDriver {
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        let page_offset = page_no as u64 * PAGE_SIZE as u64 + mem::size_of::<LSN>() as u64;
        // Start of the next page
        let expected_size = page_offset + PAGE_SIZE as u64;
        let mut lock = self.file.lock().unwrap();
        let file = &mut lock.0;
        if stream_len(file)? >= expected_size {
            file.seek(SeekFrom::Start(page_offset))?;
            file.read_exact(page)?;
            Ok(Some(lock.1))
        } else {
            Ok(None)
        }
    }

    fn write_pages(&self, pages: &[(PageNo, &[u8; PAGE_SIZE])], lsn: LSN) -> io::Result<()> {
        let mut lock = self.file.lock().unwrap();
        let file = &mut lock.0;
        for (page_no, page) in pages.iter().copied() {
            let page_offset = page_no as u64 * PAGE_SIZE as u64 + mem::size_of::<LSN>() as u64;
            file.seek(SeekFrom::Start(page_offset))?;
            file.write_all(page)?;
        }

        file.seek(SeekFrom::Start(0))?;
        let lsn_bytes = lsn.to_le_bytes();
        file.write_all(&lsn_bytes)?;
        file.flush()?;
        lock.1 = lsn;
        Ok(())
    }

    fn current_lsn(&self) -> LSN {
        self.file.lock().unwrap().1
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
    pub pages: Vec<crate::volume::Page>,
    pub lsn: LSN,
}

#[cfg(test)]
impl PageDriver for TestPageDriver {
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        if let Some(data) = self.pages.get(page_no as usize) {
            page.copy_from_slice(data.as_ref());
            Ok(Some(self.lsn))
        } else {
            Ok(None)
        }
    }

    fn write_pages(&self, _pages: &[(PageNo, &[u8; PAGE_SIZE])], _lsn: LSN) -> io::Result<()> {
        Ok(())
    }

    fn current_lsn(&self) -> LSN {
        self.lsn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[test]
    fn can_read_and_write_page() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("test.db");
        let driver = FileDriver::from_file(file)?;

        let pages = [[1; PAGE_SIZE], [2; PAGE_SIZE], [3; PAGE_SIZE]];
        let expected_lsn = 42;

        let pages_to_write = pages
            .iter()
            .enumerate()
            .map(|(page_no, page)| (page_no as PageNo, page))
            .collect::<Vec<_>>();
        driver.write_pages(pages_to_write.as_slice(), expected_lsn)?;

        let driver = FileDriver::new(driver.into_inner())?;

        let mut page_copy = [0; PAGE_SIZE];
        for (page_no, page) in pages.iter().enumerate() {
            let lsn = driver
                .read_page(page_no as PageNo, &mut page_copy)?
                .unwrap();
            assert_eq!(lsn, expected_lsn);
            assert_eq!(page, &page_copy);
        }

        Ok(())
    }
}
