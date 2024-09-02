use crate::volume::{PageNo, PAGE_SIZE};
use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub trait PageDriver: Send {
    fn read_page(&mut self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<()>;
    fn write_page(&mut self, page_no: PageNo, page: &[u8; PAGE_SIZE]) -> io::Result<()>;

    fn flush(&mut self) -> io::Result<()>;
}

pub struct NoDriver;

impl PageDriver for NoDriver {
    fn read_page(&mut self, _page_no: PageNo, _page: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
        Ok(())
    }

    fn write_page(&mut self, _page_no: PageNo, _page: &[u8; PAGE_SIZE]) -> io::Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct FileDriver {
    file: File,
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
        Self { file }
    }

    pub fn into_inner(self) -> File {
        self.file
    }
}

impl PageDriver for FileDriver {
    fn read_page(&mut self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
        let expected_size = (page_no as u64 + 1) * (PAGE_SIZE as u64);
        if stream_len(&mut self.file)? >= expected_size {
            let page_offset = page_no as u64 * PAGE_SIZE as u64;
            self.file.seek(SeekFrom::Start(page_offset))?;
            self.file.read_exact(page)?;
        }
        Ok(())
    }

    fn write_page(&mut self, page_no: PageNo, page: &[u8; PAGE_SIZE]) -> io::Result<()> {
        let page_offset = page_no as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(page_offset))?;
        self.file.write_all(page)?;

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn stream_len<T: Seek>(file: &mut T) -> io::Result<u64> {
    let current_pos = file.stream_position()?;
    let end_pos = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(current_pos))?;
    Ok(end_pos)
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
        let page = [42; PAGE_SIZE];
        let mut page_copy = [0; PAGE_SIZE];
        let mut driver = FileDriver::from_file(file)?;
        let page_no = 0;
        driver.write_page(page_no, &page)?;
        driver.flush()?;

        let mut driver = FileDriver::new(driver.into_inner());
        let page_no = 0;
        driver.read_page(page_no, &mut page_copy)?;
        assert_eq!(page, page_copy);
        Ok(())
    }
}
