use crate::page::{PageNo, PAGE_SIZE};
use std::{collections::BTreeMap, io};

trait PageDriver {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<&[u8; PAGE_SIZE]>;
    fn read_page_mut(&mut self, page_no: PageNo) -> io::Result<&mut [u8; PAGE_SIZE]>;

    fn flush(&mut self) -> io::Result<()>;
}

pub struct MemoryDriver {
    pages: BTreeMap<PageNo, Box<[u8; PAGE_SIZE]>>,
}

impl PageDriver for MemoryDriver {
    fn read_page(&mut self, page_no: PageNo) -> io::Result<&[u8; PAGE_SIZE]> {
        Ok(&*self.read_page_mut(page_no)?)
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
