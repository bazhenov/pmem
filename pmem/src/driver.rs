use crate::page::{PageNo, PAGE_SIZE};
use std::{collections::BTreeMap, io};

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
