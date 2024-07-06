use crate::{
    memory::{parse_optional_ptr, write_optional_ptr, SlicePtr},
    Handle, Ptr, Storable, Transaction,
};
use binrw::{binrw, BinRead, BinWrite};
use std::{
    ffi::OsStr,
    fmt,
    path::{Component, PathBuf},
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum Error {
    #[error("not found")]
    NotFound,

    #[error("not supported")]
    NotSupported,

    #[error("path must be absolute")]
    PathMustBeAbsolute,

    #[error("already exists")]
    AlreadyExists,
}

struct Filesystem {
    root: Ptr<ListNode<FsEntry>>,
    tx: Transaction,
}

impl Storable for Filesystem {
    type Seed = ListNode<FsEntry>;

    fn open(tx: Transaction, root: Ptr<Self::Seed>) -> Self {
        Self { root, tx }
    }

    fn allocate(mut tx: Transaction) -> Self {
        let name = tx.write_slice("/".as_bytes());
        let list_node = tx.write(ListNode {
            value: FsEntry {
                name: Str(name),
                node_type: NodeType::Directory,
                children: None,
            },
            next: None,
        });
        let root = list_node.ptr();
        Self { root, tx }
    }

    fn finish(self) -> Transaction {
        todo!()
    }
}

impl Filesystem {
    fn get_root(&self) -> FsEntry {
        let root = self.tx.lookup(self.root).into_inner();
        root.value
    }

    pub fn lookup(&self, name: impl Into<PathBuf>) -> Result<FsEntry> {
        let path = name.into();
        let mut components = path.components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut cur_node: FsEntry = self.tx.lookup(self.root).into_inner().value;
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            let child = cur_node.children.and_then(|c| self.find_child(c, name));
            let Some(child) = child else {
                return Err(Error::NotFound);
            };
            cur_node = child;
        }

        Ok(cur_node)
    }

    pub fn create_dir(&mut self, name: impl Into<PathBuf>) -> Result<FsEntry> {
        let path = name.into();

        let mut components = path.components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut node: Handle<ListNode<FsEntry>> = self.tx.lookup(self.root);
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            if node.value.children.is_none() {
                let new_node = self.write_fsnode(name.to_str().unwrap());
                node.value.children = Some(new_node.ptr());
                self.tx.update(&node);
                node = new_node;
            } else {
                node = self.find_or_insert_child(node.ptr(), name)?;
            }
        }

        Ok(node.into_inner().value)
    }

    fn find_child(&self, start_node: Ptr<ListNode<FsEntry>>, name: &OsStr) -> Option<FsEntry> {
        let name = name.to_str().unwrap();
        let mut cur_node = Some(start_node);
        while let Some(node) = cur_node {
            let value = self.tx.lookup(node);
            let child_name = value.value.name(&self.tx);
            if name == child_name.as_str() {
                return Some(value.into_inner().value);
            }
            cur_node = value.next;
        }
        None
    }

    fn find_or_insert_child(
        &mut self,
        start_node: Ptr<ListNode<FsEntry>>,
        name: &OsStr,
    ) -> Result<Handle<ListNode<FsEntry>>> {
        let name = name.to_str().unwrap();
        let mut prev_node = start_node;
        let mut cur_node = Some(start_node);
        while let Some(node) = cur_node {
            let value = self.tx.lookup(node);
            let child_name = value.value.name(&self.tx);
            if name == child_name.as_str() {
                return Ok(value);
            }
            prev_node = node;
            cur_node = value.next;
        }
        let new_node = self.write_fsnode(name);
        let mut prev_node = self.tx.lookup(prev_node);
        prev_node.next = Some(new_node.ptr());
        self.tx.update(&prev_node);
        Ok(new_node)
    }

    fn write_fsnode(&mut self, name: &str) -> Handle<ListNode<FsEntry>> {
        let name = self.tx.write_slice(name.as_bytes());
        let fs_entry = self
            .tx
            .write(FsEntry {
                name: Str(name),
                node_type: NodeType::Directory,
                children: None,
            })
            .into_inner();
        let node = ListNode {
            next: None,
            value: fs_entry.clone(),
        };
        self.tx.write(node)
    }
}

#[binrw]
#[brw(little)]
#[derive(Clone, Debug)]
struct FsEntry {
    name: Str,
    node_type: NodeType,

    #[br(parse_with = parse_optional_ptr)]
    #[bw(write_with = write_optional_ptr)]
    children: Option<Ptr<ListNode<FsEntry>>>,
}

impl FsEntry {
    fn name(&self, tx: &Transaction) -> String {
        let bytes = tx.read_slice(self.name.0);
        String::from_utf8(bytes).unwrap()
    }
}

#[binrw]
#[brw(little)]
#[derive(Debug)]
struct ListNode<T>
where
    for<'a> T: BinRead<Args<'a> = ()> + BinWrite<Args<'a> = ()>,
{
    #[br(parse_with = parse_optional_ptr)]
    #[bw(write_with = write_optional_ptr)]
    next: Option<Ptr<ListNode<T>>>,
    value: T,
}

#[binrw]
#[brw(repr = u8)]
#[derive(PartialEq, Clone, Debug)]
enum NodeType {
    Directory,
    File,
}

#[derive(Clone)]
#[binrw]
struct Str(SlicePtr<u8>);

impl fmt::Debug for Str {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("Str({:?})", &self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fs::NodeType, Memory, Storable};

    #[test]
    fn check_root() {
        let mem = Memory::default();
        let tx = mem.start();
        let fs = Filesystem::allocate(tx);

        let tx = &fs.tx;
        let root = fs.get_root();
        assert_eq!(root.name(&tx), "/");
        assert_eq!(root.node_type, NodeType::Directory);

        let node = fs.lookup("/foo").expect_err("Error should be generated");
        assert_eq!(node, Error::NotFound)
    }

    #[test]
    fn check_creating_directories() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/etc")?;

        let tx = &fs.tx;
        let node = fs.lookup("/etc").expect("/etc should be found");
        assert_eq!(node.name(&tx), "etc");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }

    #[test]
    fn check_creating_multiple_directories() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/usr/bin")?;

        let tx = &fs.tx;
        let node = fs.lookup("/usr/bin").expect("/usr/bin should be found");
        assert_eq!(node.name(&tx), "bin");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }
}
