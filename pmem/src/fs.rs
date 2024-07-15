use crate::{memory::SlicePtr, Handle, Ptr, Storable, Transaction};
use pmem_derive::Record;
use std::{
    ffi::OsStr,
    fmt,
    path::{Component, Path, PathBuf},
    string::FromUtf8Error,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("not found")]
    NotFound,

    #[error("not supported")]
    NotSupported,

    #[error("path must be absolute")]
    PathMustBeAbsolute,

    #[error("already exists")]
    AlreadyExists,

    #[error("pmem error")]
    PMemError(#[from] crate::memory::Error),

    #[error("Invalid utf8 encoding")]
    Utf8(#[from] FromUtf8Error),
}

struct Filesystem {
    volume: Handle<VolumeInfo>,
    tx: Transaction,
}

#[derive(Debug, Record)]
struct VolumeInfo {
    next_fid: u64,
    root: Ptr<FileInfo>,
}

impl Storable for Filesystem {
    type Seed = VolumeInfo;

    fn open(tx: Transaction, volume: Ptr<Self::Seed>) -> Self {
        let volume = tx.lookup(volume).unwrap();
        Self { volume, tx }
    }

    fn allocate(mut tx: Transaction) -> Self {
        let name = tx.write_slice("/".as_bytes());
        let root_entry = FileInfo {
            name: Str(name),
            fid: 1,
            node_type: NodeType::Directory.into(),
            children: None,
            next: None,
        };
        let root = tx.write(root_entry).ptr();
        let volume = tx.write(VolumeInfo { root, next_fid: 2 });
        Self { volume, tx }
    }

    fn finish(self) -> Transaction {
        todo!()
    }
}

impl Filesystem {
    pub fn get_root(&self) -> FileInfo {
        self.get_root_handle().into_inner()
    }

    fn get_root_handle(&self) -> Handle<FileInfo> {
        self.tx.lookup(self.volume.root).unwrap()
    }

    pub fn lookup(&self, path: impl AsRef<Path>) -> Result<Option<FileInfo>> {
        let mut components = path.as_ref().components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut cur_node = self.get_root_handle();
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            let Some(children) = cur_node.children else {
                return Ok(None);
            };
            let child = self.find_child(children, name)?;
            let Some(child) = child else {
                return Ok(None);
            };
            cur_node = child.node;
        }

        Ok(Some(cur_node.into_inner()))
    }

    pub fn delete(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut components = path.as_ref().components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut target_node = FileInfoReferent {
            referent: None,
            node: self.get_root_handle(),
        };
        let mut parent = None;
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            let current_node = target_node.node;
            let Some(children) = current_node.children else {
                return Err(Error::NotFound);
            };
            let child = self.find_child(children, name)?;
            let Some(child) = child else {
                return Err(Error::NotFound);
            };
            target_node = child;
            parent = Some(current_node);
        }

        let next_ptr = target_node.node.next;
        // TODO old node need to be removed
        // self.tx.reclaim(target_node.node);

        let Some(mut parent) = parent else {
            // Client tries to remove the root node
            return Err(Error::NotSupported);
        };

        if let Some(mut referent) = target_node.referent {
            // Child is not first in a list
            referent.next = next_ptr;
            self.tx.update(&referent);
        } else {
            parent.children = next_ptr;
            self.tx.update(&parent);
        }
        Ok(())
    }

    pub fn create_dir(&mut self, name: impl Into<PathBuf>) -> Result<FileInfo> {
        let path = name.into();

        let mut components = path.components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut node: Handle<FileInfo> = self.get_root_handle();
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            if node.children.is_none() {
                let new_node = self.write_fsnode(name.to_str().unwrap());
                node.children = Some(new_node.ptr());
                self.tx.update(&node);
                node = new_node;
            } else {
                node = self.find_or_insert_child(node.ptr(), name)?;
            }
        }

        Ok(node.into_inner())
    }

    fn find_child(
        &self,
        start_node: Ptr<FileInfo>,
        name: &OsStr,
    ) -> Result<Option<FileInfoReferent>> {
        let name = name.to_str().unwrap();
        let mut cur_node = Some(start_node);
        let mut referent = None;
        while let Some(node) = cur_node {
            let node = self.tx.lookup(node)?;
            let child_name = node.name(&self.tx)?;
            if name == child_name.as_str() {
                return Ok(Some(FileInfoReferent { referent, node }));
            }
            cur_node = node.next;
            referent = Some(node);
        }
        Ok(None)
    }

    fn find_or_insert_child(
        &mut self,
        start_node: Ptr<FileInfo>,
        name: &OsStr,
    ) -> Result<Handle<FileInfo>> {
        let name = name.to_str().unwrap();
        let mut prev_node = start_node;
        let mut cur_node = Some(start_node);
        while let Some(node) = cur_node {
            let value = self.tx.lookup(node)?;
            let child_name = value.name(&self.tx)?;
            if name == child_name.as_str() {
                return Ok(value);
            }
            prev_node = node;
            cur_node = value.next;
        }
        let new_node = self.write_fsnode(name);
        let mut prev_node = self.tx.lookup(prev_node)?;
        prev_node.next = Some(new_node.ptr());
        self.tx.update(&prev_node);
        Ok(new_node)
    }

    fn write_fsnode(&mut self, name: &str) -> Handle<FileInfo> {
        let fid = self.volume.next_fid;
        self.volume.next_fid += 1;
        let name = self.tx.write_slice(name.as_bytes());
        let entry = FileInfo {
            name: Str(name),
            fid,
            node_type: NodeType::Directory.into(),
            children: None,
            next: None,
        };
        self.tx.update(&self.volume);
        self.tx.write(entry)
    }
}

struct FileInfoReferent {
    // Previous FileInfo node in a linked list, if None the target node is the first one
    referent: Option<Handle<FileInfo>>,
    node: Handle<FileInfo>,
}

#[derive(Clone, Debug, Record)]
struct FileInfo {
    name: Str,
    // Unique file id on a volume
    fid: u64,
    node_type: u8,
    children: Option<Ptr<FileInfo>>,
    next: Option<Ptr<FileInfo>>,
}

impl FileInfo {
    fn name(&self, tx: &Transaction) -> Result<String> {
        let bytes = tx.read_slice(self.name.0)?;
        Ok(String::from_utf8(bytes)?)
    }
}

#[derive(PartialEq, Clone, Debug)]
enum NodeType {
    Directory,
    File,
}

impl Into<u8> for NodeType {
    fn into(self) -> u8 {
        match self {
            NodeType::Directory => 0,
            NodeType::File => 1,
        }
    }
}

impl From<u8> for NodeType {
    fn from(value: u8) -> Self {
        match value {
            0 => NodeType::Directory,
            1 => NodeType::File,
            _ => panic!("Invalid value for NodeType"),
        }
    }
}

#[derive(Clone, Record)]
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
    fn check_root() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let fs = Filesystem::allocate(tx);

        let tx = &fs.tx;
        let root = fs.get_root();
        assert_eq!(root.name(&tx)?, "/");
        assert_eq!(root.node_type, NodeType::Directory.into());

        let node = fs.lookup("/foo")?;
        assert!(node.is_none(), "Node should be missing");
        Ok(())
    }

    #[test]
    fn check_creating_directories() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/etc")?;

        let tx = &fs.tx;
        let node = fs.lookup("/etc")?.expect("/etc should be found");
        assert_eq!(node.name(&tx)?, "etc");
        assert_eq!(node.node_type, NodeType::Directory.into());

        Ok(())
    }

    #[test]
    fn check_creating_multiple_directories() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/usr/bin")?;

        let tx = &fs.tx;
        let node = fs.lookup("/usr/bin")?.expect("/usr/bin should be found");
        assert_eq!(node.name(&tx)?, "bin");
        assert_eq!(node.node_type, NodeType::Directory.into());

        Ok(())
    }

    #[test]
    fn can_delete_directory() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/usr")?;
        fs.delete("/usr")?;

        assert!(fs.lookup("/usr")?.is_none(), "/usr should be missing");

        Ok(())
    }

    #[test]
    fn check_each_fs_entry_has_its_own_id() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        fs.create_dir("/usr/lib/bin")?;

        let usr = fs.lookup("/usr")?.unwrap();
        let lib = fs.lookup("/usr/lib")?.unwrap();
        let bin = fs.lookup("/usr/lib/bin")?.unwrap();

        assert!(usr.fid > 0);
        assert!(lib.fid > 0);
        assert!(bin.fid > 0);

        assert!(usr.fid < lib.fid);
        assert!(lib.fid < bin.fid);

        Ok(())
    }
}
