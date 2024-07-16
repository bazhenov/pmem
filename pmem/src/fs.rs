use crate::{memory::SlicePtr, Handle, Ptr, Storable, Transaction};
use pmem_derive::Record;
use std::{
    convert::Into,
    ffi::OsStr,
    fmt,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
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

    #[error("IO Error")]
    IOError(#[from] std::io::Error),

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

enum FoundOrCreated<T> {
    Found(T),
    Created(T),
}

impl<T> FoundOrCreated<T> {
    fn into(self) -> T {
        match self {
            FoundOrCreated::Found(value) => value,
            FoundOrCreated::Created(value) => value,
        }
    }
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
            file_content: None,
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
        Ok(self.do_lookup_file(path)?.map(Handle::into_inner))
    }

    fn do_lookup_file(&self, path: impl AsRef<Path>) -> Result<Option<Handle<FileInfo>>> {
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
        Ok(Some(cur_node))
    }

    pub fn delete(&mut self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let mut components = path.components();
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
            let children = current_node.children.ok_or(Error::NotFound)?;
            target_node = self.find_child(children, name)?.ok_or(Error::NotFound)?;
            parent = Some(current_node);
        }

        let next_ptr = target_node.node.next;
        self.tx.reclaim(target_node.node);

        // Client tries to remove the root node
        let mut parent = parent.ok_or(Error::NotSupported)?;

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

    pub fn create_file<'a>(&'a mut self, path: impl Into<PathBuf>) -> Result<File<'a>> {
        let mut path: PathBuf = path.into();
        let file_name = path.file_name().ok_or(Error::NotSupported)?.to_owned();
        assert!(path.pop());

        let mut parent = self.do_lookup_file(path)?.ok_or(Error::NotFound)?;
        let file_info = if let Some(children) = parent.children {
            let FoundOrCreated::Created(file_info) =
                self.find_or_insert_child(children, file_name.as_os_str())?
            else {
                return Err(Error::AlreadyExists);
            };
            file_info
        } else {
            let new_node = self.write_fsnode(file_name.to_str().unwrap());
            parent.children = Some(new_node.ptr());
            self.tx.update(&parent);
            new_node
        };

        Ok(File {
            cursor: Cursor::new(vec![]),
            fs: self,
            file_info,
        })
    }

    pub fn open_file<'a>(&'a mut self, path: impl AsRef<Path>) -> Result<File<'a>> {
        let file_info = self.do_lookup_file(path)?.ok_or(Error::NotFound)?;

        let content = if let Some(content) = file_info.file_content {
            self.tx.read_slice(content)?
        } else {
            vec![]
        };

        Ok(File {
            cursor: Cursor::new(content),
            fs: self,
            file_info,
        })
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

            node = if node.children.is_none() {
                let new_node = self.write_fsnode(name.to_str().unwrap());
                node.children = Some(new_node.ptr());
                self.tx.update(&node);
                new_node
            } else {
                self.find_or_insert_child(node.ptr(), name)?.into()
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

    /// Returns `Ok(child)` if it was found otherwise create new [FileInfo] and returns `Err(new_child)`
    fn find_or_insert_child(
        &mut self,
        start_node: Ptr<FileInfo>,
        name: &OsStr,
    ) -> Result<FoundOrCreated<Handle<FileInfo>>> {
        let name = name.to_str().unwrap();
        let mut prev_node = start_node;
        let mut cur_node = Some(start_node);
        while let Some(node) = cur_node {
            let value = self.tx.lookup(node)?;
            let child_name = value.name(&self.tx)?;
            if name == child_name.as_str() {
                return Ok(FoundOrCreated::Found(value));
            }
            prev_node = node;
            cur_node = value.next;
        }
        let new_node = self.write_fsnode(name);
        let mut prev_node = self.tx.lookup(prev_node)?;
        prev_node.next = Some(new_node.ptr());
        self.tx.update(&prev_node);
        Ok(FoundOrCreated::Created(new_node))
    }

    fn write_fsnode(&mut self, name: &str) -> Handle<FileInfo> {
        let fid = self.volume.next_fid;
        self.volume.next_fid += 1;
        let name = self.tx.write_slice(name.as_bytes());
        let entry = FileInfo {
            name: Str(name),
            fid,
            node_type: NodeType::Directory.into(),
            file_content: None,
            children: None,
            next: None,
        };
        self.tx.update(&self.volume);
        self.tx.write(entry)
    }
}

pub struct File<'a> {
    cursor: Cursor<Vec<u8>>,
    file_info: Handle<FileInfo>,
    fs: &'a mut Filesystem,
}

impl<'a> Read for File<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl<'a> Write for File<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        let slice = self.fs.tx.write_slice(self.cursor.get_ref());
        self.file_info.file_content = Some(slice);
        self.fs.tx.update(&self.file_info);
        Ok(())
    }
}

impl<'a> Seek for File<'a> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

/// Helper structure that is used when iterating over linked list of [FileInfo] instances
///
/// Contains the node itself as well as the previous node. It is usefull when we need to remove the node
/// from the list, and update the reference of the previous node.
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
    file_content: Option<SlicePtr<u8>>,
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

    #[test]
    fn create_file() -> Result<()> {
        let mem = Memory::default();
        let tx = mem.start();
        let mut fs = Filesystem::allocate(tx);

        let mut file = fs.create_file("/file.txt")?;
        let expected_content = "Hello world";
        file.write(expected_content.as_bytes()).unwrap();
        file.flush()?;

        let mut file = fs.open_file("/file.txt")?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        assert_eq!(content, expected_content);

        Ok(())
    }
}
