use crate::{memory::SlicePtr, Handle, Ptr, Storable, Transaction};
use pmem_derive::Record;
use std::{
    convert::Into,
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

pub struct Filesystem {
    volume: Handle<VolumeInfo>,
    tx: Transaction,
}

#[derive(Debug, Record)]
pub struct VolumeInfo {
    next_fid: u64,
    root: Ptr<FNode>,
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

    fn take_if_created(self) -> Option<T> {
        match self {
            FoundOrCreated::Created(value) => Some(value),
            FoundOrCreated::Found(_) => None,
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
        let root_entry = FNode {
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
    pub fn get_root(&self) -> Result<FileMeta> {
        FileMeta::from(self.get_root_handle(), &self.tx)
    }

    fn get_root_handle(&self) -> Handle<FNode> {
        self.tx.lookup(self.volume.root).unwrap()
    }

    pub fn find(&self, path: impl AsRef<Path>) -> Result<Option<FileMeta>> {
        if let Some(info) = self.do_lookup_file(path)? {
            FileMeta::from(info, &self.tx).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn lookup(&self, dir: &FileMeta, name: impl AsRef<str>) -> Result<Option<FileMeta>> {
        let Some(children) = self.lookup_inode(dir)?.and_then(|dir| dir.children) else {
            return Ok(None);
        };

        self.find_child(children, name.as_ref())?
            .map(|child| FileMeta::from(child.node, &self.tx))
            .transpose()
    }

    fn lookup_inode(&self, meta: &FileMeta) -> Result<Option<Handle<FNode>>> {
        assert!(meta.fid <= u32::MAX as u64);

        Ptr::from_addr(meta.fid as u32)
            .map(|p| self.tx.lookup(p))
            .transpose()
            .map_err(|e| e.into())
    }

    pub fn lookup_by_id(&self, id: u64) -> Result<Option<FileMeta>> {
        assert!(id <= u32::MAX as u64);
        let Some(ptr) = Ptr::<FNode>::from_addr(id as u32) else {
            return Ok(None);
        };
        let handle = self.tx.lookup(ptr)?;
        FileMeta::from(handle, &self.tx).map(Some)
    }

    fn do_lookup_file(&self, path: impl AsRef<Path>) -> Result<Option<Handle<FNode>>> {
        let mut components = path.as_ref().components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };
        let mut cur_node = self.get_root_handle();
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };
            let name = name.to_str().unwrap();

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

    pub fn delete(&mut self, dir: &FileMeta, name: impl AsRef<str>) -> Result<()> {
        let mut dir = self.lookup_inode(dir)?.ok_or(Error::NotFound)?;

        let Some(children) = dir.children else {
            return Err(Error::NotFound);
        };
        let file = self
            .find_child(children, name.as_ref())?
            .ok_or(Error::NotFound)?;

        let next_ptr = file.node.next;
        self.tx.reclaim(file.node);

        if let Some(mut referent) = file.referent {
            // Child is not first in a list
            referent.next = next_ptr;
            self.tx.update(&referent);
        } else {
            dir.children = next_ptr;
            self.tx.update(&dir);
        }
        Ok(())
    }

    pub fn create_file<'a>(&mut self, dir: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let file_name = name.as_ref();

        let mut dir = self.lookup_inode(dir)?.ok_or(Error::NotFound)?;
        let file_info = if let Some(children) = dir.children {
            self.find_or_create_child(children, file_name, NodeType::File)?
                .take_if_created()
                .ok_or(Error::AlreadyExists)?
        } else {
            let new_node = self.write_fsnode(file_name, NodeType::File);
            dir.children = Some(new_node.ptr());
            self.tx.update(&dir);
            new_node
        };

        FileMeta::from(file_info, &self.tx)
    }

    pub fn open_file<'a>(&'a mut self, file: &FileMeta) -> Result<File<'a>> {
        let file_info = self.lookup_inode(file)?.ok_or(Error::NotFound)?;

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

    pub fn create_dir(&mut self, parent: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let mut parent = self.lookup_inode(parent)?.ok_or(Error::NotFound)?;

        let directory_inode = if let Some(first_child) = parent.children {
            self.find_or_create_child(first_child, name.as_ref(), NodeType::Directory)?
                .take_if_created()
                .ok_or(Error::AlreadyExists)?
        } else {
            let dir_inode = self.write_fsnode(name.as_ref(), NodeType::Directory);
            parent.children = Some(dir_inode.ptr());
            self.tx.update(&parent);
            dir_inode
        };

        FileMeta::from(directory_inode, &self.tx)
    }

    pub fn readdir(&self, dir: &FileMeta) -> Result<Vec<FileMeta>> {
        let parent = self.lookup_inode(dir)?.ok_or(Error::NotFound)?;

        if let Some(children) = parent.children {
            let mut result = vec![];

            let mut current_node = Some(children);
            while let Some(node) = current_node {
                let child = self.tx.lookup(node)?;
                current_node = child.next;
                result.push(FileMeta::from(child, &self.tx)?);
            }
            Ok(result)
        } else {
            Ok(vec![])
        }
    }

    pub fn create_dirs(&mut self, name: impl Into<PathBuf>) -> Result<FileMeta> {
        let path = name.into();

        let mut components = path.components();
        let Some(Component::RootDir) = components.next() else {
            return Err(Error::PathMustBeAbsolute);
        };

        let mut node: Handle<FNode> = self.get_root_handle();
        for component in components {
            let Component::Normal(name) = component else {
                return Err(Error::NotSupported);
            };

            node = if node.children.is_none() {
                let new_node = self.write_fsnode(name.to_str().unwrap(), NodeType::Directory);
                node.children = Some(new_node.ptr());
                self.tx.update(&node);
                new_node
            } else {
                self.find_or_create_child(node.ptr(), name.to_str().unwrap(), NodeType::Directory)?
                    .into()
            }
        }

        FileMeta::from(node, &self.tx)
    }

    fn find_child(&self, start_node: Ptr<FNode>, name: &str) -> Result<Option<FileInfoReferent>> {
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
    fn find_or_create_child(
        &mut self,
        start_node: Ptr<FNode>,
        name: &str,
        node_type: NodeType,
    ) -> Result<FoundOrCreated<Handle<FNode>>> {
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
        let new_node = self.write_fsnode(name, node_type);
        let mut prev_node = self.tx.lookup(prev_node)?;
        prev_node.next = Some(new_node.ptr());
        self.tx.update(&prev_node);
        Ok(FoundOrCreated::Created(new_node))
    }

    fn write_fsnode(&mut self, name: &str, node_type: NodeType) -> Handle<FNode> {
        let fid = self.volume.next_fid;
        self.volume.next_fid += 1;
        let name = self.tx.write_slice(name.as_bytes());
        let entry = FNode {
            name: Str(name),
            fid,
            node_type: node_type.into(),
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
    file_info: Handle<FNode>,
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
        if let Some(_old_content) = self.file_info.file_content.take() {
            // Remove old content
        }
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
    referent: Option<Handle<FNode>>,
    node: Handle<FNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FileMeta {
    pub name: String,
    // Unique file id on a volume
    pub fid: u64,
    pub node_type: NodeType,
}

impl FileMeta {
    fn from(file_info: Handle<FNode>, tx: &Transaction) -> Result<Self> {
        let name = file_info.name(tx)?;
        Ok(FileMeta {
            name,
            fid: u64::from(file_info.ptr().unwrap_addr()),
            node_type: NodeType::from(file_info.node_type),
        })
    }
}

#[derive(Clone, Debug, Record)]
struct FNode {
    name: Str,
    // Unique file id on a volume
    fid: u64,
    node_type: u8,
    children: Option<Ptr<FNode>>,
    file_content: Option<SlicePtr<u8>>,
    next: Option<Ptr<FNode>>,
}

impl FNode {
    fn name(&self, tx: &Transaction) -> Result<String> {
        let bytes = tx.read_slice(self.name.0)?;
        Ok(String::from_utf8(bytes)?)
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum NodeType {
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
        let (fs, _) = create_fs();

        let root = fs.get_root()?;
        assert_eq!(root.name, "/");
        assert_eq!(root.node_type, NodeType::Directory);
        Ok(())
    }

    #[test]
    fn check_find_should_return_none_if_dir_is_missing() -> Result<()> {
        let (fs, _) = create_fs();

        let node = fs.find("/foo")?;
        assert!(node.is_none(), "Node should be missing");
        Ok(())
    }

    #[test]
    fn check_lookup_dir() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/etc")?;
        let root = fs.get_root()?;
        let node = fs.lookup(&root, "etc")?;
        assert!(node.is_some(), "Node should be present");
        Ok(())
    }

    #[test]
    fn check_creating_directories() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "etc")?;
        assert!(fs.lookup(&root, "etc")?.is_some(), "Dir should be present");

        let node = fs.find("/etc")?.expect("/etc should be found");
        assert_eq!(node.name, "etc");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }

    #[test]
    fn check_creating_multiple_directories() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/usr/bin")?;

        let node = fs.find("/usr/bin")?.expect("/usr/bin should be found");
        assert_eq!(node.name, "bin");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }

    #[test]
    fn can_delete_directory() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "usr")?;
        fs.delete(&root, "usr")?;

        assert!(fs.find("/usr")?.is_none(), "/usr should be missing");

        Ok(())
    }

    #[test]
    fn can_delete_file() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_file(&root, "swap")?;
        fs.delete(&root, "swap")?;

        assert!(fs.find("/swap")?.is_none(), "/swap should be missing");

        Ok(())
    }

    #[test]
    fn can_remove_several_items() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        let etc = fs.create_dir(&root, "etc")?;
        let swap = fs.create_file(&root, "swap")?;

        fs.delete(&root, "etc")?;
        assert!(fs.find("/etc")?.is_none(), "/etc should be missing");

        assert!(
            fs.lookup(&root, "swap")?.is_some(),
            "/swap should be present"
        );
        fs.delete(&root, "swap")?;
        assert!(fs.find("/swap")?.is_none(), "/swap should be missing");

        Ok(())
    }

    #[test]
    fn check_each_fs_entry_has_its_own_id() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/usr/lib/bin")?;

        let usr = fs.find("/usr")?.ok_or(Error::NotFound)?;
        let lib = fs.find("/usr/lib")?.ok_or(Error::NotFound)?;
        let bin = fs.find("/usr/lib/bin")?.ok_or(Error::NotFound)?;

        assert!(usr.fid > 0);
        assert!(lib.fid > 0);
        assert!(bin.fid > 0);

        assert!(usr.fid < lib.fid);
        assert!(lib.fid < bin.fid);

        let bin_ref = fs.lookup_by_id(bin.fid)?.ok_or(Error::NotFound)?;
        assert_eq!(bin.name, bin_ref.name);

        Ok(())
    }

    #[test]
    fn create_file() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        let file = fs.create_file(&root, "file.txt")?;
        assert!(
            fs.lookup(&root, "file.txt")?.is_some(),
            "File should be present"
        );
        let mut file = fs.open_file(&file)?;
        let expected_content = "Hello world";
        file.write(expected_content.as_bytes()).unwrap();
        file.flush()?;

        let file_meta = fs.find("/file.txt")?.expect("File should be found");
        assert_eq!(file_meta.node_type, NodeType::File);

        let mut file = fs.open_file(&file_meta)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        assert_eq!(content, expected_content);

        Ok(())
    }

    #[test]
    fn write_file_partially() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        let file_meta = fs.create_file(&root, "file.txt")?;
        let mut file = fs.open_file(&file_meta)?;
        file.write("Hello world".as_bytes())?;
        file.flush()?;

        let mut file = fs.open_file(&file_meta)?;
        file.seek(SeekFrom::Start(6))?;
        file.write("Rust!".as_bytes())?;
        file.flush()?;

        let mut file = fs.open_file(&file_meta)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        assert_eq!(content, "Hello Rust!");

        Ok(())
    }

    #[test]
    fn readdir() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;

        let etc = fs.create_dir(&root, "etc")?;
        let bin = fs.create_dir(&root, "bin")?;
        let swap = fs.create_file(&root, "swap")?;

        let children = fs.readdir(&root)?;

        assert!(children.contains(&etc), "Root should contains etc");
        assert!(children.contains(&bin), "Root should contains bin");
        assert!(children.contains(&swap), "Root should contains spaw");

        Ok(())
    }

    fn create_fs() -> (Filesystem, Memory) {
        let mem = Memory::default();
        let tx = mem.start();
        (Filesystem::allocate(tx), mem)
    }
}
