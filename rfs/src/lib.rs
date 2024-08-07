use pmem::{
    memory::{self, SlicePtr, PTR_SIZE},
    Handle, Ptr, Record, Transaction,
};
use pmem_derive::Record;
use std::{
    cmp::Ordering,
    convert::Into,
    fmt,
    io::{self, Error, ErrorKind, Read, Seek, SeekFrom, Write},
    ops::{Deref, DerefMut},
    path::{self, Component, Path, PathBuf},
};
use tracing::{instrument, trace};

type Result<T> = std::io::Result<T>;
type CreateResult = std::result::Result<Handle<FNode>, Handle<FNode>>;

pub mod nfs;

pub struct Filesystem {
    volume: Handle<VolumeInfo>,
    tx: Transaction,
}

#[derive(Debug, Record)]
pub struct VolumeInfo {
    root: Ptr<FNode>,
}

impl Filesystem {
    pub fn open(tx: Transaction) -> Self {
        let volume = tx.read_super_block().unwrap();
        Self { volume, tx }
    }

    pub fn allocate(mut tx: Transaction) -> Self {
        let super_block_ptr = tx.alloc::<VolumeInfo>().unwrap();
        let name = tx.write_slice("/".as_bytes()).unwrap();
        let root_entry = FNode {
            name: Str(name),
            size: 0,
            node_type: NodeType::Directory,
            children: None,
            content: BlockPointers::default(),
            next: None,
        };
        let root = tx.write(root_entry).unwrap().ptr();
        let volume = tx.write_at(super_block_ptr, VolumeInfo { root }).unwrap();
        Self { volume, tx }
    }
}

impl Filesystem {
    pub fn get_root(&self) -> Result<FileMeta> {
        FileMeta::from(self.get_root_handle(), &self.tx)
    }

    /// Finds the file/directory at the given path
    ///
    /// Return [ErrorKind::NotFound] if the file/directory does not exist
    pub fn find(&self, path: impl AsRef<str>) -> Result<FileMeta> {
        FileMeta::from(self.do_lookup_file(path)?, &self.tx)
    }

    /// Finds the given child in the given directory
    ///
    /// Return [ErrorKind::NotFound] if the child does not exist
    pub fn lookup(&self, dir: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let children = self
            .lookup_inode(dir)?
            .children
            .ok_or(ErrorKind::NotFound)?;
        let child = self.find_child(children, name.as_ref())?;
        FileMeta::from(child.node, &self.tx)
    }

    /// Returns the file/directory with a given inode ([FileMeta::fid])
    pub fn lookup_by_id(&self, id: u64) -> Result<FileMeta> {
        assert!(id <= u32::MAX as u64);
        let ptr = Ptr::<FNode>::from_addr(id as u32).ok_or(ErrorKind::NotFound)?;
        let handle = self.tx.lookup(ptr)?;
        FileMeta::from(handle, &self.tx)
    }

    pub fn delete(&mut self, dir: &FileMeta, name: impl AsRef<str>) -> Result<()> {
        let mut dir = self.lookup_inode(dir)?;

        let children = dir.children.ok_or(ErrorKind::NotFound)?;
        let file = self.find_child(children, name.as_ref())?;

        let next_ptr = file.node.next;
        self.tx.reclaim(file.node);

        if let Some(mut referent) = file.referent {
            // Child is not first in a list
            referent.next = next_ptr;
            self.tx.update(&referent)?;
        } else {
            dir.children = next_ptr;
            self.tx.update(&dir)?;
        }
        Ok(())
    }

    pub fn create_file(&mut self, dir: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let file_name = name.as_ref();

        let mut dir = self.lookup_inode(dir)?;
        let file_info = if let Some(children) = dir.children {
            let child = self
                .create_child(children, file_name, NodeType::File)?
                .ok()
                .ok_or(ErrorKind::AlreadyExists)?;
            // Update the parent directory if the new child is the first in the list
            if child.next == Some(children) {
                dir.children = Some(child.ptr());
                self.tx.update(&dir)?;
            }
            child
        } else {
            let new_node = self.write_fsnode(file_name, NodeType::File)?;
            dir.children = Some(new_node.ptr());
            self.tx.update(&dir)?;
            new_node
        };

        FileMeta::from(file_info, &self.tx)
    }

    /// Opens a file for reading and writing
    pub fn open_file<'a>(&'a mut self, file: &FileMeta) -> Result<File<'a>> {
        let file_info = self.lookup_inode(file)?;

        Ok(File {
            pos: 0,
            fs: self,
            meta: file_info,
        })
    }

    /// Creates a new directory in the given parent directory
    ///
    /// Returns:
    /// - [Error::AlreadyExists] if the directory already exists;
    /// - [ErrorKind::NotFound] if the parent directory does not exist;
    pub fn create_dir(&mut self, parent: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let mut parent = self.lookup_inode(parent)?;

        let directory_inode = if let Some(first_child) = parent.children {
            let new_child = self
                .create_child(first_child, name.as_ref(), NodeType::Directory)?
                .ok()
                .ok_or(ErrorKind::AlreadyExists)?;
            // Update the parent directory if the new child is the first in the list
            if new_child.next == Some(first_child) {
                parent.children = Some(new_child.ptr());
                self.tx.update(&parent)?;
            }
            new_child
        } else {
            let dir_inode = self.write_fsnode(name.as_ref(), NodeType::Directory)?;
            parent.children = Some(dir_inode.ptr());
            self.tx.update(&parent)?;
            dir_inode
        };

        FileMeta::from(directory_inode, &self.tx)
    }

    /// Reads the contents of a directory
    ///
    /// Returns:
    /// - [ErrorKind::NotFound] if the directory does not exist
    pub fn readdir<'a>(
        &'a self,
        dir: &FileMeta,
    ) -> Result<impl Iterator<Item = Result<FileMeta>> + 'a> {
        let parent = self.lookup_inode(dir)?;

        Ok(ReadDir {
            fs: self,
            next: parent.children,
        })
    }

    /// Creates a directory and all its parents
    pub fn create_dirs(&mut self, name: impl AsRef<str>) -> Result<FileMeta> {
        let path = PathBuf::from(name.as_ref());
        let mut node = self.get_root_handle();
        for component in components(&path)? {
            let Component::Normal(name) = component else {
                return Err(ErrorKind::InvalidInput.into());
            };

            node = if node.children.is_none() {
                let new_node = self.write_fsnode(name.to_str().unwrap(), NodeType::Directory)?;
                node.children = Some(new_node.ptr());
                self.tx.update(&node)?;
                new_node
            } else {
                self.create_child(node.ptr(), name.to_str().unwrap(), NodeType::Directory)?
                    .unwrap_or_else(|found_dir| found_dir)
            }
        }

        FileMeta::from(node, &self.tx)
    }

    fn get_root_handle(&self) -> Handle<FNode> {
        self.tx.lookup(self.volume.root).unwrap()
    }

    fn lookup_inode(&self, meta: &FileMeta) -> Result<Handle<FNode>> {
        assert!(meta.fid <= u32::MAX as u64);

        let ptr = Ptr::from_addr(meta.fid as u32).ok_or(ErrorKind::NotFound)?;
        self.tx.lookup(ptr).map_err(|e| e.into())
    }

    fn do_lookup_file(&self, path: impl AsRef<str>) -> Result<Handle<FNode>> {
        let path = PathBuf::from(path.as_ref());
        let components = components(&path)?;
        let mut cur_node = self.get_root_handle();
        for component in components {
            let Component::Normal(name) = component else {
                return Err(ErrorKind::InvalidInput.into());
            };
            let name = name.to_str().unwrap();

            let children = cur_node.children.ok_or(ErrorKind::NotFound)?;
            let child = self.find_child(children, name)?;
            cur_node = child.node;
        }
        Ok(cur_node)
    }

    fn find_child(&self, start_node: Ptr<FNode>, name: &str) -> Result<FileInfoReferent> {
        let mut cur_node = Some(start_node);
        let mut referent = None;

        while let Some(node) = cur_node {
            let node = self.tx.lookup(node)?;
            let child_name = node.name(&self.tx)?;
            if name == child_name.as_str() {
                return Ok(FileInfoReferent { referent, node });
            }
            cur_node = node.next;
            referent = Some(node);
        }
        Err(ErrorKind::NotFound.into())
    }

    /// Returns `Ok(child)` if it was created successfully otherwise returns existing child: `Err(child)`
    fn create_child(
        &mut self,
        start_node: Ptr<FNode>,
        name: &str,
        node_type: NodeType,
    ) -> Result<CreateResult> {
        let mut prev_node = None;
        let mut cur_node = Some(start_node);
        while let Some(node) = cur_node {
            let node = self.tx.lookup(node)?;
            let child_name = node.name(&self.tx)?;
            match child_name.as_str().cmp(name) {
                Ordering::Equal => return Ok(CreateResult::Err(node)),
                Ordering::Greater => break,
                Ordering::Less => {
                    prev_node = cur_node;
                    cur_node = node.next;
                }
            }
        }

        let mut new_node = self.write_fsnode(name, node_type)?;
        new_node.next = cur_node;
        self.tx.update(&new_node)?;

        if let Some(prev_node) = prev_node {
            let mut prev_node = self.tx.lookup(prev_node)?;
            prev_node.next = Some(new_node.ptr());
            self.tx.update(&prev_node)?;
        }

        Ok(CreateResult::Ok(new_node))
    }

    fn write_fsnode(&mut self, name: &str, node_type: NodeType) -> Result<Handle<FNode>> {
        let name = self.tx.write_slice(name.as_bytes())?;
        let entry = FNode {
            name: Str(name),
            node_type,
            size: 0,
            content: BlockPointers::default(),
            children: None,
            next: None,
        };
        self.tx.update(&self.volume)?;
        Ok(self.tx.write(entry)?)
    }
}

/// Makes sure that the path is absolute and returns its components skipping the root
fn components(path: &Path) -> Result<path::Components> {
    let mut components = path.components();
    let Some(Component::RootDir) = components.next() else {
        return Err(ErrorKind::InvalidInput.into());
    };
    Ok(components)
}

pub struct File<'a> {
    pos: u64,
    meta: Handle<FNode>,
    fs: &'a mut Filesystem,
}

impl<'a> File<'a> {
    /// Allocates given number of blocks and adds them to the file content
    #[instrument(skip(self))]
    fn allocate(&mut self, blocks: u64) -> Result<()> {
        let blocks_to_allocate = blocks;
        let mut blocks = blocks_required(self.meta.size);
        let blocks_required = blocks + blocks_to_allocate;

        let tx = &mut self.fs.tx;

        // Allocating direct data blocks if needed
        let blocks_before = blocks;
        while blocks < blocks_required && blocks <= LAST_DIRECT_BLOCK as u64 {
            make_sure_data_block_exists(tx, &mut self.meta.content.direct[blocks as usize])?;
            blocks += 1;
        }
        if blocks > blocks_before {
            trace!(cnt = blocks - blocks_before, "Allocated direct blocks");
        }

        // Allocating indirect data block if needed
        if blocks < blocks_required {
            let mut indirect_block =
                make_sure_ptr_block_exists(tx, &mut self.meta.content.indirect)?;

            let blocks_before = blocks;
            while blocks < blocks_required && blocks <= LAST_INDIRECT_BLOCK as u64 {
                let idx = blocks as usize - FIRST_INDIRECT_BLOCK;
                make_sure_data_block_exists(tx, &mut indirect_block[idx])?;
                blocks += 1;
            }
            tx.update(&indirect_block)?;
            if blocks > blocks_before {
                trace!(cnt = blocks - blocks_before, "Allocated indirect blocks");
            }
        }

        // Allocating double indirect data block if needed
        if blocks < blocks_required {
            let mut double_indirect_block =
                make_sure_ptr_block_exists(tx, &mut self.meta.content.double_indirect)?;

            let blocks_before = blocks;
            while blocks < blocks_required && blocks <= LAST_DOUBLE_INDIRECT_BLOCK as u64 {
                let idx = blocks as usize - FIRST_DOUBLE_INDIRECT_BLOCK;

                // Because we have two level of indirection we need to calculate the index of the
                // two blocks that we need to update
                let a_idx = idx / POINTERS_PER_BLOCK;
                let b_idx = idx % POINTERS_PER_BLOCK;
                let mut a_block =
                    make_sure_ptr_block_exists(tx, &mut double_indirect_block[a_idx])?;
                make_sure_data_block_exists(tx, &mut a_block[b_idx])?;
                // Here is possible performance improvement, technically we don't need to update
                // the indirect block every time, but we can do it in the end
                tx.update(&a_block)?;
                blocks += 1;
            }
            tx.update(&double_indirect_block)?;

            if blocks > blocks_before {
                trace!(
                    cnt = blocks - blocks_before,
                    "Allocated double-indirect blocks"
                );
            }
        }
        tx.update(&self.meta)?;
        Ok(())
    }

    fn get_current_block(&mut self) -> Result<Handle<DataBlock>> {
        let block_no = block_idx(self.pos) as usize;
        let tx = &mut self.fs.tx;

        match block_no {
            0..=LAST_DIRECT_BLOCK => {
                trace!("Current block: Direct({})", block_no);
                let block_ptr = self.meta.content.direct[block_no].expect("Direct block missing");
                Ok(tx.lookup(block_ptr)?)
            }
            FIRST_INDIRECT_BLOCK..=LAST_INDIRECT_BLOCK => {
                let relative_block_no = block_no - FIRST_INDIRECT_BLOCK;
                let block_ptr = lookup_step(tx, self.meta.content.indirect, relative_block_no)
                    .expect("Indirect data block missing");
                trace!("Current block: Indirect({})", relative_block_no);
                Ok(tx.lookup(block_ptr)?)
            }
            FIRST_DOUBLE_INDIRECT_BLOCK..=LAST_DOUBLE_INDIRECT_BLOCK => {
                let relative_block_no = block_no - FIRST_DOUBLE_INDIRECT_BLOCK;
                let a_idx = relative_block_no / POINTERS_PER_BLOCK;
                let b_idx = relative_block_no % POINTERS_PER_BLOCK;

                let a_ptr = lookup_step(tx, self.meta.content.double_indirect, a_idx);
                let b_ptr = lookup_step(tx, a_ptr, b_idx).expect("Data block missing");

                trace!("Current block: Double indirect({})", relative_block_no);
                Ok(tx.lookup(b_ptr)?)
            }
            _ => unimplemented!("File too large"),
        }
    }
}

struct ReadDir<'a> {
    fs: &'a Filesystem,
    next: Option<Ptr<FNode>>,
}

impl Iterator for ReadDir<'_> {
    type Item = Result<FileMeta>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(node) = self.next.take() else {
            return None;
        };
        match self.fs.tx.lookup(node) {
            Ok(child) => {
                self.next = child.next;
                Some(FileMeta::from(child, &self.fs.tx))
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

fn make_sure_data_block_exists(
    tx: &mut Transaction,
    ptr: &mut Option<Ptr<DataBlock>>,
) -> Result<Handle<DataBlock>> {
    if let Some(ptr) = ptr {
        Ok(tx.lookup(*ptr)?)
    } else {
        let data_block = tx.write(DataBlock::default())?;
        *ptr = Some(data_block.ptr());
        Ok(data_block)
    }
}

fn make_sure_ptr_block_exists<T: Record>(
    tx: &mut Transaction,
    ptr: &mut Option<Ptr<PointersBlock<T>>>,
) -> Result<Handle<PointersBlock<T>>> {
    if let Some(ptr) = ptr {
        Ok(tx.lookup(*ptr)?)
    } else {
        trace!("Allocating double indirect pointers block");
        let indirect_block = tx.write::<PointersBlock<T>>([None; BLOCK_SIZE / PTR_SIZE])?;
        *ptr = Some(indirect_block.ptr());
        Ok(indirect_block)
    }
}

fn lookup_step<T>(
    tx: &mut Transaction,
    block_ptr: NullPtr<PointersBlock<T>>,
    idx: usize,
) -> NullPtr<T> {
    let block_ptr = block_ptr.expect("Block pointer missing");
    let block = tx.lookup(block_ptr).expect("Unable to read block");
    block[idx]
}

impl<'a> Read for File<'a> {
    #[instrument(level = "trace", skip(self, buf), fields(buf.len = buf.len(), pos=self.pos), ret)]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // we must use saturating_seb here, because pos can technically be larger than file size
        let bytes_left_in_file = self.meta.size.saturating_sub(self.pos);
        if bytes_left_in_file == 0 {
            return Ok(0);
        }

        let offset = self.pos as usize % BLOCK_SIZE;
        let bytes_left_in_block = BLOCK_SIZE - offset;

        let len = buf
            .len()
            .min(bytes_left_in_block)
            .min(bytes_left_in_file as usize);

        let block = self.get_current_block()?;
        buf[..len].copy_from_slice(&block[offset..offset + len]);

        self.pos += len as u64;
        Ok(len)
    }
}

impl<'a> Write for File<'a> {
    #[instrument(level = "trace", skip(self, buf), fields(buf.len = buf.len(), pos=self.pos), ret)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // user can seek after the end of file and write there, so we use self.pos here
        // TODO what we right after the end of file, should we extend the file?
        if self.pos + buf.len() as u64 > self.meta.size {
            // there is not enough space in the file, we need to allocate more blocks
            let current_blocks = blocks_required(self.meta.size);
            let new_blocks = blocks_required(self.pos + buf.len() as u64);
            let blocks_to_allocate = new_blocks - current_blocks;
            if blocks_to_allocate > 0 {
                self.allocate(blocks_to_allocate)?;
            }
        }

        let mut block = self.get_current_block()?;

        let offset = self.pos as usize % BLOCK_SIZE;
        let bytes_left = BLOCK_SIZE - offset;
        let len = buf.len().min(bytes_left);

        block[offset..(offset + len)].copy_from_slice(&buf[..len]);
        self.fs.tx.update(&block)?;

        self.pos += len as u64;
        self.meta.size = self.meta.size.max(self.pos);

        self.fs.tx.update(&self.meta)?;

        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> Seek for File<'a> {
    #[instrument(level = "trace", skip(self), fields(curr=self.pos), ret)]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Calculate new position in a file
        let abs_pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(pos) => ((self.meta.size as i64) + pos) as u64,
            SeekFrom::Current(pos) => (self.pos as i64 + pos) as u64,
        };

        self.pos = abs_pos;
        Ok(abs_pos)
    }
}

/// Returns the number of blocks required to store `n` bytes
fn blocks_required(n: u64) -> u64 {
    (n + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64
}

/// Returns the index of the block that contains the byte at position `n`
fn block_idx(n: u64) -> u64 {
    n / BLOCK_SIZE as u64
}

/// Helper structure that is used when iterating over linked list of [FileInfo] instances
///
/// Contains the node itself as well as the previous node. It is useful when we need to remove the node
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
    pub size: u64,
}

impl FileMeta {
    fn from(file_info: Handle<FNode>, tx: &Transaction) -> Result<Self> {
        let name = file_info.name(&tx)?;
        Ok(FileMeta {
            name,
            fid: u64::from(file_info.ptr().unwrap_addr()),
            size: file_info.size,
            node_type: file_info.node_type,
        })
    }
}

#[derive(Clone, Debug, Record)]
struct FNode {
    name: Str,
    node_type: NodeType,
    size: u64,
    children: Option<Ptr<FNode>>,
    content: BlockPointers,
    next: Option<Ptr<FNode>>,
}

/// The size of a data block containing file data in bytes
///
/// It is analogous to the block size (cluster) of a file system. File size in the storage
/// is increased in multiples of this size.
#[cfg(not(test))]
const BLOCK_SIZE: usize = 4096;

/// For the tests we use a smaller block size to be able to hit all code paths that are related
/// to indirect and double indirect blocks. Also error messages are more comprehensible.
#[cfg(test)]
const BLOCK_SIZE: usize = 64;

/// The number of pointers that fit into a block
const POINTERS_PER_BLOCK: usize = BLOCK_SIZE / PTR_SIZE;

const DIRECT_BLOCKS: usize = 10;
const INDIRECT_BLOCKS: usize = BLOCK_SIZE / PTR_SIZE;
const DOUBLE_INDIRECT_BLOCKS: usize = INDIRECT_BLOCKS * INDIRECT_BLOCKS;

const LAST_DIRECT_BLOCK: usize = DIRECT_BLOCKS - 1;

const FIRST_INDIRECT_BLOCK: usize = LAST_DIRECT_BLOCK + 1;
const LAST_INDIRECT_BLOCK: usize = FIRST_INDIRECT_BLOCK + INDIRECT_BLOCKS - 1;

const FIRST_DOUBLE_INDIRECT_BLOCK: usize = LAST_INDIRECT_BLOCK + 1;
const LAST_DOUBLE_INDIRECT_BLOCK: usize = FIRST_DOUBLE_INDIRECT_BLOCK + DOUBLE_INDIRECT_BLOCKS - 1;

/// Simple type alias to minimize the amount of angle brackets in the code
type NullPtr<T> = Option<Ptr<T>>;
type PointersBlock<T> = [NullPtr<T>; BLOCK_SIZE / PTR_SIZE];

/// This wrapper exists primarily to opt out of default implementation of `Record` trait
/// for arrays. We want to have a custom implementation that will write the array directly as slice.
struct DataBlock([u8; BLOCK_SIZE]);

impl Record for DataBlock {
    const SIZE: usize = BLOCK_SIZE;

    fn read(data: &[u8]) -> std::result::Result<Self, memory::Error> {
        assert!(data.len() == Self::SIZE);

        let mut result = [0; Self::SIZE];
        result.copy_from_slice(data);
        Ok(Self(result))
    }

    fn write(&self, data: &mut [u8]) -> std::result::Result<(), memory::Error> {
        assert!(data.len() == Self::SIZE);

        data.copy_from_slice(&self.0);
        Ok(())
    }
}

impl Default for DataBlock {
    fn default() -> Self {
        DataBlock([0; BLOCK_SIZE])
    }
}

impl Deref for DataBlock {
    type Target = [u8; BLOCK_SIZE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DataBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Pointers addressing blocks of file data.
///
/// This struct follows the ext4 layout for block pointers [1]. It contains limited number of
/// direct pointers, one indirect pointer and double-indirect pointer.
/// Indirect pointer points to a block containing pointers to data blocks.
/// Double-indirect pointer points to a block containing pointers to indirect pointers (2 levels of indirection).
///
/// Therefore the maximum file size is limited by the number data blocks that can be addressed by the
/// directo pointers, indirect pointers and double-indirect pointers.
///
/// The maximum file size is calculated as follows:
/// ```ignore
/// BLOCK_SIZE * (10 + POINTERS_PER_BLOCK + POINTERS_PER_BLOCK^2)
///               ^             ^                     ^
///      direct pointers  indirect pointers  double-indirect pointers
/// ```
/// 4096 * (10 + 1024 + 1024^2) = 4,229,202,560 bytes
///
/// [1]: https://github.com/torvalds/linux/blob/master/Documentation/filesystems/ext4/blockmap.rst
#[derive(Debug, Record, Clone, Default)]
struct BlockPointers {
    direct: [NullPtr<DataBlock>; DIRECT_BLOCKS],
    indirect: NullPtr<PointersBlock<DataBlock>>,
    double_indirect: NullPtr<PointersBlock<PointersBlock<DataBlock>>>,
}

impl FNode {
    fn name(&self, tx: &Transaction) -> Result<String> {
        let bytes = tx.read_bytes(self.name.0)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| e.utf8_error())
            .map_err(Error::other)
    }
}

#[derive(PartialEq, Clone, Debug, Copy, Record)]
#[repr(u8)]
pub enum NodeType {
    Directory = 1,
    File = 2,
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
    use pmem::Memory;
    use std::fs;

    macro_rules! assert_missing {
        ($fs:expr, $name:expr) => {
            let Some(ErrorKind::NotFound) = $fs.find($name).map_err(|e| e.kind()).err() else {
                panic!("{} should be missing", $name);
            };
        };
    }

    #[test]
    fn check_root() -> Result<()> {
        let fs = create_fs();

        let root = fs.get_root()?;
        assert_eq!(root.name, "/");
        assert_eq!(root.node_type, NodeType::Directory);
        Ok(())
    }

    #[test]
    fn check_find_should_return_none_if_dir_is_missing() -> Result<()> {
        let fs = create_fs();

        let node = fs.find("/foo");
        let Some(ErrorKind::NotFound) = node.map_err(|e| e.kind()).err() else {
            panic!("Node should be missing");
        };
        Ok(())
    }

    #[test]
    fn check_lookup_dir() -> Result<()> {
        let mut fs = create_fs();

        fs.create_dirs("/etc")?;
        let root = fs.get_root()?;
        let _ = fs.lookup(&root, "etc")?;
        Ok(())
    }

    #[test]
    fn check_creating_directories() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "etc")?;
        let _ = fs.lookup(&root, "etc")?;

        let node = fs.find("/etc")?;
        assert_eq!(node.name, "etc");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }

    #[test]
    fn check_creating_multiple_directories() -> Result<()> {
        let mut fs = create_fs();

        fs.create_dirs("/usr/bin")?;

        let node = fs.find("/usr/bin")?;
        assert_eq!(node.name, "bin");
        assert_eq!(node.node_type, NodeType::Directory);

        Ok(())
    }

    #[test]
    fn can_delete_directory() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "usr")?;
        fs.delete(&root, "usr")?;

        assert_missing!(fs, "/usr");
        Ok(())
    }

    #[test]
    fn can_delete_file() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        fs.create_file(&root, "swap")?;
        fs.delete(&root, "swap")?;

        assert_missing!(fs, "/swap");

        Ok(())
    }

    #[test]
    fn can_remove_several_items() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "etc")?;
        fs.create_file(&root, "swap")?;

        fs.delete(&root, "etc")?;
        assert_missing!(fs, "/etc");

        let _ = fs.lookup(&root, "swap")?;

        fs.delete(&root, "swap")?;
        assert_missing!(fs, "/swap");

        Ok(())
    }

    #[test]
    fn check_each_fs_entry_has_its_own_id() -> Result<()> {
        let mut fs = create_fs();

        fs.create_dirs("/usr/lib/bin")?;

        let usr = fs.find("/usr")?;
        let lib = fs.find("/usr/lib")?;
        let bin = fs.find("/usr/lib/bin")?;

        assert!(usr.fid > 0);
        assert!(lib.fid > 0);
        assert!(bin.fid > 0);

        assert!(usr.fid < lib.fid);
        assert!(lib.fid < bin.fid);

        let bin_ref = fs.lookup_by_id(bin.fid)?;
        assert_eq!(bin.name, bin_ref.name);

        Ok(())
    }

    #[test]
    fn create_file() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;
        let _ = fs.lookup(&root, "file.txt")?;

        let mut file = fs.open_file(&meta)?;
        let expected_content = "Hello world";
        file.write_all(expected_content.as_bytes())?;
        file.flush()?;

        let meta = fs.lookup_by_id(meta.fid)?;
        assert_eq!(meta.node_type, NodeType::File);
        assert_eq!(meta.size, 11);

        let mut file = fs.open_file(&meta)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        assert_eq!(content, expected_content);

        Ok(())
    }

    #[test]
    fn write_file_partially() -> Result<()> {
        let mut fs = create_fs();

        let root = fs.get_root()?;
        let file = fs.create_file(&root, "file.txt")?;

        write_file(&mut fs, &file, "Hello world".as_bytes(), None)?;
        write_file(&mut fs, &file, "Rust!".as_bytes(), Some(SeekFrom::Start(6)))?;

        let mut file = fs.open_file(&file)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        assert_eq!(content, "Hello Rust!");

        Ok(())
    }

    #[test]
    fn readdir_directories() -> Result<()> {
        let mut fs = create_fs();
        let root = fs.get_root()?;

        let etc = fs.create_dir(&root, "etc")?;
        let bin = fs.create_dir(&root, "bin")?;
        let swap = fs.create_file(&root, "swap")?;

        let children = fs.readdir(&root)?.collect::<Result<Vec<_>>>()?;

        // All children should be present
        assert!(children.contains(&etc), "Root should contains etc");
        assert!(children.contains(&bin), "Root should contains bin");
        assert!(children.contains(&swap), "Root should contains spawn");

        // The children should be sorted by name
        assert!(children[0] == bin);
        assert!(children[1] == etc);
        assert!(children[2] == swap);

        Ok(())
    }

    #[test]
    fn readdir_files() -> Result<()> {
        // At the moment we need to test the same logic for files, because there is
        // duplicate in implementation that should go away after migration to BTree
        let mut fs = create_fs();
        let root = fs.get_root()?;

        let etc = fs.create_file(&root, "etc")?;
        let bin = fs.create_file(&root, "bin")?;
        let swap = fs.create_file(&root, "swap")?;

        let children = fs.readdir(&root)?.collect::<Result<Vec<_>>>()?;

        // All children should be present
        assert!(children.contains(&etc), "Root should contains etc");
        assert!(children.contains(&bin), "Root should contains bin");
        assert!(children.contains(&swap), "Root should contains spawn");

        // The children should be sorted by name
        assert!(children[0] == bin);
        assert!(children[1] == etc);
        assert!(children[2] == swap);

        Ok(())
    }

    #[test]
    fn write_direct_blocks() -> Result<()> {
        let mut fs = create_fs();
        let root = fs.get_root()?;
        let file_meta = fs.create_file(&root, "file.txt")?;

        let data = [1u8; BLOCK_SIZE * 2];
        write_file(&mut fs, &file_meta, &data, None)?;
        let read_data = read_file(fs, file_meta, BLOCK_SIZE * 2, None)?;

        assert_eq!(&read_data, &data); // Check initial data

        Ok(())
    }

    #[test]
    fn write_indirect_blocks() -> Result<()> {
        let mut fs = create_fs();
        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;

        let pos = SeekFrom::Start((BLOCK_SIZE * FIRST_INDIRECT_BLOCK) as u64);

        let data = [1u8; 2 * BLOCK_SIZE];
        write_file(&mut fs, &meta, &data, Some(pos))?;
        let read_data = read_file(fs, meta, 2 * BLOCK_SIZE, Some(pos))?;

        assert_eq!(&read_data, &data);
        Ok(())
    }

    #[test]
    fn write_double_indirect_blocks() -> Result<()> {
        let mut fs = create_fs();
        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;

        let pos = SeekFrom::Start((BLOCK_SIZE * FIRST_DOUBLE_INDIRECT_BLOCK) as u64);

        let data = [1u8; 2 * BLOCK_SIZE];
        write_file(&mut fs, &meta, &data, Some(pos))?;
        let read_data = read_file(fs, meta, 2 * BLOCK_SIZE, Some(pos))?;

        assert_eq!(&read_data, &data);
        Ok(())
    }

    fn read_file(
        mut fs: Filesystem,
        meta: FileMeta,
        size: usize,
        pos: Option<SeekFrom>,
    ) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; size];
        let mut file = fs.open_file(&meta)?;
        if let Some(pos) = pos {
            file.seek(pos)?;
        }
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn write_file(
        fs: &mut Filesystem,
        meta: &FileMeta,
        data: &[u8],
        pos: Option<SeekFrom>,
    ) -> Result<()> {
        let mut file = fs.open_file(meta)?;
        if let Some(pos) = pos {
            file.seek(pos)?;
        }
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[should_panic(expected = "NoSpaceLeft")]
    fn no_space_left() {
        // Maximum file size is 17984 bytes in test environment. So in order to trigger NoSpaceLeft
        // we need to write 64Kib (1 page in PagePool) / 17984 = 4 files.
        const MAX_FILE_SIZE: usize = BLOCK_SIZE * LAST_DOUBLE_INDIRECT_BLOCK;

        let mut fs = create_fs();
        let root = fs.get_root().unwrap();

        let pos = Some(SeekFrom::Start((MAX_FILE_SIZE - 1) as u64));
        for idx in 0..4 {
            let f = fs.create_file(&root, format!("swap{}", idx)).unwrap();
            write_file(&mut fs, &f, &[1], pos).unwrap();
        }
    }

    #[test]
    fn block_calculation_utilities() {
        assert_eq!(blocks_required(0), 0);
        assert_eq!(blocks_required(1), 1);
        assert_eq!(blocks_required(BLOCK_SIZE as u64 - 1), 1);
        assert_eq!(blocks_required(BLOCK_SIZE as u64), 1);
        assert_eq!(blocks_required(BLOCK_SIZE as u64 + 1), 2);

        assert_eq!(block_idx(0), 0);
        assert_eq!(block_idx(1), 0);
        assert_eq!(block_idx(BLOCK_SIZE as u64 - 1), 0);
        assert_eq!(block_idx(BLOCK_SIZE as u64), 1);
        assert_eq!(block_idx(BLOCK_SIZE as u64 + 1), 1);
    }

    fn create_fs() -> Filesystem {
        let mem = Memory::default();
        Filesystem::allocate(mem.start())
    }

    #[cfg(not(miri))]
    mod proptests {
        use super::*;
        use pmem::page::PagePool;
        use proptest::{collection::vec, prelude::*, prop_oneof, proptest, strategy::Strategy};
        use std::fs;
        use tempfile::TempDir;

        #[derive(Debug)]
        enum WriteOperation {
            Write(Vec<u8>),
            Seek(SeekFrom),
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 1000,
                ..ProptestConfig::default()
            })]

            #[test]
            fn can_write_file(ops in vec(any_write_operation(), 0..10)) {
                let mem = Memory::new(PagePool::new(1024 * 1024));
                let mut fs = Filesystem::allocate(mem.start());

                let tmp_dir = TempDir::new().unwrap();
                let root = fs.get_root().unwrap();
                let file = fs.create_file(&root, "file.txt").unwrap();
                let shadow_file_path = tmp_dir.path().join("file.txt");

                let mut file = fs.open_file(&file).unwrap();
                let mut shadow_file = fs::File::create_new(shadow_file_path).unwrap();

                for op in ops {
                    match op {
                        WriteOperation::Write(data) => {
                            file.write_all(&data).unwrap();
                            shadow_file.write_all(&data).unwrap();
                        }
                        WriteOperation::Seek(seek) => {
                            let seek = adjust_seek(seek, &mut shadow_file)?;
                            let pos_a = file.seek(seek).unwrap();
                            let pos_b = shadow_file.seek(seek).unwrap();
                            prop_assert_eq!(pos_a, pos_b, "Seek positions differs");
                        }
                    }
                }
                file.flush().unwrap();
                shadow_file.flush().unwrap();

                let pos_a = file.seek(SeekFrom::Start(0)).unwrap();
                let pos_b = shadow_file.seek(SeekFrom::Start(0)).unwrap();
                prop_assert_eq!(pos_a, 0, "Seek is not at the beginning");
                prop_assert_eq!(pos_b, 0, "Seek is not at the beginning");

                let mut file_buf = vec![];
                file.read_to_end(&mut file_buf).unwrap();

                let mut shadow_file_buf = vec![];
                shadow_file.read_to_end(&mut shadow_file_buf).unwrap();

                prop_assert_eq!(file_buf, shadow_file_buf);
            }
        }

        fn any_write_operation() -> impl Strategy<Value = WriteOperation> {
            prop_oneof![any_write(), any_seek()]
        }

        fn any_write() -> impl Strategy<Value = WriteOperation> {
            vec(any::<u8>(), 0..10).prop_map(WriteOperation::Write)
        }

        fn any_seek() -> impl Strategy<Value = WriteOperation> {
            const MAX_SEEK: usize = BLOCK_SIZE * INDIRECT_BLOCKS;
            let seek_range = -10i64 * BLOCK_SIZE as i64..10 * BLOCK_SIZE as i64;

            let from_end = seek_range.clone().prop_map(SeekFrom::End);
            let from_start = (0u64..MAX_SEEK as u64).prop_map(SeekFrom::Start);
            let from_current = seek_range.prop_map(SeekFrom::Current);

            prop_oneof![from_end, from_start, from_current].prop_map(WriteOperation::Seek)
        }
    }

    // This function adjusts the seek operation to be valid for a given file
    //
    // Compensate for the fact that it's invalid to seek before the start of a file
    fn adjust_seek(seek: SeekFrom, file: &mut fs::File) -> io::Result<SeekFrom> {
        let seek = match seek {
            SeekFrom::Current(offset) => {
                let pos = file.stream_position()?;
                // If we're at position `pos` offsets `-pos` and before are incorrect
                let offset = offset.max(-(pos as i64));
                SeekFrom::Current(offset)
            }
            SeekFrom::End(offset) => {
                let file_len = stream_len(file)?;
                // offsets `-file_len` and before are incorrect
                let offset = offset.max(-(file_len as i64));
                SeekFrom::End(offset)
            }
            // it is correct to seek after the end of the file, so we don't need to do anything
            seek @ SeekFrom::Start(..) => seek,
        };
        Ok(seek)
    }

    fn stream_len<T: Seek>(file: &mut T) -> io::Result<u64> {
        let current_pos = file.stream_position()?;
        let end_pos = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(current_pos))?;
        Ok(end_pos)
    }

    /// Used for occasional manual debugging of tests. Just throw it to the start of the test
    /// to initialize tracing subscriber and use `RUST_LOG` env var to control verbosity.
    #[allow(unused)]
    fn init_tracing() {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

        let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
        let filter_layer = EnvFilter::from_default_env();

        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(filter_layer)
            .init();
    }
}
