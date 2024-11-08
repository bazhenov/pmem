use pmem::{
    memory::{Blob, SlicePtr, Slots, SlotsState, TxReadExt, TxWriteExt, NULL_PTR_SIZE},
    volume::{Addr, TxRead, TxWrite},
    Handle, Memory, Ptr, Record,
};
use pmem_derive::Record;
use std::{
    cmp::Ordering,
    convert::Into,
    fmt,
    io::{self, Error, ErrorKind, Read, Seek, SeekFrom, Write},
    iter::Peekable,
    path::{self, Component, Path, PathBuf},
    rc::Rc,
};
use tracing::{instrument, trace};

type Result<T> = std::io::Result<T>;
type CreateResult = std::result::Result<Handle<FNode>, Handle<FNode>>;

pub mod nfs;

const VOLUME_INFO_ADDR: Addr = 0x50;
const SLOTS_ADDR: Addr = 0x100;

pub struct Filesystem<S> {
    volume: Handle<VolumeInfo>,
    fnode_slots: Slots<FNode>,
    mem: Memory<S>,
}

#[derive(Debug, Record)]
pub struct VolumeInfo {
    root: Ptr<FNode>,
}

impl<S: TxRead> Filesystem<S> {
    pub fn open(snapshot: S) -> Result<Self> {
        let volume = snapshot.lookup(Ptr::<VolumeInfo>::from_addr(VOLUME_INFO_ADDR).unwrap())?;
        let slot_state = snapshot.lookup(Ptr::<SlotsState<_>>::from_addr(SLOTS_ADDR).unwrap())?;
        let mem = Memory::open(snapshot);
        let fnode_slots = Slots::open(slot_state.into_inner());
        Ok(Self {
            volume,
            mem,
            fnode_slots,
        })
    }

    pub fn get_root(&self) -> Result<FileMeta> {
        FileMeta::from(self.get_root_handle(), &self.mem)
    }

    /// Finds the file/directory at the given path
    ///
    /// Return [ErrorKind::NotFound] if the file/directory does not exist
    pub fn find(&self, path: impl AsRef<str>) -> Result<FileMeta> {
        FileMeta::from(self.do_lookup_file(path)?, &self.mem)
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
        FileMeta::from(child.node, &self.mem)
    }

    /// Returns the file/directory with a given inode ([FileMeta::fid])
    pub fn lookup_by_id(&self, id: u64) -> Result<FileMeta> {
        let ptr = Ptr::<FNode>::from_addr(id as Addr).ok_or(ErrorKind::NotFound)?;
        let handle = self
            .fnode_slots
            .read(&self.mem, ptr)?
            .ok_or(ErrorKind::NotFound)?;
        FileMeta::from(handle, &self.mem)
    }

    pub fn changes_from<'a, O: TxRead>(
        &'a self,
        other: &'a Filesystem<O>,
    ) -> impl Iterator<Item = Change> + 'a {
        let a = other.do_readdir(&other.get_root().unwrap());
        let b = self.do_readdir(&self.get_root().unwrap());
        Changes {
            a_fs: other,
            b_fs: self,
            stack: vec![Join::new(a, b)],
            path: Rc::new(PathBuf::from("/")),
        }
    }

    /// Opens a file for reading and writing
    pub fn open_file<'a>(&'a mut self, file: &FileMeta) -> Result<File<'a, S>> {
        let file_info = self.lookup_inode(file)?;

        Ok(File {
            pos: 0,
            fs: self,
            meta: file_info,
        })
    }

    /// Reads the contents of a directory
    ///
    /// Returns:
    /// - [ErrorKind::NotFound] if the directory does not exist
    pub fn readdir<'a>(&'a self, dir: &FileMeta) -> impl Iterator<Item = FileMeta> + 'a {
        self.do_readdir(dir)
    }

    fn get_root_handle(&self) -> Handle<FNode> {
        self.mem.lookup(self.volume.root).unwrap()
    }

    fn lookup_inode(&self, meta: &FileMeta) -> Result<Handle<FNode>> {
        let ptr = Ptr::from_addr(meta.fid as Addr).ok_or(ErrorKind::InvalidInput)?;
        self.mem.lookup(ptr).map_err(|e| e.into())
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
        let mut referent = None;
        let mut cur_node = Some(start_node);

        while let Some(claim) = cur_node {
            let node = self.mem.lookup(claim)?;
            let child_name = node.name(&self.mem)?;
            if name == child_name.as_str() {
                return Ok(FileInfoReferent { referent, node });
            }
            cur_node = node.next;
            referent = Some(node);
        }
        Err(ErrorKind::NotFound.into())
    }

    fn do_readdir(&self, dir: &FileMeta) -> ReadDir<S> {
        let parent = self.lookup_inode(dir).unwrap();

        ReadDir {
            fs: self,
            next: parent.children,
        }
    }
}

impl<S: TxWrite> Filesystem<S> {
    pub fn allocate(snapshot: S) -> Self {
        let mut mem = Memory::init(snapshot);
        let mut fnode_slots = Slots::init(&mut mem).unwrap();
        let name = mem.write_bytes("/".as_bytes()).unwrap();
        let root_entry = FNode {
            name: Str(name),
            size: 0,
            node_type: NodeType::Directory,
            children: None,
            content: BlockPointers::default(),
            next: None,
        };
        let root = fnode_slots
            .allocate_and_write(&mut mem, root_entry)
            .unwrap()
            .ptr();
        // let root = mem.write(root_entry).unwrap().ptr();
        let super_block_ptr = Ptr::<VolumeInfo>::from_addr(VOLUME_INFO_ADDR).unwrap();
        let volume = mem.write_at(super_block_ptr, VolumeInfo { root }).unwrap();
        Self {
            volume,
            mem,
            fnode_slots,
        }
    }

    pub fn delete(&mut self, dir: &FileMeta, name: impl AsRef<str>) -> Result<()> {
        let mut dir = self.lookup_inode(dir)?;

        let children = dir.children.ok_or(ErrorKind::NotFound)?;
        let file = self.find_child(children, name.as_ref())?;

        if file.node.node_type == NodeType::Directory && file.node.children.is_some() {
            return Err(ErrorKind::InvalidData.into());
        }

        let next_ptr = file.node.next;

        // Delete all data blocks
        for data_block_ptr in self.iter_blocks(&file.node.content)? {
            self.mem.reclaim(data_block_ptr)?;
        }
        // Reclaiming memory of the indirect block it it exists
        if let Some(indirect_ptr) = file.node.content.indirect {
            self.mem.reclaim(indirect_ptr)?;
        }
        // Reclaiming memory of double indirect blocks if they exist
        if let Some(double_indirect_ptr) = file.node.content.double_indirect {
            let double_indirect = self.mem.lookup(double_indirect_ptr)?.into_iter();
            for indirect_ptr in double_indirect.into_iter().flatten() {
                self.mem.reclaim(indirect_ptr)?;
            }
            self.mem.reclaim(double_indirect_ptr)?;
        }

        self.mem.reclaim(file.node.name.0)?;
        self.fnode_slots.free(&mut self.mem, file.node)?;

        if let Some(mut referent) = file.referent {
            // Child is not first in a list
            referent.next = next_ptr;
            self.mem.update(&referent)?;
        } else {
            dir.children = next_ptr;
            self.mem.update(&dir)?;
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
                self.mem.update(&dir)?;
            }
            child
        } else {
            let node = self.write_fsnode(file_name, NodeType::File)?;
            dir.children = Some(node.ptr());
            self.mem.update(&dir)?;
            node
        };

        FileMeta::from(file_info, &self.mem)
    }

    /// Creates a new directory in the given parent directory
    ///
    /// Returns:
    /// - [`ErrorKind::AlreadyExists`] if the directory already exists;
    /// - [`ErrorKind::NotFound`] if the parent directory does not exist;
    pub fn create_dir(&mut self, parent: &FileMeta, name: impl AsRef<str>) -> Result<FileMeta> {
        let mut parent = self.lookup_inode(parent)?;

        let directory_inode = if let Some(first_child) = parent.children {
            let new_child = self
                .create_child(first_child, name.as_ref(), NodeType::Directory)
                .expect("Unable to create dir")
                .ok()
                .ok_or(ErrorKind::AlreadyExists)?;
            // Update the parent directory if the new child is the first in the list
            if new_child.next == Some(first_child) {
                parent.children = Some(new_child.ptr());
                self.mem.update(&parent)?;
            }
            new_child
        } else {
            let dir_inode = self
                .write_fsnode(name.as_ref(), NodeType::Directory)
                .expect("Unable to create dir");
            parent.children = Some(dir_inode.ptr());
            self.mem.update(&parent)?;
            dir_inode
        };

        FileMeta::from(directory_inode, &self.mem)
    }

    /// Creates a directory and all its parents
    pub fn create_dirs(&mut self, name: impl AsRef<str>) -> Result<FileMeta> {
        let path = PathBuf::from(name.as_ref());
        let mut node = self.get_root_handle();
        for component in components(&path)? {
            let Component::Normal(name) = component else {
                return Err(ErrorKind::InvalidInput.into());
            };

            node = if let Some(first_child) = node.children {
                let new_child = self
                    .create_child(first_child, name.to_str().unwrap(), NodeType::Directory)?
                    .unwrap_or_else(|found_dir| found_dir);
                // Update the directory FNode if the new child is the first in the list
                if new_child.next == Some(first_child) {
                    node.children = Some(new_child.ptr());
                    self.mem.update(&node)?;
                }
                new_child
            } else {
                let new_node = self.write_fsnode(name.to_str().unwrap(), NodeType::Directory)?;
                node.children = Some(new_node.ptr());
                self.mem.update(&node)?;
                new_node
            }
        }

        FileMeta::from(node, &self.mem)
    }

    pub fn finish(self) -> Result<S> {
        let Self {
            mem,
            fnode_slots,
            volume,
        } = self;
        let mut snapshot = mem.finish()?;
        let fnode_slots = fnode_slots.finish();
        snapshot.update(&Handle::new(SLOTS_ADDR, fnode_slots))?;
        snapshot.update(&volume)?;
        Ok(snapshot)
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
            let node = self.mem.lookup(node).expect("Unable to read dir");
            let child_name = node.name(&self.mem)?;
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
        self.mem.update(&new_node)?;

        if let Some(prev_node) = prev_node {
            let mut prev_node = self.mem.lookup(prev_node)?;
            prev_node.next = Some(new_node.ptr());
            self.mem.update(&prev_node)?;
        }

        Ok(CreateResult::Ok(new_node))
    }

    fn write_fsnode(&mut self, name: &str, node_type: NodeType) -> Result<Handle<FNode>> {
        let name = self.mem.write_bytes(name.as_bytes())?;
        let entry = FNode {
            name: Str(name),
            node_type,
            size: 0,
            content: BlockPointers::default(),
            children: None,
            next: None,
        };
        self.mem.update(&self.volume)?;
        let handle = self.fnode_slots.allocate_and_write(&mut self.mem, entry)?;
        Ok(handle)
    }

    /// Returns iterator over pointers to all [`DataBlock`]: direct, indirect and double indirect
    fn iter_blocks(&self, ptrs: &BlockPointers) -> Result<impl Iterator<Item = Ptr<DataBlock>>> {
        let mut iters = vec![];

        let direct = DataBlockPtrIterator {
            block: ptrs.direct.to_vec(),
            idx: 0,
        };
        iters.push(direct);

        if let Some(indirect) = ptrs.indirect {
            let iter = DataBlockPtrIterator {
                block: self.mem.lookup(indirect)?.to_vec(),
                idx: 0,
            };
            iters.push(iter);
        }

        if let Some(double_indirect) = ptrs.double_indirect {
            let double_indirect = self.mem.lookup(double_indirect)?;
            for ptr in double_indirect.into_iter().flatten() {
                let block = self.mem.lookup(ptr)?.to_vec();
                let iter = DataBlockPtrIterator { block, idx: 0 };
                iters.push(iter);
            }
        }

        Ok(iters.into_iter().flatten())
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

pub struct File<'a, S> {
    pos: u64,
    meta: Handle<FNode>,
    fs: &'a mut Filesystem<S>,
}

impl<'a, S: TxRead> File<'a, S> {
    fn get_current_block(&self) -> Result<Handle<DataBlock>> {
        let block_no = block_idx(self.pos) as usize;
        let tx = &self.fs.mem;

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

impl<'a, S: TxWrite> File<'a, S> {
    /// Allocates given number of blocks and adds them to the file content
    #[instrument(skip(self))]
    fn allocate(&mut self, blocks: u64) -> Result<()> {
        let blocks_to_allocate = blocks;
        let mut blocks = blocks_required(self.meta.size);
        let blocks_required = blocks + blocks_to_allocate;

        let tx = &mut self.fs.mem;

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
}

/// Iteaor over the directory entries. Created by [`Filesystem::readdir`]
struct ReadDir<'a, S> {
    fs: &'a Filesystem<S>,
    next: Option<Ptr<FNode>>,
}

impl<'a, S> ReadDir<'a, S> {
    fn empty(fs: &'a Filesystem<S>) -> Self {
        Self { fs, next: None }
    }
}

impl<S: TxRead> Iterator for ReadDir<'_, S> {
    type Item = FileMeta;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.next.take() {
            let next_child = self.fs.mem.lookup(next).unwrap();
            self.next = next_child.next;
            Some(FileMeta::from(next_child, &self.fs.mem).unwrap())
        } else {
            None
        }
    }

    // TODO implement `nth()` method so skip() can be implemented in efficient way
}

#[derive(Clone)]
pub struct Change {
    #[allow(unused)]
    path: Rc<PathBuf>,
    #[allow(unused)]
    entry: FileMeta,
    #[allow(unused)]
    kind: ChangeKind,
}

impl Change {
    fn deleted(path: &Rc<PathBuf>, entry: FileMeta) -> Self {
        let path = Rc::clone(path);
        Self {
            path,
            entry,
            kind: ChangeKind::Delete,
        }
    }

    fn added(path: &Rc<PathBuf>, entry: FileMeta) -> Self {
        let path = Rc::clone(path);
        Self {
            path,
            entry,
            kind: ChangeKind::Add,
        }
    }

    pub fn into_path(self) -> PathBuf {
        let mut path = Rc::unwrap_or_clone(self.path);
        path.push(self.entry.name());
        path
    }

    pub fn kind(&self) -> ChangeKind {
        self.kind
    }

    #[cfg(test)]
    fn take_if_added(self) -> Option<Change> {
        match self.kind {
            ChangeKind::Add => Some(self),
            _ => None,
        }
    }

    #[cfg(test)]
    fn take_if_deleted(self) -> Option<Change> {
        match self.kind {
            ChangeKind::Delete => Some(self),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ChangeKind {
    Add,
    Delete,
    Update,
}

pub struct Changes<'a, A: TxRead, B: TxRead> {
    a_fs: &'a Filesystem<A>,
    b_fs: &'a Filesystem<B>,

    /// Stack of joined [`ReadDir`] iterators. Each iterator corresponds to the same directory in
    /// two filesystems.
    ///
    /// Each time we encounter a new directory, we push [`Join`] of two iterators to the stack. Thus effectively
    /// we are implementing a depth-first search of the directory tree.
    stack: Vec<Join<ReadDir<'a, A>, ReadDir<'a, B>>>,
    path: Rc<PathBuf>,
}

impl<'a, S: TxRead, O: TxRead> Changes<'a, S, O> {
    /// Adds two directories to the stack
    ///
    /// If one of the directories is None, it means that the directory does not exist in the
    /// corresponding filesystem. Syntactic empty iterator is created in this case. It will
    /// effectively emits no entries and the join loop will emit added/deleted changes
    /// depending on which filesystem has the directory.
    fn push_to_stack(&mut self, dir_a: Option<&FileMeta>, dir_b: Option<&FileMeta>) {
        assert!(
            dir_a.is_some() || dir_b.is_some(),
            "At least one directory must be present"
        );
        if dir_a.is_some() && dir_b.is_some() {
            let dir_a = dir_a.unwrap();
            let dir_b = dir_b.unwrap();
            debug_assert!(
                dir_a.name == dir_b.name,
                "Directories must have the same name. {:?} != {:?}",
                dir_a,
                dir_b,
            );
        }

        let dir_a = dir_a
            .map(|dir| self.a_fs.do_readdir(dir))
            .unwrap_or(ReadDir::empty(self.a_fs));
        let dir_b = dir_b
            .map(|dir| self.b_fs.do_readdir(dir))
            .unwrap_or(ReadDir::empty(self.b_fs));
        self.stack.push(Join::new(dir_a, dir_b));
    }
}

impl<'a, A: TxRead, B: TxRead> Iterator for Changes<'a, A, B> {
    type Item = Change;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let join = self.stack.last_mut().unwrap();

            let next_changed_item = match join.next() {
                Some(Joined::Left(a)) => {
                    // All items in B have been exhausted, all remaining items in A are deleted
                    let path = Rc::clone(&self.path);
                    if a.is_directory() {
                        self.push_to_stack(Some(&a), None);
                        Rc::make_mut(&mut self.path).push(a.name());
                    }
                    Change::deleted(&path, a)
                }

                Some(Joined::Right(b)) => {
                    // All items in A have been exhausted, all remaining items in B are added
                    let path = Rc::clone(&self.path);
                    if b.is_directory() {
                        self.push_to_stack(None, Some(&b));
                        Rc::make_mut(&mut self.path).push(b.name());
                    }
                    Change::added(&path, b)
                }

                Some(Joined::Both(a, b)) => match a.name().cmp(b.name()) {
                    Ordering::Less => {
                        // A has an item that B does not have, it means it was deleted in B
                        let path = Rc::clone(&self.path);
                        if a.is_directory() {
                            self.push_to_stack(Some(&a), None);
                            Rc::make_mut(&mut self.path).push(a.name());
                        }
                        Change::deleted(&path, a)
                    }

                    Ordering::Greater => {
                        // B has an item that A does not have, it means it was added in B
                        let path = Rc::clone(&self.path);
                        if b.is_directory() {
                            self.push_to_stack(None, Some(&b));
                            Rc::make_mut(&mut self.path).push(b.name());
                        }
                        Change::added(&path, b)
                    }

                    Ordering::Equal if a.node_type == b.node_type => {
                        // Both A and B have the same item, inspecting children if it's a directory
                        if a.node_type == NodeType::Directory {
                            self.push_to_stack(Some(&a), Some(&b));
                            Rc::make_mut(&mut self.path).push(a.name());
                        }
                        continue;
                    }

                    Ordering::Equal => {
                        // here we have a situation when both A and B has changed. A has been
                        // removed and B with the same name has been added.
                        // We can't return two changes at once here, so we returning only one change
                        // and we rely on the fact that the next call to next() will return another change
                        // because it was not removed from peekable iterator.
                        Change::deleted(&self.path, a)
                    }
                },

                None => {
                    // Both directories are exhausted, popping the stack
                    self.stack.pop();
                    Rc::make_mut(&mut self.path).pop();
                    continue;
                }
            };
            return Some(next_changed_item);
        }
        None
    }
}

fn make_sure_data_block_exists(
    mem: &mut Memory<impl TxWrite>,
    ptr: &mut Option<Ptr<DataBlock>>,
) -> Result<Handle<DataBlock>> {
    if let Some(ptr) = ptr {
        Ok(mem.lookup(*ptr)?)
    } else {
        let data_block = mem.write(DataBlock::default())?;
        *ptr = Some(data_block.ptr());
        Ok(data_block)
    }
}

fn make_sure_ptr_block_exists<T: Record>(
    mem: &mut Memory<impl TxWrite>,
    ptr: &mut Option<Ptr<PointersBlock<T>>>,
) -> Result<Handle<PointersBlock<T>>> {
    if let Some(ptr) = ptr {
        Ok(mem.lookup(*ptr)?)
    } else {
        trace!("Allocating double indirect pointers block");
        let indirect_block = mem.write::<PointersBlock<T>>([None; POINTERS_PER_BLOCK])?;
        *ptr = Some(indirect_block.ptr());
        Ok(indirect_block)
    }
}

fn lookup_step<T: Record>(
    mem: &Memory<impl TxRead>,
    block_ptr: NullPtr<PointersBlock<T>>,
    idx: usize,
) -> NullPtr<T> {
    let block_ptr = block_ptr.expect("Block pointer missing");
    let block = mem.lookup(block_ptr).expect("Unable to read block");
    block[idx]
}

impl<'a, S: TxRead> Read for File<'a, S> {
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

        // TODO: profiling results show that get_current_block() is bottleneck of this function
        //       in order to improve performance we need to read multiple blocks at once
        let block = self.get_current_block()?;
        buf[..len].copy_from_slice(&block[offset..offset + len]);

        self.pos += len as u64;
        Ok(len)
    }
}

impl<'a, S: TxWrite> Write for File<'a, S> {
    #[instrument(level = "trace", skip(self, buf), fields(buf.len = buf.len(), pos=self.pos), ret)]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // user can seek after the end of file and write there, so we use self.pos here
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
        self.fs.mem.update(&block)?;

        self.pos += len as u64;
        self.meta.size = self.meta.size.max(self.pos);

        self.fs.mem.update(&self.meta)?;

        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a, S: TxRead> Seek for File<'a, S> {
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
    fn from(file_info: Handle<FNode>, mem: &Memory<impl TxRead>) -> Result<Self> {
        let name = file_info.name(mem)?;
        Ok(FileMeta {
            name,
            fid: file_info.ptr().unwrap_addr(),
            size: file_info.size,
            node_type: file_info.node_type,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    #[allow(unused)]
    fn is_directory(&self) -> bool {
        self.node_type == NodeType::Directory
    }

    #[allow(unused)]
    fn is_file(&self) -> bool {
        self.node_type == NodeType::File
    }
}

impl Ord for FileMeta {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name.cmp(&other.name)
    }
}

impl PartialOrd for FileMeta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for FileMeta {}

#[derive(Clone, Debug, Record)]
struct FNode {
    name: Str,
    node_type: NodeType,
    size: u64,
    children: Option<Ptr<FNode>>,
    content: BlockPointers,
    next: Option<Ptr<FNode>>,
}

impl FNode {
    fn name(&self, mem: &Memory<impl TxRead>) -> Result<String> {
        let bytes = mem.read_bytes(self.name.0).unwrap();
        String::from_utf8(bytes.to_vec())
            .map_err(|e| e.utf8_error())
            .map_err(Error::other)
    }
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
const POINTERS_PER_BLOCK: usize = BLOCK_SIZE / NULL_PTR_SIZE;

const DIRECT_BLOCKS: usize = 10;
const INDIRECT_BLOCKS: usize = POINTERS_PER_BLOCK;
const DOUBLE_INDIRECT_BLOCKS: usize = INDIRECT_BLOCKS * INDIRECT_BLOCKS;

const LAST_DIRECT_BLOCK: usize = DIRECT_BLOCKS - 1;

const FIRST_INDIRECT_BLOCK: usize = LAST_DIRECT_BLOCK + 1;
const LAST_INDIRECT_BLOCK: usize = FIRST_INDIRECT_BLOCK + INDIRECT_BLOCKS - 1;

const FIRST_DOUBLE_INDIRECT_BLOCK: usize = LAST_INDIRECT_BLOCK + 1;
const LAST_DOUBLE_INDIRECT_BLOCK: usize = FIRST_DOUBLE_INDIRECT_BLOCK + DOUBLE_INDIRECT_BLOCKS - 1;

/// Simple type alias to minimize the amount of angle brackets in the code
type NullPtr<T> = Option<Ptr<T>>;
type PointersBlock<T> = [NullPtr<T>; POINTERS_PER_BLOCK];

struct DataBlockPtrIterator {
    block: Vec<NullPtr<DataBlock>>,
    idx: usize,
}

impl Iterator for DataBlockPtrIterator {
    type Item = Ptr<DataBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.block.len() {
            let item = self.block[self.idx];
            self.idx += 1;
            item
        } else {
            None
        }
    }
}

type DataBlock = Blob<BLOCK_SIZE>;

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

/// Join two sorted iterators into a single iterator of type [`Joined<T>`] on the same elements.
///
/// The input iterators must be sorted in ascending order. Emits [`Joined::Left`], [`Joined::Right`]
/// or [`Joined::Both`] depending on which iterarots has the element present.
struct Join<A: Iterator, B: Iterator>(Peekable<A>, Peekable<B>);

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Join<A, B> {
    fn new(a: A, b: B) -> Self {
        Self(a.peekable(), b.peekable())
    }
}

impl<T: Ord, A: Iterator<Item = T>, B: Iterator<Item = T>> Iterator for Join<A, B> {
    type Item = Joined<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let Join(a_iter, b_iter) = self;
        match (a_iter.peek(), b_iter.peek()) {
            (Some(a), Some(b)) => match a.cmp(b) {
                Ordering::Less => Some(Joined::Left(a_iter.next().unwrap())),
                Ordering::Equal => {
                    Some(Joined::Both(a_iter.next().unwrap(), b_iter.next().unwrap()))
                }
                Ordering::Greater => Some(Joined::Right(b_iter.next().unwrap())),
            },
            (Some(_), None) => Some(Joined::Left(a_iter.next().unwrap())),
            (None, Some(_)) => Some(Joined::Right(b_iter.next().unwrap())),
            (None, None) => None,
        }
    }
}

#[derive(Debug, PartialEq)]
enum Joined<T: PartialEq> {
    Left(T),
    Right(T),
    Both(T, T),
}

/// Helper structure that represents the filesystem as a tree in a form of [`Debug`] format
/// (eg. `{:?}` in print! macro will print the tree structure of the filesystem)
pub struct FsTree<'a, S>(pub &'a Filesystem<S>);

impl<'a, S: TxRead> fmt::Debug for FsTree<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn do_print(
            f: &mut fmt::Formatter<'_>,
            fs: &Filesystem<impl TxRead>,
            dir: &FileMeta,
            is_last_vec: &mut Vec<bool>,
        ) -> fmt::Result {
            let mut children = fs.readdir(dir).peekable();
            while let Some(child) = children.next() {
                let is_last = children.peek().is_none();

                let node_type = match child.node_type {
                    NodeType::Directory => "/",
                    NodeType::File => "",
                };
                let marker = if is_last { "└─" } else { "├─" };
                for is_last in &*is_last_vec {
                    if *is_last {
                        write!(f, "   ")?;
                    } else {
                        write!(f, " │ ")?;
                    }
                }
                writeln!(f, " {} {}{}", marker, child.name, node_type)?;
                if child.is_directory() {
                    is_last_vec.push(is_last);
                    do_print(f, fs, &child, is_last_vec)?;
                    is_last_vec.pop();
                }
            }
            Ok(())
        }

        writeln!(f, " /")?;
        let fs = self.0;
        do_print(f, fs, &fs.get_root().unwrap(), &mut vec![])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fmt::Debug;
    use pmem::volume::{Transaction, Volume, PAGE_SIZE};
    use rand::{rngs::SmallRng, RngCore, SeedableRng};
    use std::{cmp::max, collections::HashSet, fs};

    // Maximum file size is 5184 bytes in test environment.
    const MAX_FILE_SIZE: usize = BLOCK_SIZE * LAST_DOUBLE_INDIRECT_BLOCK;

    macro_rules! assert_not_exists {
        ($fs:expr, $name:expr) => {
            match $fs.find($name).map_err(|e| e.kind()) {
                Ok(_) => panic!("{} should not exists", $name),
                Err(ErrorKind::NotFound) => (),
                e @ Err(..) => panic!("find() failed: {:?}", e),
            }
        };
    }

    macro_rules! assert_exists {
        ($fs:expr, $name:expr) => {
            if let Err(e) = $fs.find($name) {
                match e.kind() {
                    ErrorKind::NotFound => panic!("{} should exists", $name),
                    _ => panic!("find() failed: {:?}", e),
                }
            }
        };
    }

    #[test]
    fn check_root() -> Result<()> {
        let (fs, _) = create_fs();

        let root = fs.get_root()?;
        assert_eq!(root.name(), "/");
        assert!(root.is_directory());
        Ok(())
    }

    #[test]
    fn check_find_should_return_none_if_dir_is_missing() -> Result<()> {
        let (fs, _) = create_fs();

        let node = fs.find("/foo");
        let Some(ErrorKind::NotFound) = node.map_err(|e| e.kind()).err() else {
            panic!("Node should be missing");
        };
        Ok(())
    }

    #[test]
    fn check_lookup_dir() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/etc")?;
        assert_exists!(fs, "/etc");
        Ok(())
    }

    #[test]
    fn can_create_directories() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "etc")?;
        assert_exists!(fs, "/etc");

        let node = fs.find("/etc")?;
        assert_eq!(node.name(), "etc");
        assert!(node.is_directory());

        Ok(())
    }

    #[test]
    fn check_creating_multiple_directories() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/usr/bin")?;

        let node = fs.find("/usr/bin")?;
        assert_eq!(node.name(), "bin");
        assert!(node.is_directory());

        Ok(())
    }

    #[test]
    fn check_creating_already_existing_directories() -> Result<()> {
        let (mut fs, _) = create_fs();

        fs.create_dirs("/usr/bin")?;
        fs.create_dirs("/usr/bin")?;
        fs.create_dirs("/")?;

        Ok(())
    }

    #[test]
    fn can_delete_directory() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_dir(&root, "usr")?;
        fs.delete(&root, "usr")?;

        assert_not_exists!(fs, "/usr");
        Ok(())
    }

    #[test]
    fn will_not_delete_not_empty_directory() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_dirs("/usr/bin")?;
        assert!(fs.delete(&root, "usr").is_err());

        assert_exists!(fs, "/usr");
        Ok(())
    }

    #[test]
    fn can_delete_file() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        fs.create_file(&root, "swap")?;
        fs.delete(&root, "swap")?;

        assert_not_exists!(fs, "/swap");

        Ok(())
    }

    #[test]
    fn can_remove_several_items() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        mkdirs(&mut fs, &["/etc/"]);
        mkfiles(&mut fs, &["/swap"]);

        fs.delete(&root, "etc")?;
        assert_not_exists!(fs, "/etc");
        assert_exists!(fs, "/swap");

        fs.delete(&root, "swap")?;
        assert_not_exists!(fs, "/swap");

        Ok(())
    }

    #[test]
    fn check_each_fs_entry_has_its_own_id() -> Result<()> {
        let (mut fs, _) = create_fs();

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
        assert_eq!(bin.name(), bin_ref.name());

        Ok(())
    }

    #[test]
    fn create_file() -> Result<()> {
        let (mut fs, _) = create_fs();

        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;
        assert_exists!(fs, "/file.txt");

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
        let (mut fs, _) = create_fs();

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
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;

        let etc = fs.create_dir(&root, "etc")?;
        let bin = fs.create_dir(&root, "bin")?;
        let swap = fs.create_file(&root, "swap")?;

        let children = fs.readdir(&root).collect::<Vec<_>>();

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
    fn read_removed_directory() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;

        let etc = fs.create_dir(&root, "etc")?;
        fs.delete(&root, "etc")?;

        let removed_etc = fs.lookup_by_id(etc.fid);
        match removed_etc {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => panic!("Expected NotFound error: found {}", e),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn removing_removed_directory() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;

        fs.create_dir(&root, "etc")?;
        fs.delete(&root, "etc")?;

        let removed_etc = fs.delete(&root, "etc");
        match removed_etc {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => panic!("Expected NotFound error: found {}", e),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn readdir_files() -> Result<()> {
        // At the moment we need to test the same logic for files, because there is
        // duplicate in implementation that should go away after migration to BTree
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;

        let etc = fs.create_file(&root, "etc")?;
        let bin = fs.create_file(&root, "bin")?;
        let swap = fs.create_file(&root, "swap")?;

        let children = fs.readdir(&root).collect::<Vec<_>>();

        // All children should be present
        assert!(children.contains(&etc), "Root should contains etc");
        assert!(children.contains(&bin), "Root should contains bin");
        assert!(children.contains(&swap), "Root should contains spawn");

        // The children should be sorted by name
        assert_uniq_and_sorted(children.iter().map(|meta| meta.name()));

        Ok(())
    }

    #[test]
    fn detect_changes() -> Result<()> {
        let (mut fs_a, mut mem) = create_fs();
        mkdirs(&mut fs_a, &["/etc"]);
        mem.commit(fs_a.finish()?).unwrap();
        let tx_a = mem.start();

        let mut fs_b = Filesystem::open(mem.start())?;
        fs_b.delete(&fs_b.get_root()?, "etc")?;
        mkdirs(&mut fs_b, &["/bin"]);
        mem.commit(fs_b.finish()?).unwrap();
        let tx_b = mem.start();

        let fs_a = Filesystem::open(tx_a)?;
        let fs_b = Filesystem::open(tx_b)?;

        let (added, deleted) = fs_changes(&fs_a, &fs_b);

        assert_eq!(added.len(), 1);
        assert_eq!(added[0].entry.name(), "bin");
        assert!(added[0].entry.is_directory());

        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0].entry.name(), "etc");
        assert!(deleted[0].entry.is_directory());

        Ok(())
    }

    #[test]
    fn write_direct_blocks() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;
        let file_meta = fs.create_file(&root, "file.txt")?;

        let data = [1u8; BLOCK_SIZE * 2];
        write_file(&mut fs, &file_meta, &data, None)?;
        let read_data = read_file(&mut fs, &file_meta, BLOCK_SIZE * 2, None)?;

        assert_eq!(&read_data, &data);

        Ok(())
    }

    #[test]
    fn write_indirect_blocks() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;

        let pos = SeekFrom::Start((BLOCK_SIZE * FIRST_INDIRECT_BLOCK) as u64);

        let data = [1u8; 2 * BLOCK_SIZE];
        write_file(&mut fs, &meta, &data, Some(pos))?;
        let read_data = read_file(&mut fs, &meta, 2 * BLOCK_SIZE, Some(pos))?;

        assert_eq!(&read_data, &data);
        Ok(())
    }

    #[test]
    fn write_double_indirect_blocks() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "file.txt")?;

        let pos = SeekFrom::Start((BLOCK_SIZE * FIRST_DOUBLE_INDIRECT_BLOCK) as u64);

        let data = [1u8; 2 * BLOCK_SIZE];
        write_file(&mut fs, &meta, &data, Some(pos))?;
        let read_data = read_file(&mut fs, &meta, 2 * BLOCK_SIZE, Some(pos))?;

        assert_eq!(&read_data, &data);
        Ok(())
    }

    fn read_file(
        fs: &mut Filesystem<impl TxRead>,
        meta: &FileMeta,
        size: usize,
        pos: Option<SeekFrom>,
    ) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; size];
        let mut file = fs.open_file(meta)?;
        if let Some(pos) = pos {
            file.seek(pos)?;
        }
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn write_file(
        fs: &mut Filesystem<impl TxWrite>,
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
    #[should_panic(expected = "NoSpaceLeft")]
    fn no_space_left() {
        let (mut fs, _) = create_fs_with_size(PAGE_SIZE);
        let root = fs.get_root().unwrap();

        let pos = Some(SeekFrom::Start((MAX_FILE_SIZE - 1) as u64));
        // In order to trigger NoSpaceLeft we need to write 64Kib (1 page in Volume) / MAX_FILE_SIZE (5184) = 13 files.
        for idx in 0..13 {
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

    #[test]
    fn check_join_iterator() {
        use Joined::*;

        let a = vec![1, 4, 5];
        let b = vec![3, 4, 6];

        let entries = Join::new(a.into_iter(), b.into_iter()).collect::<Vec<_>>();

        assert_eq!(
            entries,
            vec![Left(1), Right(3), Both(4, 4), Left(5), Right(6),]
        );
    }

    #[test]
    fn check_intermediate_paths() {
        let path = PathBuf::from("/a/b/c/");

        let paths = all_intermediate_paths(&path).collect::<HashSet<_>>();
        let mut expected = HashSet::new();
        expected.insert(PathBuf::from("/a"));
        expected.insert(PathBuf::from("/a/b"));
        expected.insert(PathBuf::from("/a/b/c"));
        assert_eq!(paths, expected);
    }

    #[test]
    fn check_block_pointer_iterator() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;
        let file_meta = fs.create_file(&root, "test_file.txt")?;

        let file_size = MAX_FILE_SIZE;
        let data = vec![42u8; file_size];
        write_file(&mut fs, &file_meta, &data, None)?;

        let file_info = fs.lookup_inode(&file_meta)?;
        let block_count = fs.iter_blocks(&file_info.content)?.count();

        let expected_blocks = file_size / BLOCK_SIZE;
        assert_eq!(block_count, expected_blocks);

        Ok(())
    }

    #[test]
    fn check_large_write() -> Result<()> {
        let (mut fs, _) = create_fs();
        let root = fs.get_root()?;
        let mut rng = SmallRng::from_entropy();

        let file_meta = fs.create_file(&root, "test_file.txt")?;
        let file_size = MAX_FILE_SIZE;
        let mut data = vec![0u8; file_size];
        rng.fill_bytes(&mut data[..]);

        let mut file = fs.open_file(&file_meta)?;
        for chunk in data.chunks(512) {
            file.write_all(chunk)?;
        }
        file.flush()?;

        let content = read_file(&mut fs, &file_meta, file_size, None)?;
        assert_eq!(content, data);
        Ok(())
    }

    #[test]
    fn check_fs_reuses_freed_space() -> Result<()> {
        let (mut fs, _) = create_fs_with_size(3 * PAGE_SIZE);
        let root = fs.get_root()?;

        // Calculate the number of times we need to write to fill the volume
        let writes_to_fill = max(1, PAGE_SIZE / MAX_FILE_SIZE);

        let content = "A".repeat(MAX_FILE_SIZE);

        // Write file severals times to fill the filesystem
        for _ in 0..writes_to_fill {
            let file = fs.create_file(&root, "file.txt")?;
            write_file(&mut fs, &file, content.as_bytes(), None)?;
            fs.delete(&root, "file.txt")?;
        }

        // If in previous step space was not reused, this write will fail
        let file2 = fs.create_file(&root, "file2.txt")?;
        let content2 = "B".repeat(MAX_FILE_SIZE);
        write_file(&mut fs, &file2, content2.as_bytes(), None)?;

        // Verify that file2 exists and has correct content
        let read_content = read_file(&mut fs, &file2, content2.len(), None)?;
        assert_eq!(
            read_content,
            content2.as_bytes(),
            "file2.txt content is incorrect"
        );

        Ok(())
    }

    #[test]
    fn check_fs_reuses_space_from_fnodes() -> Result<()> {
        let (mut fs, _) = create_fs_with_size(3 * PAGE_SIZE);
        let root = fs.get_root()?;

        let very_long_name = "a".repeat(128);

        // If some space from file nodes is not reused, this test will fail
        for _ in 0..10000 {
            fs.create_file(&root, &very_long_name)?;
            fs.delete(&root, &very_long_name)?;

            fs.create_dir(&root, &very_long_name)?;
            fs.delete(&root, &very_long_name)?;
        }

        Ok(())
    }

    /// Iterates over all intermediate paths of the given path excluding the root.
    ///
    /// For example, for the path `/a/b/c` it will return (in no particular order):
    /// - `/a/b/c`
    /// - `/a/b`
    /// - `/a`
    fn all_intermediate_paths(path: impl AsRef<Path>) -> impl Iterator<Item = PathBuf> {
        let path = path.as_ref();

        // On windows we can't use `path.is_absolute()` because only paths starting with a drive
        // letter are considered absolute.
        #[cfg(unix)]
        debug_assert!(path.is_absolute(), "Path must be absolute");

        let mut next = Some(path.to_path_buf());
        std::iter::from_fn(move || {
            let path = next.take().filter(|p| p != Path::new("/"))?;
            next = path.parent().map(Path::to_path_buf);
            Some(path)
        })
    }

    fn assert_uniq_and_sorted<T: Ord + Debug>(mut iter: impl Iterator<Item = T>) {
        let mut prev = iter.next();
        while let (Some(a), Some(b)) = (prev, iter.next()) {
            assert!(
                a < b,
                "Items are not sorted (a >= b)\n A: {:?}\n B: {:?}",
                a,
                b
            );
            prev = Some(b);
        }
    }

    /// Returns the changes to the `target` full filesystem from the `base` filesystem.
    ///
    /// The changes are returned as a tuple of `(added, deleted)` files.
    fn fs_changes<S: TxRead>(
        base: &Filesystem<S>,
        target: &Filesystem<S>,
    ) -> (Vec<Change>, Vec<Change>) {
        let changes = target.changes_from(base).collect::<Vec<_>>();
        let deleted = changes
            .clone()
            .into_iter()
            .filter_map(Change::take_if_deleted)
            .collect::<Vec<_>>();
        let added = changes
            .into_iter()
            .filter_map(Change::take_if_added)
            .collect::<Vec<_>>();
        (added, deleted)
    }

    fn mkdirs(fs: &mut Filesystem<impl TxWrite>, directories: &[&str]) {
        for path in directories {
            fs.create_dirs(path).unwrap();
        }
    }

    fn mkfiles(fs: &mut Filesystem<impl TxWrite>, files: &[&str]) {
        for path in files {
            let parent = PathBuf::from(path);
            let file_name = parent.file_name().unwrap().to_str().unwrap();
            let dir_name = parent.parent().unwrap().to_str().unwrap();
            let dir_meta = fs.create_dirs(dir_name).unwrap();
            fs.create_file(&dir_meta, file_name).unwrap();
        }
    }

    fn create_fs_with_size(size: usize) -> (Filesystem<Transaction>, Volume) {
        let volume = Volume::with_capacity(size);
        (Filesystem::allocate(volume.start()), volume)
    }

    fn create_fs() -> (Filesystem<Transaction>, Volume) {
        let volume = Volume::new_in_memory(3);
        (Filesystem::allocate(volume.start()), volume)
    }

    /// A filesystem action that can be applied to a filesystem
    ///
    /// See [`apply_fs_actions`] for more information
    #[derive(Debug, Clone)]
    enum FsAction {
        CreateFile(PathBuf),
        CreateDirectory(PathBuf),
    }

    impl FsAction {
        fn path(&self) -> &PathBuf {
            match self {
                FsAction::CreateFile(path) => path,
                FsAction::CreateDirectory(path) => path,
            }
        }
    }

    fn apply_fs_actions<'a>(
        fs: &mut Filesystem<impl TxWrite>,
        actions: impl Iterator<Item = &'a FsAction>,
    ) -> Result<()> {
        for action in actions {
            match action {
                FsAction::CreateDirectory(path) => {
                    fs.create_dirs(path.to_str().unwrap())?;
                }
                FsAction::CreateFile(path) => {
                    let file_name = path.file_name().unwrap().to_str().unwrap();
                    let dir_name = path.parent().unwrap().to_str().unwrap();
                    fs.create_dirs(dir_name)?;
                    let dir = fs.find(dir_name)?;
                    fs.create_file(&dir, file_name)?;
                }
            }
        }

        Ok(())
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

    mod proptests {
        use super::*;
        use pmem::volume::Volume;
        use prop::collection::hash_set;
        use proptest::{collection::vec, prelude::*, prop_oneof, proptest, strategy::Strategy};
        use std::{fs, ops::Range};
        use tempfile::TempDir;

        #[derive(Debug)]
        enum WriteOperation {
            Write(Vec<u8>),
            Seek(SeekFrom),
        }

        /// Random strings used to generate file and directory names
        const NATO_ALPHABET: [&str; 26] = [
            "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliett", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango", "uniform", "victor", "whiskey", "x-ray", "yankee", "zulu",
        ];

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 1000,
                ..ProptestConfig::default()
            })]

            #[test]
            fn can_write_file(ops in vec(any_write_operation(), 0..10)) {
                let volume = Volume::with_capacity(1024 * 1024);
                let mut fs = Filesystem::allocate(volume.start());

                let tmp_dir = TempDir::new()?;
                let root = fs.get_root()?;
                let file = fs.create_file(&root, "file.txt")?;
                let shadow_file_path = tmp_dir.path().join("file.txt");

                let mut file = fs.open_file(&file)?;
                let mut shadow_file = fs::File::create_new(shadow_file_path)?;

                for op in ops {
                    match op {
                        WriteOperation::Write(data) => {
                            file.write_all(&data)?;
                            shadow_file.write_all(&data)?;
                        }
                        WriteOperation::Seek(seek) => {
                            let seek = adjust_seek(seek, &mut shadow_file)?;
                            let pos_a = file.seek(seek)?;
                            let pos_b = shadow_file.seek(seek)?;
                            prop_assert_eq!(pos_a, pos_b, "Seek positions differs");
                        }
                    }
                }
                file.flush()?;
                shadow_file.flush()?;

                let pos_a = file.seek(SeekFrom::Start(0))?;
                let pos_b = shadow_file.seek(SeekFrom::Start(0))?;
                prop_assert_eq!(pos_a, 0, "Seek is not at the beginning");
                prop_assert_eq!(pos_b, 0, "Seek is not at the beginning");

                let mut file_buf = vec![];
                file.read_to_end(&mut file_buf)?;

                let mut shadow_file_buf = vec![];
                shadow_file.read_to_end(&mut shadow_file_buf)?;

                prop_assert_eq!(file_buf, shadow_file_buf);
            }

            #[test]
            fn created_directory_can_be_found(paths in vec(any_path(None), 1..10)) {
                let (mut fs, _) = create_fs();

                for path in paths.iter() {
                    let path = path.to_str().unwrap();
                    fs.create_dirs(path)?;
                    fs.find(path)?;
                }
            }

            #[test]
            fn created_file_can_be_found(paths in hash_set(any_path(None), 1..10)) {
                let (mut fs, _) = create_fs();

                for path in paths.into_iter() {
                    // Adding file extension to make sure file names will not collide with directory names
                    let path = path.with_extension("txt");

                    let dir_path = path.parent().unwrap().to_str().unwrap();
                    let file_name = path.file_name().unwrap().to_str().unwrap();

                    let dir = fs.create_dirs(dir_path)?;
                    fs.create_file(&dir, file_name)?;
                    fs.find(path.to_str().unwrap())?;
                }
            }

            #[test]
            fn can_remove_file(file_names in hash_set(any_file_name(None), 10)) {
                let (mut fs, _) = create_fs();

                // Creating files in a different order than removing them
                let mut sorted_files_names = file_names.iter().cloned().collect::<Vec<_>>();
                sorted_files_names.sort();

                let root = fs.get_root()?;
                for (idx, file_name) in sorted_files_names.iter().enumerate() {
                    // Creating directories and files to make sure both are removed correctly
                    if idx % 2 == 0 {
                        fs.create_file(&root, file_name)?;
                    } else {
                        fs.create_dir(&root, file_name)?;
                    }
                }

                for file_name in &file_names {
                    fs.delete(&root, file_name)?;
                }

                prop_assert_eq!(fs.readdir(&root).count(), 0)
            }

            #[test]
            fn can_detect_changes_on_fs(
                a in any_fs_actions_uniq(Some("a-"), 1..10),
                b in any_fs_actions_uniq(Some("b-"), 1..10),
                common in any_fs_actions_uniq(None, 1..10)
            ) {
                // We have 3 disjoint sets of actions:
                // - A and B - actions that are applied to FS A and FS B respectively
                // - common - actions that are applied to both FS

                let (mut fs_a, _) = create_fs();
                let (mut fs_b, _) = create_fs();

                apply_fs_actions(&mut fs_a, a.iter().chain(common.iter()))?;
                apply_fs_actions(&mut fs_b, b.iter().chain(common.iter()))?;

                let (added, deleted) = fs_changes(&fs_a, &fs_b);

                let added = added.into_iter().map(Change::into_path).collect::<HashSet<_>>();
                let added_expected = b.iter()
                    .map(FsAction::path)
                    .flat_map(all_intermediate_paths)
                    .collect::<HashSet<_>>();

                prop_assert_eq!(added, added_expected,
                    "\nFS A:\n {:?}\nFS B:\n {:?}",
                    FsTree(&fs_a),
                    FsTree(&fs_b));

                let deleted = deleted.into_iter().map(Change::into_path).collect::<HashSet<_>>();
                let deleted_expected = a.iter()
                    .map(FsAction::path)
                    .flat_map(all_intermediate_paths)
                    .collect::<HashSet<_>>();

                prop_assert_eq!(deleted, deleted_expected,
                    "\nFS A:\n {:?}\nFS B:\n {:?}",
                    FsTree(&fs_a),
                    FsTree(&fs_b));

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

        fn any_action_type(path: PathBuf) -> impl Strategy<Value = FsAction> {
            prop_oneof![
                Just(FsAction::CreateFile(path.with_extension("txt"))),
                Just(FsAction::CreateDirectory(path.clone())),
            ]
        }

        fn any_fs_actions_uniq(
            prefix: Option<&str>,
            range: Range<usize>,
        ) -> impl Strategy<Value = Vec<FsAction>> + '_ {
            hash_set(any_path(prefix), range).prop_flat_map(any_action_each)
        }

        fn any_action_each(
            paths: impl IntoIterator<Item = PathBuf>,
        ) -> impl Strategy<Value = Vec<FsAction>> {
            paths.into_iter().map(any_action_type).collect::<Vec<_>>()
        }

        fn any_path(prefix: Option<&str>) -> impl Strategy<Value = PathBuf> + '_ {
            vec(any_file_name(prefix), 3)
                .prop_map(|c| format!("/{}", c.join("/")))
                .prop_map(PathBuf::from)
        }

        fn any_file_name(prefix: Option<&str>) -> impl Strategy<Value = String> + '_ {
            (0..NATO_ALPHABET.len())
                .prop_map(|i| NATO_ALPHABET[i])
                .prop_map(move |s| format!("{}{}", prefix.unwrap_or_default(), s))
        }
    }
}
