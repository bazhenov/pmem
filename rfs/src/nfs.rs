//! Implementation of RFS filesystem adapter for a NFS protocol.
//!
//! To test use following command:
//! ```ignore
//! $ mkdir mnt
//! $ mount -t nfs -o nolocks,vers=3,tcp,port=11111,mountport=11111,soft 127.0.0.1:/ mnt/
//! ```
use crate::{FileMeta, Filesystem, NodeType};
use async_trait::async_trait;
use nfsserve::{
    nfs::{
        fattr3, fileid3, filename3, ftype3, nfspath3, nfsstat3, nfsstring, nfstime3, sattr3,
        specdata3,
    },
    vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities},
};
use pmem::volume::{Transaction, Volume};
use std::{
    io::{self, Read, Seek, SeekFrom, Write},
    mem,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tokio::sync::Mutex;
use tracing::{instrument, warn};

pub struct RFS {
    state: Arc<Mutex<RfsState>>,
    root_id: fileid3,
}

pub struct RfsState {
    fs: Filesystem<Transaction>,
}

impl RfsState {
    pub async fn commit(&mut self, volume: &mut Volume) {
        let mut sw_fs = Filesystem::open(volume.start()).unwrap();
        mem::swap(&mut self.fs, &mut sw_fs);
        volume.commit(sw_fs.finish().unwrap()).unwrap();
        let new_tx = volume.start();
        self.fs = Filesystem::open(new_tx).unwrap();
    }
}

impl Deref for RfsState {
    type Target = Filesystem<Transaction>;

    fn deref(&self) -> &Self::Target {
        &self.fs
    }
}

impl DerefMut for RfsState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.fs
    }
}

impl RFS {
    pub fn new(snapshot: Transaction) -> RFS {
        let fs = Filesystem::open(snapshot).unwrap();
        let root = fs.get_root().unwrap();
        let state = RfsState { fs };
        RFS {
            state: Arc::new(Mutex::new(state)),
            root_id: root.fid,
        }
    }

    pub fn state_handle(&self) -> Arc<Mutex<RfsState>> {
        self.state.clone()
    }
}

#[async_trait]
#[allow(clippy::blocks_in_conditions)]
impl NFSFileSystem for RFS {
    #[instrument(skip(self))]
    fn root_dir(&self) -> fileid3 {
        self.root_id
    }

    fn capabilities(&self) -> VFSCapabilities {
        VFSCapabilities::ReadWrite
    }

    #[instrument( skip(self, data), fields(data.len = data.len()), err(Debug, level = "warn"))]
    async fn write(&self, id: fileid3, offset: u64, data: &[u8]) -> Result<fattr3, nfsstat3> {
        let mut fs = self.state.lock().await;

        let meta = fs.lookup_by_id(id).map_err(io_to_nfs_error)?;
        let mut file = fs.open_file(&meta).map_err(io_to_nfs_error)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))
                .map_err(io_to_nfs_error)?;
        }
        file.write_all(data).map_err(io_to_nfs_error)?;
        file.flush().map_err(io_to_nfs_error)?;
        let new_meta = fs.lookup_by_id(id).map_err(io_to_nfs_error)?;
        Ok(create_fattr(&new_meta))
    }

    #[instrument(skip(self, _attr), err(Debug, level = "warn"))]
    async fn create(
        &self,
        dirid: fileid3,
        filename: &filename3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        let mut fs = self.state.lock().await;
        let dir = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;
        let new_file = fs
            .create_file(&dir, to_string(filename))
            .map_err(io_to_nfs_error)?;
        Ok((new_file.fid, create_fattr(&new_file)))
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn create_exclusive(
        &self,
        dirid: fileid3,
        filename: &filename3,
    ) -> Result<fileid3, nfsstat3> {
        let mut fs = self.state.lock().await;
        let dir = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;
        let new_file = fs
            .create_file(&dir, to_string(filename))
            .map_err(io_to_nfs_error)?;
        Ok(new_file.fid)
    }

    async fn lookup(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        let fs = self.state.lock().await;
        let filename = to_string(filename);
        let dir = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;
        let file = fs.lookup(&dir, &filename);
        if let Err(io_err) = &file {
            if io_err.kind() != io::ErrorKind::NotFound {
                warn!(error = ?io_err, dirid = dirid, filename = filename)
            }
        }
        Ok(file.map_err(io_to_nfs_error)?.fid)
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        let fs = self.state.lock().await;
        let entry = fs.lookup_by_id(id).map_err(io_to_nfs_error)?;
        Ok(create_fattr(&entry))
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn setattr(&self, id: fileid3, _setattr: sattr3) -> Result<fattr3, nfsstat3> {
        let fs = self.state.lock().await;
        let file = fs.lookup_by_id(id).map_err(io_to_nfs_error)?;

        Ok(create_fattr(&file))
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn read(
        &self,
        id: fileid3,
        offset: u64,
        count: u32,
    ) -> Result<(Vec<u8>, bool), nfsstat3> {
        let mut fs = self.state.lock().await;

        let entry = fs.lookup_by_id(id).map_err(io_to_nfs_error)?;
        if entry.node_type != NodeType::File {
            return Err(nfsstat3::NFS3ERR_ISDIR);
        }
        let mut file = fs.open_file(&entry).map_err(io_to_nfs_error)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))
                .map_err(io_to_nfs_error)?;
        }
        let mut buf = Vec::with_capacity(count as usize);
        file.take(count as u64)
            .read_to_end(&mut buf)
            .map_err(io_to_nfs_error)?;
        Ok((buf, entry.size > offset + count as u64))
    }

    #[instrument(name = "readdir", skip(self), err(Debug, level = "warn"))]
    async fn readdir(
        &self,
        dirid: fileid3,
        start_after: fileid3,
        max_entries: usize,
    ) -> Result<ReadDirResult, nfsstat3> {
        let fs = self.state.lock().await;

        let dir = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;

        let entries = fs
            .readdir(&dir)
            // TODO this is probably incorrect interpretation of start_after
            // skip_after is not an offset, but rather an inode number
            .skip(start_after as usize)
            .take(max_entries)
            .map(to_dir_entry)
            .collect::<Vec<_>>();
        let end = entries.len() < max_entries;
        Ok(ReadDirResult { entries, end })
    }

    #[instrument(name = "remove", skip(self), err(Debug, level = "warn"))]
    async fn remove(&self, dirid: fileid3, filename: &filename3) -> Result<(), nfsstat3> {
        let mut fs = self.state.lock().await;
        let dir_meta = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;
        fs.delete(&dir_meta, to_string(filename))
            .map_err(io_to_nfs_error)
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    #[allow(unused)]
    async fn rename(
        &self,
        from_dirid: fileid3,
        from_filename: &filename3,
        to_dirid: fileid3,
        to_filename: &filename3,
    ) -> Result<(), nfsstat3> {
        return Err(nfsstat3::NFS3ERR_NOTSUPP);
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    #[allow(unused)]
    async fn mkdir(
        &self,
        dirid: fileid3,
        dirname: &filename3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        let mut fs = self.state.lock().await;

        let parent = fs.lookup_by_id(dirid).map_err(io_to_nfs_error)?;
        let name = String::from_utf8(dirname.0.clone())
            .ok()
            .ok_or(nfsstat3::NFS3ERR_INVAL)?;
        let dir = fs.create_dir(&parent, name).map_err(io_to_nfs_error)?;
        Ok((dir.fid, create_fattr(&dir)))
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn symlink(
        &self,
        _dirid: fileid3,
        _linkname: &filename3,
        _symlink: &nfspath3,
        _attr: &sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    #[instrument(skip(self), err(Debug, level = "warn"))]
    async fn readlink(&self, _id: fileid3) -> Result<nfspath3, nfsstat3> {
        return Err(nfsstat3::NFS3ERR_NOTSUPP);
    }
}

fn io_to_nfs_error(e: io::Error) -> nfsstat3 {
    use io::ErrorKind::*;
    if !matches!(e.kind(), NotFound | PermissionDenied | AlreadyExists) {
        warn!("io error: {:?}", e);
    }

    match e.kind() {
        NotFound => nfsstat3::NFS3ERR_NOENT,
        PermissionDenied => nfsstat3::NFS3ERR_PERM,
        AlreadyExists => nfsstat3::NFS3ERR_EXIST,
        InvalidInput => nfsstat3::NFS3ERR_INVAL,
        InvalidData => nfsstat3::NFS3ERR_INVAL,
        Unsupported => nfsstat3::NFS3ERR_NOTSUPP,
        OutOfMemory => nfsstat3::NFS3ERR_NOSPC,
        _ => nfsstat3::NFS3ERR_SERVERFAULT,
        // Unsupported in stable yet
        // io::ErrorKind::NotADirectory => nfsstat3::NFS3ERR_NOTDIR,
        // io::ErrorKind::IsADirectory => nfsstat3::NFS3ERR_ISDIR,
        // io::ErrorKind::DirectoryNotEmpty => nfsstat3::NFS3ERR_NOTEMPTY,
        // io::ErrorKind::ReadOnlyFilesystem => nfsstat3::NFS3ERR_ROFS,
        // io::ErrorKind::FileTooLarge => nfsstat3::NFS3ERR_FBIG,
        // io::ErrorKind::CrossesDevices => nfsstat3::NFS3ERR_XDEV,
        // io::ErrorKind::TooManyLinks => nfsstat3::NFS3ERR_MLINK,
        // io::ErrorKind::InvalidFilename => nfsstat3::NFS3ERR_INVAL,
    }
}

fn to_dir_entry(meta: FileMeta) -> DirEntry {
    let attr = create_fattr(&meta);
    let fileid = meta.fid;
    let name = nfsstring(meta.name().to_string().into_bytes());
    DirEntry { fileid, name, attr }
}

fn create_fattr(meta: &FileMeta) -> fattr3 {
    let ftype = match meta.node_type {
        NodeType::Directory => ftype3::NF3DIR,
        NodeType::File => ftype3::NF3REG,
    };
    fattr3 {
        ftype,
        mode: 0o755,
        nlink: 1,
        uid: 1000,
        gid: 1000,
        size: meta.size,
        used: meta.size,
        rdev: specdata3 {
            specdata1: 0,
            specdata2: 0,
        },
        fsid: 1,
        fileid: meta.fid,
        atime: nfstime3 {
            seconds: 0,
            nseconds: 0,
        },
        mtime: nfstime3 {
            seconds: 0,
            nseconds: 0,
        },
        ctime: nfstime3 {
            seconds: 0,
            nseconds: 0,
        },
    }
}

fn to_string(filename: &filename3) -> String {
    String::from_utf8(filename.0.clone()).unwrap()
}
