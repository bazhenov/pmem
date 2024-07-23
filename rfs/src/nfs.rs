//! Implementation of RFS filesystem adapter for a NFS protocol.
//!
//! To test use following command:
//! ```ignore
//! $ mkdir mnt
//! $ mount -t nfs -o nolocks,vers=3,tcp,port=11111,mountport=11111,soft 127.0.0.1:/ mnt/
//! ```
use crate::{Error, FileMeta, Filesystem, NodeType};
use async_trait::async_trait;
use nfsserve::{
    nfs::{
        fattr3, fileid3, filename3, ftype3, nfspath3, nfsstat3, nfsstring, nfstime3, sattr3,
        specdata3,
    },
    vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities},
};
use std::io::{self, Read, Seek, SeekFrom, Write};
use tokio::sync::Mutex;
use tracing::instrument;

pub struct RFS {
    fs: Mutex<Filesystem>,
    root_id: fileid3,
}

impl RFS {
    pub fn new(fs: Filesystem) -> RFS {
        let root = fs.get_root().unwrap();
        RFS {
            fs: Mutex::new(fs),
            root_id: root.fid,
        }
    }
}

#[async_trait]
#[allow(clippy::blocks_in_conditions)]
impl NFSFileSystem for RFS {
    #[instrument(level = "trace", skip(self), ret)]
    fn root_dir(&self) -> fileid3 {
        self.root_id
    }

    fn capabilities(&self) -> VFSCapabilities {
        VFSCapabilities::ReadWrite
    }

    #[instrument(level = "trace", skip(self, data), fields(data.len = data.len()), err(Debug, level = "warn"))]
    async fn write(&self, id: fileid3, offset: u64, data: &[u8]) -> Result<fattr3, nfsstat3> {
        let mut fs = self.fs.lock().await;

        let meta = fs.lookup_by_id(id).map_err(pmem_to_nfs_error)?;
        let mut file = fs.open_file(&meta).map_err(pmem_to_nfs_error)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))
                .map_err(io_to_nfs_error)?;
        }
        file.write_all(data).map_err(io_to_nfs_error)?;
        file.flush().map_err(io_to_nfs_error)?;
        let new_meta = fs.lookup_by_id(id).map_err(pmem_to_nfs_error)?;
        Ok(create_fattr(&new_meta))
    }

    #[instrument(level = "trace", skip(self, _attr), ret, err(Debug, level = "warn"))]
    async fn create(
        &self,
        dirid: fileid3,
        filename: &filename3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        let mut fs = self.fs.lock().await;
        let dir = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;
        let new_file = fs
            .create_file(&dir, to_string(filename))
            .map_err(pmem_to_nfs_error)?;
        Ok((new_file.fid, create_fattr(&new_file)))
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn create_exclusive(
        &self,
        dirid: fileid3,
        filename: &filename3,
    ) -> Result<fileid3, nfsstat3> {
        let mut fs = self.fs.lock().await;
        let dir = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;
        let new_file = fs
            .create_file(&dir, to_string(filename))
            .map_err(pmem_to_nfs_error)?;
        Ok(new_file.fid)
    }

    #[instrument(level = "trace", skip(self))]
    async fn lookup(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        let fs = self.fs.lock().await;
        let filename = to_string(filename);
        let dir = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;
        let result = fs.lookup(&dir, filename).map_err(pmem_to_nfs_error)?;
        Ok(result.fid)
    }

    #[instrument(level = "trace", skip(self))]
    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        let fs = self.fs.lock().await;
        let entry = fs.lookup_by_id(id).map_err(pmem_to_nfs_error)?;
        Ok(create_fattr(&entry))
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn setattr(&self, id: fileid3, _setattr: sattr3) -> Result<fattr3, nfsstat3> {
        let fs = self.fs.lock().await;

        let file = fs.lookup_by_id(id).map_err(pmem_to_nfs_error)?;

        Ok(create_fattr(&file))
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn read(
        &self,
        id: fileid3,
        offset: u64,
        count: u32,
    ) -> Result<(Vec<u8>, bool), nfsstat3> {
        let mut fs = self.fs.lock().await;

        let entry = fs.lookup_by_id(id).map_err(pmem_to_nfs_error)?;
        if entry.node_type != NodeType::File {
            return Err(nfsstat3::NFS3ERR_ISDIR);
        }
        let mut file = fs.open_file(&entry).map_err(pmem_to_nfs_error)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))
                .map_err(io_to_nfs_error)?;
        }
        let mut buf = Vec::with_capacity(count as usize);
        let bytes_read = file.read_to_end(&mut buf).map_err(io_to_nfs_error)?;
        Ok((buf, bytes_read < count as usize))
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn readdir(
        &self,
        dirid: fileid3,
        start_after: fileid3,
        max_entries: usize,
    ) -> Result<ReadDirResult, nfsstat3> {
        let fs = self.fs.lock().await;

        let dir = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;

        let mut children = fs.readdir(&dir).map_err(pmem_to_nfs_error)?;
        let start_idx = start_after as usize;
        if start_idx >= children.len() {
            return Ok(ReadDirResult {
                entries: vec![],
                end: true,
            });
        }
        let end_idx = (start_idx + max_entries).min(children.len());
        let entries = children
            .drain(start_idx..end_idx)
            .map(to_dir_entry)
            .collect::<Vec<_>>();
        let end = end_idx <= children.len();
        Ok(ReadDirResult { entries, end })
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    #[allow(unused)]
    async fn remove(&self, dirid: fileid3, filename: &filename3) -> Result<(), nfsstat3> {
        let mut fs = self.fs.lock().await;
        let dir_meta = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;
        fs.delete(&dir_meta, to_string(filename))
            .map_err(pmem_to_nfs_error)
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
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

    #[instrument(level = "trace", skip(self), ret, err(Debug, level = "warn"))]
    #[allow(unused)]
    async fn mkdir(
        &self,
        dirid: fileid3,
        dirname: &filename3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        let mut fs = self.fs.lock().await;

        let parent = fs.lookup_by_id(dirid).map_err(pmem_to_nfs_error)?;
        let name = String::from_utf8(dirname.0.clone())
            .ok()
            .ok_or(nfsstat3::NFS3ERR_INVAL)?;
        let dir = fs.create_dir(&parent, name).map_err(pmem_to_nfs_error)?;
        Ok((dir.fid, create_fattr(&dir)))
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn symlink(
        &self,
        _dirid: fileid3,
        _linkname: &filename3,
        _symlink: &nfspath3,
        _attr: &sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    #[instrument(level = "trace", skip(self), err(Debug, level = "warn"))]
    async fn readlink(&self, _id: fileid3) -> Result<nfspath3, nfsstat3> {
        return Err(nfsstat3::NFS3ERR_NOTSUPP);
    }
}

fn pmem_to_nfs_error(e: Error) -> nfsstat3 {
    match e {
        Error::NotFound => nfsstat3::NFS3ERR_NOENT,
        Error::NotSupported => nfsstat3::NFS3ERR_NOTSUPP,
        Error::PathMustBeAbsolute => nfsstat3::NFS3ERR_INVAL,
        Error::AlreadyExists => nfsstat3::NFS3ERR_EXIST,
        Error::PMemError(e) => match e {
            pmem::memory::Error::DataIntegrity(_) => nfsstat3::NFS3ERR_SERVERFAULT,
            pmem::memory::Error::NoSpaceLeft => nfsstat3::NFS3ERR_NOSPC,
            pmem::memory::Error::NullPointer => nfsstat3::NFS3ERR_SERVERFAULT,
        },
        Error::IOError(e) => io_to_nfs_error(e),
        Error::Utf8(_) => nfsstat3::NFS3ERR_INVAL,
    }
}

fn io_to_nfs_error(e: io::Error) -> nfsstat3 {
    match e.kind() {
        io::ErrorKind::NotFound => nfsstat3::NFS3ERR_NOENT,
        io::ErrorKind::PermissionDenied => nfsstat3::NFS3ERR_PERM,
        io::ErrorKind::AlreadyExists => nfsstat3::NFS3ERR_EXIST,
        io::ErrorKind::InvalidInput => nfsstat3::NFS3ERR_INVAL,
        io::ErrorKind::InvalidData => nfsstat3::NFS3ERR_INVAL,
        io::ErrorKind::Unsupported => nfsstat3::NFS3ERR_NOTSUPP,
        io::ErrorKind::OutOfMemory => nfsstat3::NFS3ERR_NOSPC,
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
    let name = nfsstring(meta.name.into_bytes());
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
