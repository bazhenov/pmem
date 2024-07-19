use async_trait::async_trait;
use nfsserve::{
    nfs::{
        self, fattr3, fileid3, filename3, ftype3, nfspath3, nfsstat3, nfstime3, sattr3, specdata3,
    },
    tcp::*,
    vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities},
};
use pmem::{
    fs::{self, FileMeta, Filesystem, NodeType},
    memory, Memory, Storable,
};
use std::{io::Read, time::SystemTime};
use std::{
    io::{Seek, SeekFrom},
    sync::Mutex,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum Error {
    #[error("Memory")]
    MemoryError(#[from] memory::Error),
}

#[derive(Debug, Clone)]
enum FSContents {
    File(Vec<u8>),
    Directory(Vec<fileid3>),
}
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FSEntry {
    id: fileid3,
    attr: fattr3,
    name: filename3,
    parent: fileid3,
    contents: FSContents,
}

fn make_file(name: &str, id: fileid3, parent: fileid3, contents: &[u8]) -> FSEntry {
    let attr = fattr3 {
        ftype: ftype3::NF3REG,
        mode: 0o755,
        nlink: 1,
        uid: 507,
        gid: 507,
        size: contents.len() as u64,
        used: contents.len() as u64,
        rdev: specdata3::default(),
        fsid: 0,
        fileid: id,
        atime: nfstime3::default(),
        mtime: nfstime3::default(),
        ctime: nfstime3::default(),
    };
    FSEntry {
        id,
        attr,
        name: name.as_bytes().into(),
        parent,
        contents: FSContents::File(contents.to_vec()),
    }
}

fn make_dir(name: &str, id: fileid3, parent: fileid3, contents: Vec<fileid3>) -> FSEntry {
    let attr = fattr3 {
        ftype: ftype3::NF3DIR,
        mode: 0o777,
        nlink: 1,
        uid: 507,
        gid: 507,
        size: 0,
        used: 0,
        rdev: specdata3::default(),
        fsid: 0,
        fileid: id,
        atime: nfstime3::default(),
        mtime: nfstime3::default(),
        ctime: nfstime3::default(),
    };
    FSEntry {
        id,
        attr,
        name: name.as_bytes().into(),
        parent,
        contents: FSContents::Directory(contents),
    }
}

pub struct DemoFS {
    fs: Mutex<Filesystem>,
}

impl Default for DemoFS {
    fn default() -> DemoFS {
        let mem = Memory::default();
        let fs = Mutex::new(Filesystem::allocate(mem.start()));
        DemoFS { fs }
    }
}

#[async_trait]
impl NFSFileSystem for DemoFS {
    fn root_dir(&self) -> fileid3 {
        let fs = self.fs.lock().unwrap();
        fs.get_root().unwrap().fid
    }

    fn capabilities(&self) -> VFSCapabilities {
        VFSCapabilities::ReadWrite
    }

    async fn write(&self, id: fileid3, offset: u64, data: &[u8]) -> Result<fattr3, nfsstat3> {
        {
            let mut fs = self.fs.lock().unwrap();
            let inode = fs.lookup_by_id(id);
            let mut fssize = todo!();
            if let FSContents::File(bytes) = &mut fs[id as usize].contents {
                let offset = offset as usize;
                if offset + data.len() > bytes.len() {
                    bytes.resize(offset + data.len(), 0);
                    bytes[offset..].copy_from_slice(data);
                    fssize = bytes.len() as u64;
                }
            }
            fs[id as usize].attr.size = fssize;
            fs[id as usize].attr.used = fssize;
        }
        self.getattr(id).await
    }

    async fn create(
        &self,
        dirid: fileid3,
        filename: &filename3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        let mut fs = self.fs.lock().unwrap();
        let dir = fs
            .lookup_by_id(dirid)
            .map_err(to_nfs_error)?
            .ok_or(nfsstat3::NFS3ERR_NOENT)?;
        let new_file = fs
            .create_file(&dir, to_string(&filename))
            .map_err(to_nfs_error)?;
        Ok((new_file.fid, create_fattr(new_file)))
    }

    async fn create_exclusive(
        &self,
        _dirid: fileid3,
        _filename: &filename3,
    ) -> Result<fileid3, nfsstat3> {
        Err(nfsstat3::NFS3ERR_NOTSUPP)
    }

    async fn lookup(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        let fs = self.fs.lock().unwrap();
        let dir = fs
            .lookup_by_id(dirid as u64)
            .map_err(to_nfs_error)?
            .ok_or(nfsstat3::NFS3ERR_NOENT)?;
        fs.lookup(&dir, to_string(filename))
            .map_err(to_nfs_error)?
            .ok_or(nfsstat3::NFS3ERR_NOENT)
            .map(|f| f.fid)
    }

    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        let fs = self.fs.lock().unwrap();
        let entry = fs
            .lookup_by_id(id as u64)
            .map_err(to_nfs_error)?
            .ok_or(nfsstat3::NFS3ERR_NOENT)?;
        Ok(create_fattr(entry))
    }

    async fn setattr(&self, _id: fileid3, _setattr: sattr3) -> Result<fattr3, nfsstat3> {
        Err(nfsstat3::NFS3ERR_NOTSUPP)
    }

    async fn read(
        &self,
        id: fileid3,
        offset: u64,
        count: u32,
    ) -> Result<(Vec<u8>, bool), nfsstat3> {
        let mut fs = self.fs.lock().unwrap();

        let entry = fs
            .lookup_by_id(id)
            .map_err(to_nfs_error)?
            .ok_or(nfsstat3::NFS3ERR_NOENT)?;
        if entry.node_type == NodeType::Directory {
            let mut file = fs.open_file(&entry).map_err(to_nfs_error)?;
            file.seek(SeekFrom::Start(offset));
            let mut buf = vec![0u8; count as usize];
            file.read_exact(&mut buf);
            Ok((buf, true))
        } else {
            Err(nfsstat3::NFS3ERR_ISDIR)
        }
    }

    async fn readdir(
        &self,
        dirid: fileid3,
        start_after: fileid3,
        max_entries: usize,
    ) -> Result<ReadDirResult, nfsstat3> {
        let fs = self.fs.lock().unwrap();
        let entry = fs.get(dirid as usize).ok_or(nfsstat3::NFS3ERR_NOENT)?;
        if let FSContents::File(_) = entry.contents {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        } else if let FSContents::Directory(dir) = &entry.contents {
            let mut ret = ReadDirResult {
                entries: Vec::new(),
                end: false,
            };
            let mut start_index = 0;
            if start_after > 0 {
                if let Some(pos) = dir.iter().position(|&r| r == start_after) {
                    start_index = pos + 1;
                } else {
                    return Err(nfsstat3::NFS3ERR_BAD_COOKIE);
                }
            }
            let remaining_length = dir.len() - start_index;

            for i in dir[start_index..].iter() {
                ret.entries.push(DirEntry {
                    fileid: *i,
                    name: fs[(*i) as usize].name.clone(),
                    attr: fs[(*i) as usize].attr,
                });
                if ret.entries.len() >= max_entries {
                    break;
                }
            }
            if ret.entries.len() == remaining_length {
                ret.end = true;
            }
            return Ok(ret);
        }
        Err(nfsstat3::NFS3ERR_NOENT)
    }

    /// Removes a file.
    /// If not supported dur to readonly file system
    /// this should return Err(nfsstat3::NFS3ERR_ROFS)
    #[allow(unused)]
    async fn remove(&self, dirid: fileid3, filename: &filename3) -> Result<(), nfsstat3> {
        return Err(nfsstat3::NFS3ERR_NOTSUPP);
    }

    /// Removes a file.
    /// If not supported dur to readonly file system
    /// this should return Err(nfsstat3::NFS3ERR_ROFS)
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

    #[allow(unused)]
    async fn mkdir(
        &self,
        _dirid: fileid3,
        _dirname: &filename3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn symlink(
        &self,
        _dirid: fileid3,
        _linkname: &filename3,
        _symlink: &nfspath3,
        _attr: &sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }
    async fn readlink(&self, _id: fileid3) -> Result<nfspath3, nfsstat3> {
        return Err(nfsstat3::NFS3ERR_NOTSUPP);
    }
}

fn to_nfs_error(e: fs::Error) -> nfsstat3 {
    match e {
        fs::Error::NotFound => nfsstat3::NFS3ERR_NOENT,
        fs::Error::NotSupported => nfsstat3::NFS3ERR_NOTSUPP,
        fs::Error::PathMustBeAbsolute => nfsstat3::NFS3ERR_INVAL,
        fs::Error::AlreadyExists => nfsstat3::NFS3ERR_EXIST,
        fs::Error::PMemError(_) => nfsstat3::NFS3ERR_IO,
        fs::Error::IOError(_) => nfsstat3::NFS3ERR_IO,
        fs::Error::Utf8(_) => nfsstat3::NFS3ERR_INVAL,
    }
}

fn create_fattr(meta: FileMeta) -> fattr3 {
    let ftype = match meta.node_type {
        fs::NodeType::Directory => ftype3::NF3DIR,
        fs::NodeType::File => ftype3::NF3REG,
    };
    fattr3 {
        ftype,
        mode: 0,
        nlink: 1,
        uid: 0,
        gid: 0,
        size: 0,
        used: 0,
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

const HOSTPORT: u32 = 11111;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::stderr)
        .init();
    let listener = NFSTcpListener::bind(&format!("127.0.0.1:{HOSTPORT}"), DemoFS::default())
        .await
        .unwrap();
    listener.handle_forever().await.unwrap();
}
// Test with
// mount -t nfs -o nolocks,vers=3,tcp,port=12000,mountport=12000,soft 127.0.0.1:/ mnt/
