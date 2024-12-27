use crate::{Change, ChangeKind, Filesystem};
use pmem::volume::{TxRead, TxWrite};
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read, Write},
    rc::Rc,
};
use tracing::warn;

pub type FileUpdater = fn(input: &mut dyn Read, out: &mut dyn Write) -> io::Result<()>;
pub struct FsSync(pub FileUpdater);

pub fn write_sha256(input: &mut dyn Read, out: &mut dyn Write) -> io::Result<()> {
    let mut sha = Sha256::new();
    let mut buffer = vec![0; 4096];
    let mut size = input.read(&mut buffer)?;
    while size > 0 {
        sha.write_all(&buffer[..size])?;
        size = input.read(&mut buffer)?;
    }
    for b in sha.finalize() {
        write!(out, "{:02x}", b)?;
    }
    Ok(())
}

impl FsSync {
    pub fn update_fs(
        &self,
        fs: &mut Filesystem<impl TxWrite>,
        base: &Filesystem<impl TxRead>,
    ) -> io::Result<()> {
        let mut removed = vec![];
        let mut updated = vec![];

        for change in fs.changes_from(base).filter(|i| i.entry.is_file()) {
            let Change {
                kind, path, entry, ..
            } = change;
            match kind {
                ChangeKind::Add | ChangeKind::Update => updated.push((path, entry.name)),
                ChangeKind::Delete => removed.push((path, entry.name)),
            }
        }

        for (path, name) in removed {
            let path = Rc::unwrap_or_clone(path);
            let path = path.to_str().unwrap();
            let file_name = format!("{}.sha", name);
            if let Ok(dir) = fs.find(path) {
                if fs.delete(&dir, &file_name).is_err() {
                    warn!("Failed to delete file: {}", file_name);
                }
            }
        }

        for (path, name) in updated {
            let path = Rc::unwrap_or_clone(path);
            let directory_name = path.to_str().unwrap();

            let directory = fs.find(directory_name)?;
            let mut changed_file = fs.open_file(&fs.lookup(&directory, &name)?)?;

            let csum_file_name = format!("{}.sha", name);

            let csum_file = match fs.lookup(&directory, &csum_file_name) {
                Ok(f) => f,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    fs.create_file(&directory, &csum_file_name)?
                }
                Err(e) => return Err(e),
            };
            let mut csum_file = fs.open_file(&csum_file)?;
            (self.0)(&mut changed_file, &mut csum_file)?;
            csum_file.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};

    use super::*;
    use crate::Filesystem;
    use pmem::volume::{Transaction, Volume};

    #[test]
    fn test_fs_sync() -> io::Result<()> {
        let (mut fs, volume) = create_fs();
        let snapshot = volume.snapshot();
        let base = Filesystem::open(snapshot)?;

        let s = FsSync(write_sha256);

        let root = fs.get_root()?;
        let meta = fs.create_file(&root, "test.txt")?;
        assert!(meta.is_file());
        {
            let mut f = fs.open_file(&meta)?;
            f.write_all(b"Hello")?;
            f.flush()?;
        }
        s.update_fs(&mut fs, &base)?;

        let f = fs.find("/test.txt.sha")?;
        let mut f = fs.open_file(&f)?;
        let mut str = String::new();
        f.read_to_string(&mut str)?;

        assert_eq!(
            str,
            "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
        );

        Ok(())
    }

    fn create_fs() -> (Filesystem<Transaction>, Volume) {
        let mut volume = Volume::with_capacity(1024 * 1024);
        let fs = Filesystem::allocate(volume.start());
        volume.commit(fs.finish().unwrap()).unwrap();
        let fs = Filesystem::open(volume.start()).unwrap();
        (fs, volume)
    }
}
