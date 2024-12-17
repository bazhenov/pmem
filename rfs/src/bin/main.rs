use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use pmem::{
    driver::FileDriver,
    volume::{TxRead, TxWrite, Volume},
};
use rfs::{
    nfs::RFS,
    sync::{write_sha256, FsSync},
    Filesystem, FsTree,
};
use std::{
    env, fs,
    io::{self, Write},
    process::Command,
};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

const HOSTPORT: u32 = 11111;

#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter_layer = EnvFilter::from_default_env();

    let update_sha = env::var("UPDATE_SHA").ok().is_some();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter_layer)
        .init();

    let file_path = "./target/test.db";
    let db_exists = fs::metadata(file_path).is_err();
    let driver = FileDriver::from_file(file_path).unwrap();
    let mut volume = Volume::with_capacity_and_driver(2 * 1024 * 1024 * 1024, driver)
        .expect("Volume creation failed");
    if db_exists {
        warn!(path = file_path, "Allocating FS");
        let tx = Filesystem::allocate(volume.start()).finish().unwrap();
        volume.commit(tx).unwrap();
    } else {
        info!(path = file_path, "Opening FS");
    }

    let mut base = Filesystem::open(volume.start()).unwrap();

    let fs = Filesystem::open(volume.start()).unwrap();
    let mut rfs = RFS::new(fs).unwrap();
    let listener = NFSTcpListener::bind(&format!("127.0.0.1:{HOSTPORT}"), rfs.clone())
        .await
        .unwrap();

    let _ = replication::start_replication_server("127.0.0.1:1111", volume.handle()).await;

    tokio::spawn(async move { listener.handle_forever().await });

    mount_nfs_share();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut cmd = String::new();
        io::stdin().read_line(&mut cmd).unwrap();

        // empty string is CTRL+D
        if cmd.trim() == "exit" || cmd.is_empty() {
            break;
        } else if cmd.trim() == "commit" {
            if update_sha {
                update_hashes(&mut *rfs.lock().await, &base);
            }
            rfs.commit(&mut volume).await;

            let fs = Filesystem::open(volume.start()).unwrap();
            let changes = fs.changes_from(&base);
            for change in changes {
                println!("  {:?} {}", change.kind(), change.into_path().display());
            }
            println!("{:?}", FsTree(&fs));
            base = fs;

            println!("Committed")
        } else {
            println!("Unknown command: {:?}", cmd)
        }
    }
    unmount_nfs_share();
}

fn mount_nfs_share() {
    println!("Mounting NFS share...");
    let child = Command::new("sh")
        .arg("-c")
        .arg("mount -t nfs -o nolocks,vers=3,tcp,port=11111,mountport=11111,soft 127.0.0.1:/ mnt/")
        .output();
    match child {
        Ok(output) if output.status.success() => println!("NFS share mounted successfully."),
        Ok(output) => eprintln!(
            "Failed to mount NFS share. Error: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(e) => eprintln!("Failed to execute mount command: {}", e),
    }
}

fn unmount_nfs_share() {
    let child = Command::new("sh").arg("-c").arg("umount mnt/").output();
    match child {
        Ok(output) if output.status.success() => {}
        Ok(output) => eprintln!(
            "Failed to unmount NFS share. Error: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(e) => eprintln!("Failed to execute mount command: {}", e),
    }
}

pub fn update_hashes(fs: &mut Filesystem<impl TxWrite>, base: &Filesystem<impl TxRead>) {
    let sync = FsSync(write_sha256);
    sync.update_fs(fs, base).unwrap();
}
