use rfs::{Filesystem, FsTree};
use std::io;
use tokio::task::spawn_blocking;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter_layer = EnvFilter::from_default_env();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter_layer)
        .init();

    let (mut volume, _) = replication::replica_connect("127.0.0.1:1111")
        .await
        .unwrap();

    let mut commit_notify = volume.commit_notify();
    let mut snapshot = volume.snapshot();
    loop {
        let next_snapshot = commit_notify.next_snapshot();
        println!("New snapshot LSN: {}", next_snapshot.lsn());

        let a = snapshot.clone();
        let b = next_snapshot.clone();
        spawn_blocking(move || {
            let fs_a = Filesystem::open(a).unwrap();
            let fs_b = Filesystem::open(b).unwrap();
            let changes = fs_b.changes_from(&fs_a);
            println!("{:?}", FsTree(&fs_a));
            for change in changes {
                println!("{:?} {}", change.kind(), change.into_path().display());
            }
        });
        snapshot = next_snapshot;
    }
}
