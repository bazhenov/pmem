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

    loop {
        let snapshot = volume.wait();
        println!("New snapshot LSN: {}", snapshot.lsn());

        spawn_blocking(move || {
            let fs = Filesystem::open(snapshot);
            println!("{:?}", FsTree(&fs));
        });
    }
}
