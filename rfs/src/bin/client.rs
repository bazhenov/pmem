use std::io;

use rfs::{Filesystem, FsTree};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter_layer = EnvFilter::from_default_env();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter_layer)
        .init();

    let (mut pool, _) = replication::replica_connect("127.0.0.1:1111")
        .await
        .unwrap();

    loop {
        let snapshot = pool.wait();
        println!("New snapshot LSN: {}", snapshot.lsn());

        let fs = Filesystem::open(snapshot);
        println!("{:?}", FsTree(&fs));
    }
}
