use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use pmem::{page::PagePool, Memory};
use rfs::{nfs::RFS, Filesystem};
use std::io;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

const HOSTPORT: u32 = 11111;

#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(io::stderr);
    let filter_layer = EnvFilter::from_default_env();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter_layer)
        .init();

    let pool = PagePool::new(1024);
    let mem = Memory::new(pool);
    let fs = Filesystem::allocate(mem.start());
    let listener = NFSTcpListener::bind(&format!("127.0.0.1:{HOSTPORT}"), RFS::new(fs))
        .await
        .unwrap();
    listener.handle_forever().await.unwrap();
}
