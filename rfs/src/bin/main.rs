use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use pmem::page::PagePool;
use rfs::{nfs::RFS, Filesystem};
use std::io::{self, Write};
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

    let mut pool = PagePool::with_capacity(100 * 1024 * 1024);
    let tx = Filesystem::allocate(pool.start()).finish();
    pool.commit(tx);

    let commit_notify = pool.commit_notify();

    let rfs = RFS::new(pool.start());
    let state = rfs.state_handle();
    let listener = NFSTcpListener::bind(&format!("127.0.0.1:{HOSTPORT}"), rfs)
        .await
        .unwrap();

    let _ = replication::start_replication_server("127.0.0.1:1111", commit_notify).await;

    tokio::spawn(async move { listener.handle_forever().await });

    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut cmd = String::new();
        io::stdin().read_line(&mut cmd).unwrap();

        if cmd.trim() == "exit" {
            break;
        } else if cmd.trim() == "commit" {
            let mut s = state.lock().await;
            s.commit(&mut pool).await;
            println!("Commited")
        } else {
            println!("Unknown command: {}", cmd)
        }
    }
}
