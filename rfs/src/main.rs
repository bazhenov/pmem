use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use pmem::{page::PagePool, Memory};
use rfs::{nfs::RFS, ChangeKind, Filesystem};
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
    let mem = Memory::new(pool.snapshot());
    let tx = Filesystem::allocate(mem).finish();
    pool.commit(tx);

    let rfs = RFS::new(pool.snapshot());
    let state = rfs.state_handle();
    let listener = NFSTcpListener::bind(&format!("127.0.0.1:{HOSTPORT}"), rfs)
        .await
        .unwrap();

    tokio::spawn(async move { listener.handle_forever().await });

    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut cmd = String::new();
        io::stdin().read_line(&mut cmd).unwrap();

        if cmd.trim() == "exit" {
            break;
        } else if cmd.trim() == "commit" {
            let changes = {
                let mut s = state.lock().await;
                s.commit_and_get_changes(&mut pool).await
            };
            for change in changes {
                let marker = match change.kind() {
                    ChangeKind::Add => "A",
                    ChangeKind::Delete => "D",
                    ChangeKind::Update => "M",
                };
                println!("{} {}", marker, change.into_path().display());
            }
        } else {
            println!("Unknown command: {}", cmd)
        }
    }
}
