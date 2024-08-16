use pmem::page::{PagePool, TxRead, TxWrite};
use replication::{replica_connect, ReplicationServer};
use std::io;
use tracing::init_tracing;

mod tracing;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication() -> io::Result<()> {
    init_tracing();
    let mut pool = PagePool::default();

    let server = ReplicationServer::bind("127.0.0.1:3315").await?;
    let notify = pool.commit_notify();
    let _handle = tokio::spawn(async move { server.run(notify).await });

    let mut snapshot = pool.snapshot();
    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    pool.commit(snapshot);

    let (mut replica, ctrl) = replica_connect("127.0.0.1:3315").await?;

    let snapshot = replica.next_snapshot().await;
    assert_eq!(&*snapshot.read(0, 4), &bytes);

    ctrl.shutdown().await??;

    Ok(())
}
