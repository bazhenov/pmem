use pmem::page::{PagePool, TxRead, TxWrite};
use replication::{replica_connect, run_replication_server};
use std::io;
use tracing::init_tracing;

mod tracing;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication() -> io::Result<()> {
    init_tracing();
    let mut pool = PagePool::default();

    let notify = pool.commit_notify();
    let (addr, server_ctrl) = run_replication_server("127.1:0", notify).await?;
    let (mut replica, replica_ctrl) = replica_connect(addr).await?;

    let mut snapshot = pool.snapshot();
    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    pool.commit(snapshot);

    let snapshot = replica.next_snapshot().await;
    assert_eq!(&*snapshot.read(0, 4), &bytes);

    server_ctrl.abort();
    replica_ctrl.abort();

    Ok(())
}
