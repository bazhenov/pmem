use pmem::page::{PagePool, TxRead, TxWrite};
use replication::{replica_connect, run_replication_server};
use std::io;
use tracing::init_tracing;

mod tracing;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication() -> io::Result<()> {
    use tokio::task::spawn_blocking;

    init_tracing();
    let mut master_pool = PagePool::default();

    // Running server
    let notify = master_pool.commit_notify();
    let (addr, server_ctrl) = run_replication_server("127.1:0", notify).await?;
    let (mut replica, replica_ctrl) = replica_connect(addr).await?;

    // Writing to master
    let mut snapshot = master_pool.snapshot();
    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    snapshot.write(4, bytes);
    master_pool.commit(snapshot);

    // Waiting for replica to catch up
    // We need to spawn_blocking here because replica.wait_for_commit() is a blocking call
    // if Runtime is not multithreaded it may block the only thread that is running server async tasks
    let snapshot = spawn_blocking(move || replica.wait_for_commit()).await?;
    assert_eq!(&*snapshot.read(0, 4), &bytes);
    assert_eq!(&*snapshot.read(4, 4), &bytes);

    server_ctrl.abort();
    replica_ctrl.abort();

    Ok(())
}
