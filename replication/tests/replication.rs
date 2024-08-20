use pmem::page::{PagePool, TxRead, TxWrite};
use replication::{replica_connect, start_replication_server};
use std::io;
use tokio::task::spawn_blocking;
use tracing::init_tracing;

mod tracing;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication_simple_case() -> io::Result<()> {
    init_tracing();
    let mut master_pool = PagePool::default();

    // Running server
    let notify = master_pool.commit_notify();
    let (addr, server_ctrl) = start_replication_server("127.1:0", notify).await?;
    let (mut replica, replica_ctrl) = replica_connect(addr).await?;

    // Writing 2 disconnected patches to master to check that
    // replica will replicate all patches in a snapshot as a single transaction
    let mut snapshot = master_pool.snapshot();

    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    snapshot.write(10, bytes);
    let expected_lsn = master_pool.commit(snapshot);
    println!("LSN = {}", expected_lsn);

    // Waiting for replica to catch up
    // We need to spawn_blocking here because wait_for_commit() is a blocking call
    // if Runtime is not multithreaded it may block the only thread that is running server async tasks
    let snapshot = spawn_blocking(move || replica.wait_for_commit()).await?;
    assert_eq!(snapshot.lsn(), expected_lsn);
    assert_eq!(&*snapshot.read(0, 4), &bytes);
    assert_eq!(&*snapshot.read(10, 4), &bytes);

    server_ctrl.abort();
    replica_ctrl.abort();

    Ok(())
}

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication_work_if_connected_later() -> io::Result<()> {
    let mut master_pool = PagePool::default();

    // Running server
    let notify = master_pool.commit_notify();
    let (addr, server_ctrl) = start_replication_server("127.1:0", notify).await?;

    // Writing to master
    let mut snapshot = master_pool.snapshot();
    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    master_pool.commit(snapshot);

    let mut snapshot = master_pool.snapshot();
    snapshot.write(4, bytes);
    let last_lsn = master_pool.commit(snapshot);

    let (mut replica, replica_ctrl) = replica_connect(addr).await?;

    // Waiting for replica to catch up
    // We need to spawn_blocking here because replica.wait_for_commit() is a blocking call
    // if Runtime is not multithreaded it may block the only thread that is running server async tasks
    let snapshot = spawn_blocking(move || replica.wait_for_lsn(last_lsn - 1)).await?;
    assert_eq!(&*snapshot.read(0, 4), &bytes);

    replica_ctrl.abort();
    server_ctrl.abort();

    Ok(())
}
