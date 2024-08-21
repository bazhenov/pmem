use pmem::page::{
    Addr, CommittedSnapshot, PagePool, PagePoolHandle, Snapshot, TxRead, TxWrite, PAGE_SIZE,
};
use replication::{replica_connect, start_replication_server};
use std::{io, net::SocketAddr, sync::Arc};
use tokio::task::{spawn_blocking, JoinHandle};

mod tracing;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication_simple_case() -> io::Result<()> {
    let mut net = MasterAndReplica::new().await?;

    let bytes = [1, 2, 3, 4];
    let snapshot = net.master_write(|s| s.write(0, bytes)).await;
    assert_eq!(&*snapshot.read(0, 4), &bytes);
    Ok(())
}

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication_work_if_connected_later() -> io::Result<()> {
    let mut net = MasterAndReplica::new().await?;

    let bytes = [1, 2, 3, 4];
    let snapshot = net.master_write(|s| s.write(0, bytes)).await;
    net.replica_reconnect().await?;

    assert_eq!(&*snapshot.read(0, 4), &bytes);
    Ok(())
}

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication_can_resize_pool() -> io::Result<()> {
    let mut net = MasterAndReplica::with_pool(PagePool::new(2)).await?;

    let bytes = [1, 2, 3, 4];
    let snapshot = net.master_write(|s| s.write(0, bytes)).await;

    assert!(
        snapshot.valid_range(PAGE_SIZE as Addr, 1),
        "Second page should be valid"
    );
    assert!(
        !snapshot.valid_range(2 * PAGE_SIZE as Addr, 1),
        "Third page should be invalid"
    );
    Ok(())
}

struct MasterAndReplica {
    master_pool: PagePool,
    master_addr: SocketAddr,
    replica_handle: PagePoolHandle,
    replica_ctrl: JoinHandle<io::Result<()>>,
}

impl MasterAndReplica {
    async fn new() -> io::Result<Self> {
        Self::with_pool(PagePool::default()).await
    }

    async fn with_pool(pool: PagePool) -> io::Result<Self> {
        let notify = pool.commit_notify();
        let (master_addr, _) = start_replication_server("127.1:0", notify).await?;
        let (replica, replica_ctrl) = replica_connect(master_addr).await?;
        Ok(Self {
            master_pool: pool,
            master_addr,
            replica_handle: replica,
            replica_ctrl,
        })
    }

    async fn replica_reconnect(&mut self) -> io::Result<()> {
        self.replica_ctrl.abort();
        let (replica, replica_ctrl) = replica_connect(&self.master_addr).await?;
        self.replica_handle = replica;
        self.replica_ctrl = replica_ctrl;
        Ok(())
    }

    /// Write to master, wait for replica to catch up and returns corresponding snapshot from replica
    async fn master_write(&mut self, f: impl Fn(&mut Snapshot)) -> Arc<CommittedSnapshot> {
        let mut snapshot = self.master_pool.snapshot();
        f(&mut snapshot);
        let lsn = self.master_pool.commit(snapshot);

        let mut handle = self.replica_handle.clone();
        // Waiting for replica to catch up
        // We need to spawn_blocking here because replica.wait_for_commit() is a blocking call
        // if Runtime is not multithreaded it may block the only thread that is running server async tasks
        spawn_blocking(move || handle.wait_for_lsn(lsn))
            .await
            .unwrap()
    }
}
