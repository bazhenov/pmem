use pmem::page::{Addr, Transaction, TxRead, TxWrite, Volume, VolumeHandle, PAGE_SIZE};
use replication::{replica_connect, start_replication_server};
use std::{io, net::SocketAddr};
use tokio::task::{spawn_blocking, JoinHandle};

#[tokio::test]
async fn check_replication_simple_case() -> io::Result<()> {
    let mut net = MasterAndReplica::new().await?;

    let bytes = [1, 2, 3, 4];
    let snapshot = net.master_write(|s| s.write(0, bytes)).await;
    assert_eq!(&*snapshot.read(0, 4), &bytes);
    Ok(())
}

#[tokio::test]
async fn check_replication_work_if_connected_later() -> io::Result<()> {
    let mut net = MasterAndReplica::new().await?;

    let bytes = [1, 2, 3, 4];
    let snapshot = net.master_write(|s| s.write(0, bytes)).await;
    net.replica_reconnect().await?;

    assert_eq!(&*snapshot.read(0, 4), &bytes);
    Ok(())
}

#[tokio::test]
async fn check_replication_can_resize_volume() -> io::Result<()> {
    let mut net = MasterAndReplica::with_volume(Volume::new_in_memory(2)).await?;

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
    master_volume: Volume,
    master_addr: SocketAddr,
    replica_handle: VolumeHandle,
    replica_ctrl: JoinHandle<io::Result<()>>,
}

impl MasterAndReplica {
    async fn new() -> io::Result<Self> {
        Self::with_volume(Volume::new_in_memory(1)).await
    }

    async fn with_volume(volume: Volume) -> io::Result<Self> {
        let notify = volume.commit_notify();
        let (master_addr, _) = start_replication_server("127.0.0.1:0", notify).await?;
        let (replica, replica_ctrl) = replica_connect(master_addr).await?;
        Ok(Self {
            master_volume: volume,
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
    async fn master_write(&mut self, f: impl Fn(&mut Transaction)) -> impl TxRead {
        let mut notify = self.master_volume.commit_notify();
        let mut tx = self.master_volume.start();
        f(&mut tx);
        let lsn = self.master_volume.commit(tx).unwrap();

        // Waiting for replica to catch up
        // We need to spawn_blocking here because `next_commit()` is a blocking call
        // if `Runtime` is not multithreaded it may block the only thread that is running server async tasks
        spawn_blocking(move || while notify.next_commit().lsn() < lsn {})
            .await
            .unwrap();
        self.master_volume.start()
    }
}
