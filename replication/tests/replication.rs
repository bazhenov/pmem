use pmem::page::{PagePool, Patch, TxWrite};
use replication::{ReplicationClient, ReplicationServer};
use std::io;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication() -> io::Result<()> {
    let mut pool = PagePool::default();

    let server = ReplicationServer::bind("127.0.0.1:3315").await?;
    let notify = pool.commit_notify();
    let _handle = tokio::spawn(async move { server.run(notify).await });

    let mut snapshot = pool.snapshot();
    let bytes = [1, 2, 3, 4];
    snapshot.write(0, bytes);
    pool.commit(snapshot);

    let mut client = ReplicationClient::connect("127.0.0.1:3315").await?;

    let patch = client.next_patch().await?;
    assert_eq!(patch, Patch::Write(0, bytes.to_vec()));

    Ok(())
}
