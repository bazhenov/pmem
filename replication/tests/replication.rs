use replication::{ReplicationClient, ReplicationServer};
use std::io;

#[tokio::test]
#[cfg(not(miri))]
async fn check_replication() -> io::Result<()> {
    let server = ReplicationServer::bind("127.0.0.1:3315").await?;

    let handle = tokio::spawn(async move { server.run().await });

    let client = ReplicationClient::connect("127.0.0.1:3315").await?;
    dbg!("next_patch");

    Ok(())
}
