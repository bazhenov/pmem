use pmem::page::{CommitNotify, CommittedSnapshot, PagePool, PagePoolHandle, Patch, TxWrite, LSN};
use protocol::{Message, PROTOCOL_VERSION};
use std::{borrow::Cow, fmt::Debug, io, net::SocketAddr, pin::pin, sync::Arc, thread};
use tokio::{
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::{self, oneshot},
    task::JoinHandle,
};
use tracing::{info, instrument, trace};

mod protocol;

pub async fn start_replication_server(
    addr: impl ToSocketAddrs,
    notify: CommitNotify,
) -> io::Result<(SocketAddr, JoinHandle<io::Result<()>>)> {
    let listener = TcpListener::bind(addr).await?;
    let addr = listener.local_addr()?;
    info!(addr = ?addr, "Listening");
    let handle = tokio::spawn(accept_loop(listener, notify));
    Ok((addr, handle))
}

pub async fn accept_loop(listener: TcpListener, notify: CommitNotify) -> io::Result<()> {
    loop {
        let (socket, _) = listener.accept().await?;
        if let Ok(addr) = socket.peer_addr() {
            info!(addr = ?addr, "Accepted connection");
        }
        tokio::spawn(server_worker(socket, notify.clone()));
    }
}

#[instrument(skip(socket, notify), fields(addr = %socket.peer_addr().unwrap()))]
async fn server_worker(mut socket: TcpStream, mut notify: CommitNotify) -> io::Result<()> {
    Message::Handshake(PROTOCOL_VERSION)
        .write_to(pin!(&mut socket))
        .await?;
    let msg = Message::read_from(pin!(&mut socket)).await?;
    let Message::Handshake(v) = msg else {
        return io_error("Invalid handshake message");
    };
    if v != PROTOCOL_VERSION {
        return io_error("Invalid protocol version");
    }

    // We do not want to send LSN=0 to the client, because it will cause a shift in LSNs
    // between the client and the server. This problem should go away once we implement
    // FSM for tracking in-flight patches and commits with a given LSN on a client.
    let last_seen_lsn = notify.last_seen_lsn().max(1);
    trace!(lsn = last_seen_lsn, "Starting log relay");
    let (tx, mut rx) = sync::mpsc::channel(1);

    thread::spawn(move || while tx.blocking_send(notify.next_snapshot()).is_ok() {});

    while let Some(snapshot) = rx.recv().await {
        for lsn in last_seen_lsn..=snapshot.lsn() {
            let next_snapshot = snapshot.find_at_lsn(lsn).unwrap();
            let patches = next_snapshot.patches();
            trace!(lsn = lsn, patches = patches.len(), "Sending snapshot");
            for patch in patches {
                Message::Patch(Cow::Borrowed(patch))
                    .write_to(pin!(&mut socket))
                    .await?;
            }
            Message::Commit(lsn).write_to(pin!(&mut socket)).await?;
        }
    }
    Ok(())
}

pub struct ShutdownSignal(oneshot::Sender<()>, JoinHandle<io::Result<()>>);

impl ShutdownSignal {
    pub fn shutdown(self) -> JoinHandle<io::Result<()>> {
        // we can safely ignore the error. send() returns an error only if the receiver has been dropped
        // and the worker has already shutdown
        let _ = self.0.send(());
        self.1
    }
}

pub struct PoolReplica {
    pool: PagePool,
    last_snapshot: Arc<CommittedSnapshot>,
}

impl PoolReplica {
    pub fn new(pool: PagePool) -> Self {
        let last_snapshot = Arc::new(CommittedSnapshot::default());
        Self {
            pool,
            last_snapshot,
        }
    }

    pub fn snapshot(&self) -> Arc<CommittedSnapshot> {
        self.last_snapshot.clone()
    }

    pub fn commit_notify(&self) -> CommitNotify {
        self.pool.commit_notify()
    }
}

pub async fn replica_connect(
    addr: impl ToSocketAddrs + Debug,
) -> io::Result<(PagePoolHandle, JoinHandle<io::Result<()>>)> {
    let pool = PagePool::default();
    let read_handle = pool.handle();

    let client = ReplicationClient::connect(&addr).await?;
    trace!(addr = ?addr, "Connected to remote");
    let join_handle = tokio::spawn(client_worker(client, pool));

    Ok((read_handle, join_handle))
}

#[instrument(skip(client, pool), fields(addr = %client.socket.peer_addr().unwrap()))]
async fn client_worker(mut client: ReplicationClient, mut pool: PagePool) -> io::Result<()> {
    loop {
        let (lsn, snapshot) = client.next_snapshot().await?;
        trace!(
            lsn = lsn,
            patches = snapshot.len(),
            "Received snapshot from master"
        );

        let mut s = pool.snapshot();
        for patch in snapshot {
            match patch {
                Patch::Write(addr, bytes) => s.write(addr, bytes),
                Patch::Reclaim(addr, len) => s.reclaim(addr, len),
            }
        }

        let my_lsn = pool.commit(s);
        trace!(lsn = lsn, my_lsn = my_lsn, "Committed snapshot");
    }
}

pub struct ReplicationClient {
    socket: TcpStream,
}

impl ReplicationClient {
    pub async fn connect(addr: impl ToSocketAddrs) -> io::Result<Self> {
        let mut socket = TcpStream::connect(addr).await?;
        Message::Handshake(PROTOCOL_VERSION)
            .write_to(pin!(&mut socket))
            .await?;

        let msg = Message::read_from(pin!(&mut socket)).await?;
        if msg != Message::Handshake(PROTOCOL_VERSION) {
            return io_error("Invalid handshake response");
        }
        Ok(Self { socket })
    }

    pub async fn next_snapshot(&mut self) -> io::Result<(LSN, Vec<Patch>)> {
        let mut patches = vec![];
        loop {
            let msg = Message::read_from(pin!(&mut self.socket)).await?;
            match msg {
                Message::Patch(p) => patches.push(p.into_owned()),
                Message::Commit(lsn) => return Ok((lsn, patches)),
                _ => return io_error("Invalid message type"),
            }
        }
    }
}

pub(crate) fn io_error<T>(error: &str) -> Result<T, io::Error> {
    Err(io::Error::new(io::ErrorKind::InvalidData, error))
}
