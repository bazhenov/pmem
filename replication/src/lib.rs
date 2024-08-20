use pmem::page::{CommitNotify, CommittedSnapshot, PagePool, PagePoolHandle, Patch, TxWrite};
use protocol::{Message, PROTOCOL_VERSION};
use std::{borrow::Cow, io, net::SocketAddr, pin::pin, sync::Arc};
use tokio::{
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::oneshot,
    task::{self, JoinHandle},
};
use tracing::info;

mod protocol;

pub async fn run_replication_server(
    addr: impl ToSocketAddrs,
    notify: CommitNotify,
) -> io::Result<(SocketAddr, JoinHandle<io::Result<()>>)> {
    let listener = TcpListener::bind(addr).await?;
    let addr = listener.local_addr()?;
    info!("Listening on {}", addr);
    let handle = tokio::spawn(accept_loop(listener, notify));
    Ok((addr, handle))
}

pub async fn accept_loop(listener: TcpListener, notify: CommitNotify) -> io::Result<()> {
    loop {
        let (socket, _) = listener.accept().await?;
        if let Ok(addr) = socket.peer_addr() {
            info!("Accepted connection from {}", addr);
        }
        tokio::spawn(handle_client(socket, notify.clone()));
    }
}

async fn handle_client(mut socket: TcpStream, notify: CommitNotify) -> io::Result<()> {
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

    loop {
        let mut notify = notify.clone();
        let snapshot = task::spawn_blocking(move || notify.next_snapshot()).await?;

        for patch in snapshot.patches() {
            Message::Patch(Cow::Borrowed(patch))
                .write_to(pin!(&mut socket))
                .await?;
        }
    }
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
    addr: impl ToSocketAddrs,
) -> io::Result<(PagePoolHandle, JoinHandle<io::Result<()>>)> {
    let pool = PagePool::default();
    let read_handle = pool.handle();

    let client = ReplicationClient::connect(addr).await?;
    let join_handle = tokio::spawn(replicate_worker(client, pool));

    Ok((read_handle, join_handle))
}

async fn replicate_worker(mut client: ReplicationClient, mut pool: PagePool) -> io::Result<()> {
    loop {
        let patch = client.next_patch().await?;
        let mut s = pool.snapshot();
        match patch {
            Patch::Write(addr, bytes) => s.write(addr, bytes),
            Patch::Reclaim(addr, len) => s.reclaim(addr, len),
        }

        pool.commit(s);
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

    pub async fn next_patch(&mut self) -> io::Result<Patch> {
        let msg = Message::read_from(pin!(&mut self.socket)).await?;
        match msg {
            Message::Patch(p) => Ok(p.into_owned()),
            _ => io_error("Invalid message type"),
        }
    }
}

pub(crate) fn io_error<T>(error: &str) -> Result<T, io::Error> {
    Err(io::Error::new(io::ErrorKind::InvalidData, error))
}
