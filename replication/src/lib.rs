use pmem::page::{CommitNotify, PagePool, Patch, TxWrite};
use protocol::Message;
use std::{borrow::Cow, io, pin::pin, sync::mpsc::Sender};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::oneshot::{self, Receiver},
    task::JoinHandle,
};
use tracing::{info, instrument};

mod protocol;

pub struct ReplicationServer {
    listener: TcpListener,
}

impl ReplicationServer {
    pub async fn bind(addr: impl AsRef<str>) -> io::Result<Self> {
        let listener = TcpListener::bind(addr.as_ref()).await?;
        info!("Listening on {}", addr.as_ref());
        Ok(Self { listener })
    }

    pub async fn run(&self, notify: CommitNotify) -> io::Result<()> {
        loop {
            let (socket, _) = self.listener.accept().await?;
            let _ = socket
                .peer_addr()
                .map(|addr| info!("Accepted connection from {}", addr));
            tokio::spawn(handle_client(socket, notify.clone()));
        }
    }
}

async fn handle_client(mut socket: TcpStream, mut notify: CommitNotify) -> io::Result<()> {
    Message::Handshake.write_to(pin!(&mut socket)).await?;
    let msg = Message::read_from(pin!(&mut socket)).await?;
    if msg != Message::Handshake {
        return io_error("Invalid handshake message");
    }

    loop {
        let snapshot = notify.next_snapshot().await;
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

pub async fn replica_connect(addr: impl AsRef<str>) -> io::Result<(CommitNotify, ShutdownSignal)> {
    let pool = PagePool::default();
    let client = ReplicationClient::connect(addr).await?;
    let commit_notify = pool.commit_notify();

    let (tx, rx) = oneshot::channel();
    let handle = tokio::spawn(replicate_worker(client, pool, rx));
    let shutdown_signal = ShutdownSignal(tx, handle);
    Ok((commit_notify, shutdown_signal))
}

async fn replicate_worker(
    mut client: ReplicationClient,
    mut pool: PagePool,
    mut shutdown: oneshot::Receiver<()>,
) -> io::Result<()> {
    loop {
        tokio::select! {
            patch = client.next_patch() => {
                let mut s = pool.snapshot();
                match patch? {
                    Patch::Write(addr, bytes) => s.write(addr, bytes),
                    Patch::Reclaim(addr, len) => s.reclaim(addr, len),
                }
                pool.commit(s);
            }
            _ = &mut shutdown => break Ok(()),
        }
    }
}

pub struct ReplicationClient {
    socket: TcpStream,
}

impl ReplicationClient {
    pub async fn connect(addr: impl AsRef<str>) -> io::Result<Self> {
        let mut socket = TcpStream::connect(addr.as_ref()).await?;
        Message::Handshake.write_to(pin!(&mut socket)).await?;

        let msg = Message::read_from(pin!(&mut socket)).await?;
        if msg != Message::Handshake {
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
