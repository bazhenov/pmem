use pmem::page::{CommitNotify, Patch};
use protocol::Message;
use std::{borrow::Cow, io, pin::pin};
use tokio::net::{TcpListener, TcpStream};

mod protocol;

pub struct ReplicationServer {
    listener: TcpListener,
}

impl ReplicationServer {
    pub async fn bind(addr: impl AsRef<str>) -> io::Result<Self> {
        let listener = TcpListener::bind(addr.as_ref()).await?;
        Ok(Self { listener })
    }

    pub async fn run(&self, notify: CommitNotify) -> io::Result<()> {
        loop {
            let (socket, _) = self.listener.accept().await?;
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

pub struct ReplicationClient {
    socket: TcpStream,
}

impl ReplicationClient {
    pub async fn connect(addr: impl Into<String>) -> io::Result<Self> {
        let mut socket = TcpStream::connect(&addr.into()).await?;
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
