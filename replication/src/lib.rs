use pmem::volume::{CommitNotify, Patch, TxWrite, Volume, VolumeHandle, LSN};
use protocol::{Message, PROTOCOL_VERSION};
use std::{borrow::Cow, fmt::Debug, io, net::SocketAddr, pin::pin, thread};
use tokio::{
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync,
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
    Message::ServerHello(PROTOCOL_VERSION, notify.pages())
        .write_to(pin!(&mut socket))
        .await?;
    let msg = Message::read_from(pin!(&mut socket)).await?;
    let Message::ClientHello(v) = msg else {
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

    thread::spawn(move || {
        let commit = notify.next_commit();
        while tx
            .blocking_send((
                commit.lsn(),
                commit.patches().to_vec(),
                commit.undo().to_vec(),
            ))
            .is_ok()
        {}
    });

    while let Some((lsn, redo, undo)) = rx.recv().await {
        trace!(lsn = lsn, patches = redo.len(), "Sending snapshot");
        for (r, u) in redo.iter().zip(undo.iter()) {
            Message::Patch(Cow::Borrowed(r), Cow::Borrowed(u))
                .write_to(pin!(&mut socket))
                .await?;
        }
        Message::Commit(lsn).write_to(pin!(&mut socket)).await?;
    }
    Ok(())
}

pub async fn replica_connect(
    addr: impl ToSocketAddrs + Debug,
) -> io::Result<(VolumeHandle, JoinHandle<io::Result<()>>)> {
    let mut socket = TcpStream::connect(&addr).await?;
    trace!(addr = ?addr, "Connected to remote");
    Message::ClientHello(PROTOCOL_VERSION)
        .write_to(pin!(&mut socket))
        .await?;

    let msg = Message::read_from(pin!(&mut socket)).await?;
    let Message::ServerHello(version, pages) = msg else {
        return io_error("Invalid handshake response");
    };
    if version != PROTOCOL_VERSION {
        return io_error("Invalid protocol version");
    }

    let volume = Volume::new_in_memory(pages);
    let read_handle = volume.handle();

    let join_handle = tokio::spawn(client_worker(socket, volume));

    Ok((read_handle, join_handle))
}

#[instrument(skip(client, volume), fields(addr = %client.peer_addr().unwrap()))]
async fn client_worker(mut client: TcpStream, mut volume: Volume) -> io::Result<()> {
    loop {
        let (lsn, snapshot) = next_snapshot(&mut client).await?;
        trace!(
            lsn = lsn,
            patches = snapshot.len(),
            "Received snapshot from master"
        );

        let mut tx = volume.start();
        for patch in snapshot {
            match patch {
                Patch::Write(addr, bytes) => tx.write(addr, bytes),
                Patch::Reclaim(addr, len) => tx.reclaim(addr, len),
            }
        }

        let my_lsn = volume.commit(tx).unwrap();
        trace!(lsn = lsn, my_lsn = my_lsn, "Committed snapshot");
    }
}

pub async fn next_snapshot(mut socket: &mut TcpStream) -> io::Result<(LSN, Vec<Patch>)> {
    let mut patches = vec![];
    loop {
        let msg = Message::read_from(pin!(&mut socket)).await?;
        match msg {
            Message::Patch(p, _) => patches.push(p.into_owned()),
            Message::Commit(lsn) => return Ok((lsn, patches)),
            _ => return io_error("Invalid message type"),
        }
    }
}

pub(crate) fn io_error<T>(error: &str) -> Result<T, io::Error> {
    Err(io::Error::new(io::ErrorKind::InvalidData, error))
}
