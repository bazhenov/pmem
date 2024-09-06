use pmem::{
    driver::PageDriver,
    volume::{Addr, Commit, PageNo, Patch, TxRead, TxWrite, Volume, VolumeHandle, LSN, PAGE_SIZE},
};
use protocol::{Message, PROTOCOL_VERSION};
use std::{
    borrow::Cow, collections::HashMap, fmt::Debug, io, mem, net::SocketAddr, pin::pin, thread,
};
use tokio::{
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::{self, mpsc, oneshot},
    task::JoinHandle,
};
use tracing::{event, info, instrument, span, trace, warn, Level};

mod protocol;

pub async fn start_replication_server(
    addr: impl ToSocketAddrs,
    handle: VolumeHandle,
) -> io::Result<(SocketAddr, JoinHandle<io::Result<()>>)> {
    let listener = TcpListener::bind(addr).await?;
    let addr = listener.local_addr()?;
    info!(addr = ?addr, "Listening");
    let handle = tokio::spawn(accept_loop(listener, handle));
    Ok((addr, handle))
}

pub async fn accept_loop(listener: TcpListener, handle: VolumeHandle) -> io::Result<()> {
    loop {
        let (socket, _) = listener.accept().await?;
        if let Ok(addr) = socket.peer_addr() {
            info!(addr = ?addr, "Accepted connection");
        }

        // TODO: Here is the leak. advance_to_latest() is called only on a new connection.
        // let lsn = notify.advance_to_latest();
        // trace!(lsn = lsn, "Advancing to latest");
        tokio::spawn(server_worker(socket, handle.clone()));
    }
}

#[instrument(skip(socket, handle), fields(addr = %socket.peer_addr().unwrap()))]
async fn server_worker(mut socket: TcpStream, mut handle: VolumeHandle) -> io::Result<()> {
    Message::ServerHello(PROTOCOL_VERSION, handle.pages())
        .write_to(pin!(&mut socket))
        .await?;
    let msg = Message::read_from(pin!(&mut socket)).await?;
    let Message::ClientHello(v) = msg else {
        return io_result("Invalid handshake message");
    };
    if v != PROTOCOL_VERSION {
        return io_result("Invalid protocol version");
    }

    // We do not want to send LSN=0 to the client, because it will cause a shift in LSNs
    // between the client and the server. This problem should go away once we implement
    // FSM for tracking in-flight patches and commits with a given LSN on a client.
    let last_seen_lsn = handle.last_seen_lsn().max(1);
    trace!(lsn = last_seen_lsn, "Starting log relay");
    let (tx, mut rx) = sync::mpsc::channel(10);

    {
        let mut handle = handle.clone();
        thread::spawn(move || loop {
            let mut result = Ok(());
            while result.is_ok() {
                let commit = handle.wait_commit();
                trace!(lsn = commit.lsn(), "Got next commit");
                result = tx.blocking_send((
                    commit.lsn(),
                    commit.patches().to_vec(),
                    commit.undo().to_vec(),
                ));
            }
        });
    }

    let mut socket = pin!(socket);

    loop {
        tokio::select! {
            snapshot = rx.recv() => {
                if let Some((lsn, redo, undo)) = snapshot {
                    trace!(lsn = lsn, patches = redo.len(), "Sending snapshot");
                    for (r, u) in redo.iter().zip(undo.iter()) {
                        Message::Patch(Cow::Borrowed(r), Cow::Borrowed(u))
                            .write_to(pin!(&mut socket))
                            .await?;
                    }
                    Message::Commit(lsn).write_to(socket.as_mut()).await?;
                }else{
                    info!("No more snapshots");
                    return Ok(());
                }
            }
            msg = Message::read_from(socket.as_mut()) => {
                match msg? {
                    Message::PageRequest(corelation_id, page_no) => {
                        trace!(page_no, cid = corelation_id, "PageRequest received");

                        let snapshot = handle.snapshot();
                        let page = snapshot.read(page_no as Addr * PAGE_SIZE as Addr, PAGE_SIZE).into_owned();
                        trace!(page_no, cid = corelation_id, lsn = handle.last_seen_lsn(), "Sending PageReply");
                        Message::PageReply(corelation_id, Cow::Owned(page), snapshot.lsn())
                            .write_to(pin!(&mut socket))
                            .await?;
                    }
                    _ => return io_result("Invalid message"),
                }
            }
        }
    }
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
        return io_result("Invalid handshake response");
    };
    if version != PROTOCOL_VERSION {
        return io_result("Invalid protocol version");
    }

    let (tx, rx) = mpsc::channel(100);
    let driver = NetworkDriver { tx };
    let volume = Volume::new_with_driver(pages, driver);
    let read_handle = volume.handle();

    let join_handle = tokio::spawn(client_worker(socket, volume, rx));

    Ok((read_handle, join_handle))
}

#[instrument(skip(client, volume, rx), fields(addr = %client.peer_addr().unwrap()))]
async fn client_worker(
    mut client: TcpStream,
    mut volume: Volume,
    mut rx: mpsc::Receiver<(PageNo, oneshot::Sender<io::Result<([u8; PAGE_SIZE], LSN)>>)>,
) -> io::Result<()> {
    let mut assembler = PacketAssembler::default();
    let mut inflight_pages = HashMap::new();
    let mut corelation_id = 0;
    let (read, write) = client.split();
    let mut read = pin!(read);
    let mut write = pin!(write);

    let (commit_tx, mut commit_rx) = mpsc::channel(10);

    // We're doing commits in a separate task so not to block Network IO-task from fetching pages
    // from the network.
    thread::spawn(move || {
        while let Some((lsn, redo, undo)) = commit_rx.blocking_recv() {
            let commit = Commit::new(redo, undo, lsn);

            trace!(lsn, "Trying commit...");
            volume.apply_commit(commit).expect("Unable to apply commit");
            // let my_lsn = volume.commit(tx).unwrap();
            trace!(lsn = lsn, "Committed snapshot");
        }
    });

    loop {
        tokio::select! {
            Some((page_no, rx)) = rx.recv() => {
                corelation_id += 1;
                trace!(page = page_no, cid = corelation_id, "Requesting page from network");
                inflight_pages.insert(corelation_id, (page_no, rx));
                Message::PageRequest(corelation_id, page_no).write_to(write.as_mut()).await?;
            }
            msg = Message::read_from(read.as_mut()) => {
                let msg = msg.expect("Unable to read message");
                if let Some(command) = assembler.feed_packet(msg)? {
                    match command {
                        AssembledCommand::Commit(lsn, redo, undo) => {
                            trace!(
                                lsn = lsn,
                                patches = redo.len(),
                                "Received snapshot from master"
                            );
                            commit_tx.send((lsn, redo, undo)).await.expect("Unable to send page to client");
                        }
                        AssembledCommand::Page(corelation_id, lsn, data) => {
                            if let Some((page_no, tx)) = inflight_pages.remove(&corelation_id) {
                                trace!(page_no = page_no, lsn = lsn, "Received page");

                                // TODO: page type should be consistent
                                tx.send(Ok((data.try_into().unwrap(), lsn))).expect("Unable to send page to client");
                            }else{
                                warn!(cid = corelation_id, "Page not requested")
                            }

                        }
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct PacketAssembler {
    redo_patches: Vec<Patch>,
    undo_patches: Vec<Patch>,
}

enum AssembledCommand {
    Commit(LSN, Vec<Patch>, Vec<Patch>),
    Page(u64, LSN, Vec<u8>),
}

impl PacketAssembler {
    fn feed_packet(&mut self, msg: Message) -> io::Result<Option<AssembledCommand>> {
        match msg {
            Message::Patch(redo, undo) => {
                self.redo_patches.push(redo.into_owned());
                self.undo_patches.push(undo.into_owned());
                Ok(None)
            }
            Message::Commit(lsn) => {
                let redo = mem::take(&mut self.redo_patches);
                let undo = mem::take(&mut self.undo_patches);
                Ok(Some(AssembledCommand::Commit(lsn, redo, undo)))
            }
            Message::PageReply(corelation_id, data, lsn) => Ok(Some(AssembledCommand::Page(
                corelation_id,
                lsn,
                data.into_owned(),
            ))),
            _ => io_result("Invalid message type"),
        }
    }
}

pub async fn next_snapshot(mut socket: &mut TcpStream) -> io::Result<(LSN, Vec<Patch>)> {
    let mut patches = vec![];
    loop {
        let msg = Message::read_from(pin!(&mut socket)).await?;
        match msg {
            Message::Patch(p, _) => patches.push(p.into_owned()),
            Message::Commit(lsn) => return Ok((lsn, patches)),
            _ => return io_result("Invalid message type"),
        }
    }
}

struct NetworkDriver {
    tx: mpsc::Sender<(PageNo, oneshot::Sender<io::Result<([u8; PAGE_SIZE], LSN)>>)>,
}

impl PageDriver for NetworkDriver {
    #[instrument(skip(self, page), err, ret(level = "trace"))]
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<LSN> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .blocking_send((page_no, reply_tx))
            .map_err(|_| io_error("Unable to send request"))?;
        let (remote_page, lsn) = reply_rx
            .blocking_recv()
            .map_err(|_| io_error("Unable to recv response"))??;
        page.copy_from_slice(&remote_page);

        Ok(lsn)
    }

    fn write_page(&self, _page_no: PageNo, _page: &[u8; PAGE_SIZE], _lsn: LSN) -> io::Result<()> {
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn io_result<T>(error: &str) -> io::Result<T> {
    Err(io_error(error))
}

pub(crate) fn io_error(error: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
