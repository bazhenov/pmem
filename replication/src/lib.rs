use pmem::{
    driver::PageDriver,
    volume::{
        Addr, Commit, CommitNotify, PageNo, Patch, TxRead, Volume, VolumeHandle, LSN, PAGE_SIZE,
    },
};
use protocol::{Message, PROTOCOL_VERSION};
use std::{
    borrow::Cow, collections::HashMap, fmt::Debug, io, mem, net::SocketAddr, pin::pin, sync::Arc,
    thread,
};
use tokio::{
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::{self, mpsc},
    task::JoinHandle,
};
use tracing::{debug, info, instrument, span, trace, warn, Level};

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

        tokio::spawn(server_worker(socket, handle.commit_notify()));
    }
}

#[instrument(skip(socket, notify), fields(addr = %socket.peer_addr().unwrap()))]
async fn server_worker(mut socket: TcpStream, notify: CommitNotify) -> io::Result<()> {
    Message::ServerHello(PROTOCOL_VERSION, notify.pages())
        .write_to(pin!(&mut socket))
        .await?;
    let msg = Message::read_from(pin!(&mut socket)).await?;
    let Message::ClientHello(v) = msg else {
        return io_result("Invalid handshake message");
    };
    if v != PROTOCOL_VERSION {
        return io_result("Invalid protocol version");
    }

    let mut socket = pin!(socket);

    // Keeping track of current snapshot that is updated on every commit.
    // We need it to send consistent page snapshots to the client, because clients rely on happens-before
    // guarantees between commits and page snapshots. Meaning if client receives commit with LSN 10, it should
    // receive only page snapshot with LSN >= 10 after that. Otherwise, it might not be able to recover
    // page using undo logs at a given LSN that is <10.
    let mut current_snapshot = notify.snapshot();

    // Sending initial commit to the client
    {
        let initial_commit = notify.commit();
        let lsn = initial_commit.lsn();
        let redo = initial_commit.patches();
        let undo = initial_commit.undo();
        trace!(lsn, patches = redo.len(), "Starting log relay");
        for (r, u) in redo.iter().zip(undo.iter()) {
            Message::Patch(Cow::Borrowed(r), Cow::Borrowed(u))
                .write_to(pin!(&mut socket))
                .await?;
        }
        Message::Commit(lsn).write_to(socket.as_mut()).await?;
    };

    let (commit_tx, mut commit_rx) = sync::mpsc::channel(10);
    {
        let mut notify = notify.clone();
        thread::spawn(move || loop {
            let mut result = Ok(());
            while result.is_ok() {
                let commit = Arc::clone(notify.next_commit());
                let snapshot = notify.snapshot();
                trace!(lsn = commit.lsn(), "Got next commit");
                result = commit_tx.blocking_send((commit, snapshot));
            }
        });
    }

    loop {
        tokio::select! {
            next_commit = commit_rx.recv() => {
                if let Some((commit, snapshot)) = next_commit {
                    debug_assert!(commit.lsn() == snapshot.lsn(), "LSN mismatch");
                    let redo = commit.patches();
                    let undo = commit.undo();
                    let lsn = commit.lsn();
                    trace!(lsn = lsn, patches = redo.len(), "Sending commit log");
                    for (r, u) in redo.iter().zip(undo.iter()) {
                        Message::Patch(Cow::Borrowed(r), Cow::Borrowed(u))
                            .write_to(pin!(&mut socket))
                            .await?;
                    }
                    Message::Commit(lsn).write_to(socket.as_mut()).await?;
                    current_snapshot = snapshot;
                } else {
                    info!("No more snapshots");
                    return Ok(());
                }
            }
            msg = Message::read_from(socket.as_mut()) => {
                match msg? {
                    Message::PageRequest(corelation_id, page_no) => {
                        trace!(page_no, cid = corelation_id, "PageRequest received");

                        let page = current_snapshot.read(page_no as Addr * PAGE_SIZE as Addr, PAGE_SIZE).into_owned();
                        let lsn = current_snapshot.lsn();
                        trace!(page_no, cid = corelation_id, lsn, "Sending PageReply");
                        Message::PageReply(corelation_id, Cow::Owned(page), lsn)
                            .write_to(socket.as_mut())
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

    let (tx, rx) = request_reply::channel();
    let driver = NetworkDriver { tx };

    // Receiving initial commit from the server
    let volume = {
        let mut assembler = PacketAssembler::default();
        loop {
            let msg = Message::read_from(pin!(&mut socket)).await?;
            if let Some(command) = assembler.feed_packet(msg)? {
                let AssembledCommand::Commit(lsn, redo, undo) = command else {
                    return io_result("Invalid message: initial commit expected");
                };
                debug!(lsn, "Received initial commit from server");
                break if lsn == 0 {
                    // Don't need to do nothing, because master has not commits (the volume is empty)
                    // Need to check it is indeed the empty initial commit
                    assert!(redo.is_empty(), "Initial commit must be empty");
                    Volume::new_with_driver(pages, driver)
                } else {
                    let commit = Commit::new(redo, undo, lsn);
                    Volume::from_commit(pages, commit, driver)
                };
            }
        }
    };

    debug!("Replicated volume created");

    let read_handle = volume.handle();
    let join_handle = tokio::spawn(client_worker(socket, volume, rx));

    Ok((read_handle, join_handle))
}

#[instrument(skip(client, volume, rx), fields(addr = %client.peer_addr().unwrap()))]
async fn client_worker(
    mut client: TcpStream,
    mut volume: Volume,
    mut rx: request_reply::RequestReceiver<PageNo, PageAndLsn>,
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
        let span = span!(Level::TRACE, "commit_applier");
        let _ = span.enter();

        while let Some((lsn, redo, undo)) = commit_rx.blocking_recv() {
            let commit = Commit::new(redo, undo, lsn);

            // TODO in case of error we should fail the main thread
            volume.apply_commit(commit).expect("Unable to apply commit");
            info!(lsn, "Commit applied to volume");
        }
    });

    loop {
        tokio::select! {
            Some((page_no, reply)) = rx.recv() => {
                corelation_id += 1;
                trace!(page = page_no, cid = corelation_id, "Requesting page from network");
                inflight_pages.insert(corelation_id, (page_no, reply));
                Message::PageRequest(corelation_id, page_no).write_to(write.as_mut()).await?;
            }
            msg = Message::read_from(read.as_mut()) => {
                if let Some(command) = assembler.feed_packet(msg?)? {
                    match command {
                        AssembledCommand::Commit(lsn, redo, undo) => {
                            trace!(
                                lsn = lsn,
                                patches = redo.len(),
                                "Received snapshot from master"
                            );
                            commit_tx.send((lsn, redo, undo)).await.expect("Commit loop is broken");
                        }
                        AssembledCommand::Page(corelation_id, lsn, data) => {
                            if let Some((page_no, tx)) = inflight_pages.remove(&corelation_id) {
                                // TODO: page type should be consistent
                                let _ = tx.reply(Ok((data.try_into().unwrap(), lsn)));
                                trace!(page_no = page_no, lsn = lsn, "Received page");
                            } else {
                                warn!(cid = corelation_id, "Spourious page detected")
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

type PageAndLsn = io::Result<(Box<[u8; PAGE_SIZE]>, LSN)>;

struct NetworkDriver {
    tx: request_reply::Sender<PageNo, PageAndLsn>,
}

impl PageDriver for NetworkDriver {
    #[instrument(skip(self, page), err, ret(level = "trace"))]
    fn read_page(&self, page_no: PageNo, page: &mut [u8; PAGE_SIZE]) -> io::Result<Option<LSN>> {
        let (remote_page, lsn) = self.tx.blocking_send(page_no)??;
        page.copy_from_slice(remote_page.as_ref());

        // TODO there can be be None
        Ok(Some(lsn))
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

/// Simple request-reply channel. An abstraction over one mpsc channel. and a oneshot channel that is
/// used to send the reply back.
///
/// Can be used in a following way:
/// ```no_run
/// # use tokio;
/// use replication::request_reply;
///
/// let (sender, mut receiver) = request_reply::channel::<u32, String>();
///
/// // Spawn a task to handle requests
/// tokio::spawn(async move {
///     while let Some((request, reply)) = receiver.recv().await {
///         let response = format!("Processed request: {}", request);
///         let _ = reply.reply(response);
///     }
/// });
///
/// // Send a request and get the response
/// let response = sender.blocking_send(42).unwrap();
/// println!("Received response: {}", response);
/// ```
pub mod request_reply {
    use super::io_error;
    use std::io;
    use tokio::sync::{mpsc, oneshot};

    pub fn channel<Rq, Rp>() -> (Sender<Rq, Rp>, RequestReceiver<Rq, Rp>) {
        let (tx, rx) = mpsc::channel(1);
        (Sender { tx }, RequestReceiver { rx })
    }

    pub struct RequestReceiver<Rq, Rp> {
        rx: mpsc::Receiver<(Rq, Reply<Rp>)>,
    }

    impl<Rq, Rp> RequestReceiver<Rq, Rp> {
        pub async fn recv(&mut self) -> Option<(Rq, Reply<Rp>)> {
            self.rx.recv().await
        }
    }

    pub struct Sender<Rq, Rp> {
        tx: mpsc::Sender<(Rq, Reply<Rp>)>,
    }

    impl<Rq, Rp> Sender<Rq, Rp> {
        pub fn blocking_send(&self, request: Rq) -> io::Result<Rp> {
            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .blocking_send((request, Reply { reply_tx }))
                .map_err(|_| io_error("Unable to send request"))?;
            let response = reply_rx
                .blocking_recv()
                .map_err(|_| io_error("Unable to receive response"))?;
            Ok(response)
        }
    }

    pub struct Reply<Rp> {
        reply_tx: oneshot::Sender<Rp>,
    }

    impl<Rp> Reply<Rp> {
        pub fn reply(self, response: Rp) -> Result<(), Rp> {
            self.reply_tx.send(response)
        }
    }
}
