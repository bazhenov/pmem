use crate::io_result;
use pmem::volume::{MemRange, PageNo, Patch, LSN, PAGE_SIZE};
use std::{borrow::Cow, io, pin::Pin};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, PartialEq)]
pub enum Message<'a> {
    // Hello message from client, contains protocol version
    ClientHello(u8),

    // Hello message from server, contains protocol version and number of pages
    ServerHello(u8, u32),
    // Redo and undo patch
    Patch(Cow<'a, Patch>, Cow<'a, Patch>),
    Commit(LSN),

    PageRequest(u64, PageNo),
    PageReply(u64, Cow<'a, Box<[u8; PAGE_SIZE]>>, LSN),
}

pub const PROTOCOL_VERSION: u8 = 1;

const CLIENT_HELLO: u8 = 1;
const SERVER_HELLO: u8 = 2;
const PATCH: u8 = 3;
const COMMIT: u8 = 4;
const PAGE_REQUEST: u8 = 5;
const PAGE_REPLY: u8 = 6;

const PATCH_WRITE: u8 = 1;
const PATCH_RECLAIM: u8 = 2;

impl<'a> Message<'a> {
    pub async fn write_to(&self, mut out: Pin<&mut impl AsyncWriteExt>) -> io::Result<()> {
        match self {
            Message::ClientHello(version) => {
                out.write_u8(CLIENT_HELLO).await?;
                out.write_u8(*version).await?;
            }
            Message::ServerHello(version, pages) => {
                out.write_u8(SERVER_HELLO).await?;
                out.write_u8(*version).await?;
                out.write_u32(*pages).await?;
            }
            Message::Patch(r, u) => {
                out.write_u8(PATCH).await?;
                let Patch::Write(addr, undo) = u.as_ref() else {
                    panic!("Incorrect undo patch");
                };
                assert_eq!(*addr, r.as_ref().addr(), "Mismatched patch addresses");
                assert_eq!(undo.len(), r.as_ref().len(), "Mismatched patch length");
                match r.as_ref() {
                    Patch::Write(addr, bytes) => {
                        out.write_u8(PATCH_WRITE).await?;
                        out.write_u64(*addr).await?;
                        out.write_u64(bytes.len() as u64).await?;
                        out.write_all(bytes).await?;
                    }
                    Patch::Reclaim(addr, length) => {
                        out.write_u8(PATCH_RECLAIM).await?;
                        out.write_u64(*addr).await?;
                        out.write_u64(*length as u64).await?;
                    }
                }
                out.write_all(undo).await?;
            }
            Message::Commit(lsn) => {
                out.write_u8(COMMIT).await?;
                out.write_u64(*lsn).await?;
            }
            Message::PageRequest(correlation_id, page_no) => {
                out.write_u8(PAGE_REQUEST).await?;
                out.write_u64(*correlation_id).await?;
                out.write_u32(*page_no).await?;
            }
            Message::PageReply(corelation_id, data, lsn) => {
                out.write_u8(PAGE_REPLY).await?;
                out.write_u64(*corelation_id).await?;
                out.write_u64(*lsn).await?;
                out.write_all(data.as_ref().as_ref()).await?;
            }
        }
        Ok(())
    }

    pub async fn read_from(mut input: Pin<&mut impl AsyncReadExt>) -> io::Result<Self> {
        let discriminator = input.read_u8().await?;
        match discriminator {
            CLIENT_HELLO => {
                let version = input.read_u8().await?;
                Ok(Message::ClientHello(version))
            }
            SERVER_HELLO => {
                let version = input.read_u8().await?;
                let pages = input.read_u32().await?;
                Ok(Message::ServerHello(version, pages))
            }
            PATCH => match input.read_u8().await? {
                PATCH_WRITE => {
                    let addr = input.read_u64().await?;
                    let len = usize::try_from(input.read_u64().await?).unwrap();
                    if len <= PAGE_SIZE {
                        let mut redo = vec![0; len];
                        input.read_exact(&mut redo).await?;
                        let redo_patch = Patch::Write(addr, redo);

                        let mut undo = vec![0; len];
                        input.read_exact(&mut undo).await?;
                        let undo_patch = Patch::Write(addr, undo);

                        Ok(Message::Patch(
                            Cow::Owned(redo_patch),
                            Cow::Owned(undo_patch),
                        ))
                    } else {
                        io_result("Patch length exceeds page size")
                    }
                }
                PATCH_RECLAIM => {
                    let addr = input.read_u64().await?;
                    let len = usize::try_from(input.read_u64().await?).unwrap();
                    let redo = Patch::Reclaim(addr, len);

                    let mut undo = vec![0; len];
                    input.read_exact(&mut undo).await?;
                    let undo_patch = Patch::Write(addr, undo);

                    Ok(Message::Patch(Cow::Owned(redo), Cow::Owned(undo_patch)))
                }
                _ => io_result("Invalid patch type"),
            },
            COMMIT => {
                let lsn = input.read_u64().await?;
                Ok(Message::Commit(lsn))
            }

            PAGE_REQUEST => {
                let corelation_id = input.read_u64().await?;
                let page_no = input.read_u32().await?;
                Ok(Message::PageRequest(corelation_id, page_no))
            }
            PAGE_REPLY => {
                let corelation_id = input.read_u64().await?;
                let lsn = input.read_u64().await?;
                let mut data = Box::new([0; PAGE_SIZE]);
                input.read_exact(data.as_mut_slice()).await?;
                Ok(Message::PageReply(corelation_id, Cow::Owned(data), lsn))
            }
            _ => io_result("Invalid discriminator"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use io::Cursor;
    use std::pin::pin;

    #[tokio::test]
    async fn check_serialization() -> io::Result<()> {
        assert_read_write_eq(Message::ClientHello(10)).await?;
        assert_read_write_eq(Message::ServerHello(10, 20)).await?;

        let write = Patch::Write(10, vec![0, 1, 2, 3]);
        let reclaim = Patch::Reclaim(10, 4);
        let undo = Patch::Write(10, vec![0, 1, 2, 3]);
        assert_read_write_eq(Message::Patch(Cow::Owned(write), Cow::Borrowed(&undo))).await?;
        assert_read_write_eq(Message::Patch(Cow::Owned(reclaim), Cow::Borrowed(&undo))).await?;

        assert_read_write_eq(Message::PageRequest(100, 42)).await?;
        let data = Box::new([42; PAGE_SIZE]);
        assert_read_write_eq(Message::PageReply(100, Cow::Owned(data), 60)).await?;

        assert_read_write_eq(Message::Commit(100)).await?;
        Ok(())
    }

    async fn assert_read_write_eq(msg: Message<'_>) -> Result<(), io::Error> {
        let mut cursor = Cursor::new(Vec::new());
        msg.write_to(pin!(&mut cursor)).await?;
        // Writing garbage to the end of the buffer to ensure that we are not reading past the end
        cursor.write_all(&[0; 10]).await?;
        cursor.set_position(0);
        let msg_copy = Message::read_from(pin!(cursor)).await?;
        assert_eq!(msg, msg_copy);
        Ok(())
    }
}
