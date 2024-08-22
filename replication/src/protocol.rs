use crate::io_error;
use pmem::page::{Patch, LSN, PAGE_SIZE};
use std::{borrow::Cow, io, pin::Pin};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, PartialEq)]
pub enum Message<'a> {
    // Hello message from client, contains protocol version
    ClientHello(u8),

    // Hello message from server, contains protocol version and number of pages
    ServerHello(u8, u32),
    Patch(Cow<'a, Patch>),
    Commit(LSN),
}

pub const PROTOCOL_VERSION: u8 = 1;

const CLIENT_HELLO: u8 = 1;
const SERVER_HELLO: u8 = 2;
const PATCH: u8 = 3;
const COMMIT: u8 = 4;

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
            Message::Patch(p) => {
                out.write_u8(PATCH).await?;
                match p.as_ref() {
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
            }
            Message::Commit(lsn) => {
                out.write_u8(COMMIT).await?;
                out.write_u64(*lsn).await?;
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
                        let mut bytes = vec![0; len];
                        input.read_exact(&mut bytes).await?;
                        Ok(Message::Patch(Cow::Owned(Patch::Write(addr, bytes))))
                    } else {
                        io_error("Patch length exceeds page size")
                    }
                }
                PATCH_RECLAIM => {
                    let addr = input.read_u64().await?;
                    let len = usize::try_from(input.read_u64().await?).unwrap();
                    Ok(Message::Patch(Cow::Owned(Patch::Reclaim(addr, len))))
                }
                _ => io_error("Invalid patch type"),
            },
            COMMIT => {
                let lsn = input.read_u64().await?;
                Ok(Message::Commit(lsn))
            }
            _ => io_error("Invalid discriminator"),
        }
    }
}

#[cfg(test)]
mod tests {
    use io::Cursor;
    use std::pin::pin;

    use super::*;
    #[tokio::test]
    async fn check_serialization() -> io::Result<()> {
        assert_read_write_eq(Message::ClientHello(10)).await?;
        assert_read_write_eq(Message::ServerHello(10, 20)).await?;

        let write = Patch::Write(10, vec![0, 1, 2, 3]);
        let reclaim = Patch::Reclaim(20, 10);
        assert_read_write_eq(Message::Patch(Cow::Owned(write))).await?;
        assert_read_write_eq(Message::Patch(Cow::Owned(reclaim))).await?;

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
