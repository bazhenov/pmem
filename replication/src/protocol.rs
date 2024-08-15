use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use pmem::page::{Patch, PAGE_SIZE};
use std::io::{self, Read, Write};

#[derive(Debug, PartialEq)]
pub enum Message {
    Handshake,
    Patch(Patch),
}

const HANDSHAKE: u8 = 1;
const PATCH: u8 = 2;

const PATCH_WRITE: u8 = 1;
const PATCH_RECLAIM: u8 = 2;

impl Message {
    pub fn write_to(&self, cursor: &mut impl Write) -> io::Result<()> {
        match self {
            Message::Handshake => cursor.write_all(&[HANDSHAKE])?,
            Message::Patch(p) => {
                cursor.write_all(&[PATCH])?;
                match p {
                    Patch::Write(addr, bytes) => {
                        cursor.write_u8(PATCH_WRITE)?;
                        cursor.write_u64::<BigEndian>(*addr)?;
                        cursor.write_u64::<BigEndian>(bytes.len() as u64)?;
                        cursor.write_all(bytes)?;
                    }
                    Patch::Reclaim(addr, length) => {
                        cursor.write_u8(PATCH_RECLAIM)?;
                        cursor.write_u64::<BigEndian>(*addr)?;
                        cursor.write_u64::<BigEndian>(*length as u64)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn read_from(input: &mut impl io::Read) -> io::Result<Self> {
        let discriminator = read(input)?;

        match discriminator {
            HANDSHAKE => Ok(Message::Handshake),
            PATCH => match read(input)? {
                PATCH_WRITE => {
                    let addr = input.read_u64::<BigEndian>()?;
                    let len = usize::try_from(input.read_u64::<BigEndian>()?).unwrap();
                    if len <= PAGE_SIZE {
                        let mut bytes = vec![0; len];
                        input.read_exact(&mut bytes)?;
                        Ok(Message::Patch(Patch::Write(addr, bytes)))
                    } else {
                        io_error("Patch length exceeds page size")
                    }
                }
                PATCH_RECLAIM => {
                    let addr = input.read_u64::<BigEndian>()?;
                    let len = usize::try_from(input.read_u64::<BigEndian>()?).unwrap();
                    Ok(Message::Patch(Patch::Reclaim(addr, len)))
                }
                _ => io_error("Invalid patch type"),
            },
            _ => io_error("Invalid discriminator"),
        }
    }
}

fn io_error(error: &str) -> Result<Message, io::Error> {
    Err(io::Error::new(io::ErrorKind::InvalidData, error))
}

fn read(input: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[cfg(test)]
mod tests {
    use io::Cursor;

    use super::*;
    #[test]
    fn check_serialization() -> io::Result<()> {
        assert_read_write_eq(Message::Handshake)?;

        assert_read_write_eq(Message::Patch(Patch::Write(10, vec![0, 1, 2, 3])))?;
        assert_read_write_eq(Message::Patch(Patch::Reclaim(20, 10)))?;
        Ok(())
    }

    #[track_caller]
    fn assert_read_write_eq(msg: Message) -> Result<(), io::Error> {
        let mut cursor = Cursor::new(Vec::new());
        msg.write_to(&mut cursor)?;
        // Writing garbage to the end of the buffer to ensure that we are not reading past the end
        cursor.write_all(&[0; 10])?;
        cursor.set_position(0);
        let msg_copy = Message::read_from(&mut cursor)?;
        assert_eq!(msg, msg_copy);
        Ok(())
    }
}
