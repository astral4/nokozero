//! Localhost TCP control channel between the hook and driver.
//!
//! The driver binds one listener per instance and passes its address via `NOKOZERO_CONNECT`. A background thread dials it with
//! a bounded retry and publishes the stream once. After the stream is up, any I/O error, EOF, or protocol violation aborts the process.
//!
//! The game thread drives a lockstep loop. Each RL step, it sends one observation and blocks for one command
//! so the game advances at the driver's step rate.

use crate::Action;
use crate::log::fatal;
use crate::practice::{PARAMS_LEN, PracticeParams, RECORD_LEN, StageRecord};
use std::io::{ErrorKind, Read as _, Write as _};
use std::net::TcpStream;
use std::sync::OnceLock;
use std::thread::{Builder as ThreadBuilder, sleep};
use std::time::Duration;

pub(crate) const MAX_TAPE_FRAMES: usize = 36_000;

/// The largest allowed inbound command body length.
const MAX_CMD: usize = 1 + 1 + 2 * MAX_TAPE_FRAMES;

const _: () = assert!(
    MAX_CMD >= 1 + 4 + PARAMS_LEN + RECORD_LEN,
    "a RESET with a record must fit"
);

/// The largest allowed outbound frame length.
const MAX_OBS: u32 = 1 << 20;

// 250 ms * 40 = 10 s connect deadline. This bounds a transient failure to get a socket out the door.
const CONNECT_RETRY: Duration = Duration::from_millis(250);
const CONNECT_ATTEMPTS: u32 = 40;

// Driver -> hook command tags.
const CMD_RESET: u8 = 0x02;
const CMD_TAPE: u8 = 0x03;

static STREAM: OnceLock<TcpStream> = OnceLock::new();

/// This should be called once during `DLL_PROCESS_ATTACH`.
pub(crate) fn init(addr: String) {
    eprintln!("nokozero_hook::ipc: attached, dialing {addr}");
    if ThreadBuilder::new()
        .name("nokozero-ipc".into())
        .spawn(move || connect(&addr))
        .is_err()
    {
        fatal!("could not spawn the connector thread");
    }
}

/// Dials the driver until it answers, the deadline passes, or the address refuses.
fn connect(addr: &str) {
    let mut last_error = None;
    for _ in 0..CONNECT_ATTEMPTS {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                drop(stream.set_nodelay(true));
                drop(STREAM.set(stream));
                return;
            }
            Err(error) if error.kind() == ErrorKind::ConnectionRefused => {
                fatal!("{addr} refused the connection ({error})");
            }
            Err(error) => last_error = Some(error),
        }
        sleep(CONNECT_RETRY);
    }
    if let Some(error) = last_error {
        fatal!("connect deadline exceeded (last error: {error})");
    }
    fatal!("connect deadline exceeded");
}

pub(crate) fn is_connected() -> bool {
    STREAM.get().is_some()
}

/// A decoded step-ending command.
pub(crate) enum Command {
    Reset {
        seq: u32,
        params: PracticeParams,
        record: Option<Box<StageRecord>>,
    },
    Tape {
        first: Action,
        rest: Vec<Action>,
        raw: bool,
    },
}

/// The receive buffer for one command body.
pub(crate) struct CommandBuf(Vec<u8>);

impl CommandBuf {
    pub(crate) fn new() -> Self {
        Self(vec![0; MAX_CMD])
    }
}

/// An observation frame under construction.
pub(crate) struct ObsFrame<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> ObsFrame<'a> {
    #[must_use]
    pub(crate) fn begin(buf: &'a mut Vec<u8>) -> Self {
        // `u32` length prefix
        const HEADER: usize = 4;

        buf.clear();
        buf.resize(HEADER, 0);
        Self { buf }
    }

    /// Returns the buffer to append the payload to.
    pub(crate) fn payload(&mut self) -> &mut Vec<u8> {
        self.buf
    }

    /// Patches the length over the placeholder and returns the sendable frame.
    fn finish(self) -> &'a [u8] {
        #[expect(clippy::cast_possible_truncation)]
        let len = (self.buf.len() - 4) as u32;
        if len > MAX_OBS {
            fatal!("outbound frame of {len} bytes exceeds MAX_OBS ({MAX_OBS})");
        }
        self.buf[..4].copy_from_slice(&len.to_le_bytes());
        self.buf
    }
}

/// Sends the observation, then blocks until the driver sends a step-ending command and decodes it.
/// Returns `None` before the connection has been established. Aborts on any I/O error or protocol violation.
pub(crate) fn step(obs: ObsFrame<'_>, buf: &mut CommandBuf) -> Option<Command> {
    let stream = STREAM.get()?;

    let mut writer = stream;
    if writer
        .write_all(obs.finish())
        .and_then(|()| writer.flush())
        .is_err()
    {
        fatal!("send failed");
    }

    let body = &mut buf.0;
    let Some(len) = recv_frame(stream, body) else {
        fatal!("recv failed");
    };
    match decode(&body[..len]) {
        Ok(command) => Some(command),
        Err(violation) => fatal!("{violation}"),
    }
}

/// Decodes a command body (i.e. its tag byte and payload). Returns `Err(_)` with a description if there is a protocol violation.
fn decode(body: &[u8]) -> Result<Command, &'static str> {
    let Some((&tag, payload)) = body.split_first() else {
        return Err("empty command");
    };
    match tag {
        CMD_RESET => decode_reset(payload),
        CMD_TAPE => decode_tape(payload),
        _ => Err("unknown command tag"),
    }
}

fn decode_reset(payload: &[u8]) -> Result<Command, &'static str> {
    let (seq, rest) = payload.split_first_chunk().ok_or("bad RESET length")?;
    let seq = u32::from_le_bytes(*seq);
    let (params, blob) = rest
        .split_first_chunk::<PARAMS_LEN>()
        .ok_or("bad RESET length")?;
    let params = PracticeParams::parse(params).ok_or("RESET params invalid")?;
    let record = if blob.is_empty() {
        None
    } else {
        let record = <&[u8; RECORD_LEN]>::try_from(blob).map_err(|_| "bad RESET length")?;
        Some(Box::new(StageRecord(*record)))
    };
    if record
        .as_deref()
        .is_some_and(|record| !record.fits(&params))
    {
        return Err("RESET record is not for its warp's stage");
    }
    Ok(Command::Reset {
        seq,
        params,
        record,
    })
}

fn decode_tape(payload: &[u8]) -> Result<Command, &'static str> {
    let (&raw, keys) = payload.split_first().ok_or("bad TAPE length")?;
    let raw = match raw {
        0 => false,
        1 => true,
        _ => return Err("bad TAPE raw byte"),
    };
    let (words, []) = keys.as_chunks::<2>() else {
        return Err("bad TAPE length");
    };
    if words.len() > MAX_TAPE_FRAMES {
        return Err("bad TAPE length");
    }
    let [first, rest @ ..] = words else {
        return Err("bad TAPE length");
    };
    let action = |word: &[u8; 2]| {
        Action::from_wire(u32::from(u16::from_le_bytes(*word)))
            .ok_or("TAPE frame outside the action mask")
    };
    Ok(Command::Tape {
        first: action(first)?,
        rest: rest.iter().map(action).collect::<Result<_, _>>()?,
        raw,
    })
}

/// Fills the front of `body` with the entire frame body (tag byte + payload) and returns its length, or `None` on disconnect/desync.
fn recv_frame(stream: &TcpStream, body: &mut [u8]) -> Option<usize> {
    let mut reader = stream;
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes).ok()?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len == 0 || len > body.len() {
        return None;
    }
    reader.read_exact(&mut body[..len]).ok()?;
    Some(len)
}

#[cfg(test)]
mod tests {
    use super::{CMD_RESET, CMD_TAPE, Command, MAX_TAPE_FRAMES, decode};
    use crate::practice::test_support::wire_bytes;
    use crate::practice::{PARAMS_LEN, RECORD_LEN};

    fn tape(raw: u8, words: &[u16]) -> Vec<u8> {
        let mut body = vec![CMD_TAPE, raw];
        for word in words {
            body.extend_from_slice(&word.to_le_bytes());
        }
        body
    }

    fn reset(params: &[u8], record: &[u8]) -> Vec<u8> {
        let mut body = vec![CMD_RESET];
        body.extend_from_slice(&9u32.to_le_bytes());
        body.extend_from_slice(params);
        body.extend_from_slice(record);
        body
    }

    #[test]
    fn tape_command_decode() {
        let Ok(Command::Tape { first, rest, raw }) = decode(&tape(0, &[0x1, 0x11, 0x9])) else {
            panic!("a valid tape");
        };
        assert_eq!(
            (first.0, rest.iter().map(|a| a.0).collect::<Vec<_>>(), raw),
            (0x1, vec![0x11, 0x9], false)
        );
        let Ok(Command::Tape { first, rest, raw }) = decode(&tape(1, &[0x200])) else {
            panic!("a valid raw tape");
        };
        assert_eq!((first.0, rest.len(), raw), (0x200, 0, true));
    }

    #[test]
    fn tape_command_violation() {
        assert_eq!(decode(&[CMD_TAPE]).err(), Some("bad TAPE length"));
        assert_eq!(decode(&tape(0, &[])).err(), Some("bad TAPE length"));
        let mut odd = tape(0, &[0x1]);
        odd.push(0);
        assert_eq!(decode(&odd).err(), Some("bad TAPE length"));
        assert!(decode(&tape(0, &vec![0; MAX_TAPE_FRAMES])).is_ok());
        assert_eq!(
            decode(&tape(0, &vec![0; MAX_TAPE_FRAMES + 1])).err(),
            Some("bad TAPE length")
        );
        assert_eq!(decode(&tape(2, &[0x1])).err(), Some("bad TAPE raw byte"));
        assert_eq!(
            decode(&tape(0, &[0x100])).err(),
            Some("TAPE frame outside the action mask")
        );
        assert_eq!(decode(&[0x01]).err(), Some("unknown command tag"));
        assert_eq!(decode(&[]).err(), Some("empty command"));
    }

    #[test]
    fn reset_command_record() {
        let params = wire_bytes(1202, 1, 0);
        let Ok(Command::Reset { seq, record, .. }) = decode(&reset(&params, &[])) else {
            panic!("a valid reset");
        };
        assert_eq!((seq, record.is_none()), (9, true));
        assert_eq!(
            decode(&reset(&params, &[0])).err(),
            Some("bad RESET length")
        );
        assert_eq!(
            decode(&reset(&params[..PARAMS_LEN - 1], &[])).err(),
            Some("bad RESET length")
        );
        let mut stage_1 = [1; RECORD_LEN];
        stage_1[1] = 0;
        let Ok(Command::Reset { record, .. }) = decode(&reset(&params, &stage_1)) else {
            panic!("a valid reset with its record");
        };
        assert_eq!(record.map(|record| record.0[0]), Some(1));
        assert_eq!(
            decode(&reset(&params, &stage_1[1..])).err(),
            Some("bad RESET length")
        );
        assert_eq!(decode(&[CMD_RESET, 1, 2]).err(), Some("bad RESET length"));
    }

    #[test]
    fn record_warp_to_own_stage() {
        let mut record = [0; RECORD_LEN];
        record[0] = 1;
        let warp = wire_bytes(1202, 1, 0);
        assert!(decode(&reset(&warp, &record)).is_ok());
        let violation = Some("RESET record is not for its warp's stage");
        let vanilla = wire_bytes(1202, 0, 0);
        assert_eq!(decode(&reset(&vanilla, &record)).err(), violation);
        record[0] = 2;
        assert_eq!(decode(&reset(&warp, &record)).err(), violation);
    }
}
