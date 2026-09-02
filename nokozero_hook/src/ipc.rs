//! Localhost TCP control channel between the hook and driver.
//!
//! The driver binds one listener per instance and passes its address via `NOKOZERO_CONNECT`. A background thread dials it with
//! a bounded retry and publishes the stream once. After the stream is up, any I/O error, EOF, or protocol violation aborts the process.
//!
//! The game thread drives a lockstep loop. Each RL step, it sends one observation and blocks for one command
//! so the game advances at the driver's step rate. A TAPE command supplies one input per frame for many frames
//! and ends the step like an ACT; the next observation follows the tape's last frame.

use crate::Action;
use crate::log::fatal;
use crate::practice::{FLAG_RECORD, PARAMS_LEN, PracticeParams, RECORD_LEN, StageRecord};
use std::io::{ErrorKind, Read as _, Write as _};
use std::net::TcpStream;
use std::sync::OnceLock;
use std::thread::{Builder as ThreadBuilder, sleep};
use std::time::Duration;

pub(crate) const MAX_TAPE_FRAMES: usize = 36_000;

/// The largest allowed inbound command body length.
const MAX_CMD: usize = 1 + 4 + 2 * MAX_TAPE_FRAMES;

const _: () = assert!(
    MAX_CMD >= 1 + 4 + PARAMS_LEN + RECORD_LEN,
    "a RESET with a record must fit"
);

/// The largest allowed outbound frame length.
const MAX_OBS: u32 = 1 << 20;

// 250 ms * 40 = 10 s connect deadline. This bounds a transient failure to get a socket out the door.
const CONNECT_RETRY: Duration = Duration::from_millis(250);
const CONNECT_ATTEMPTS: u32 = 40;

// Hook -> driver message tags.
const MSG_OBS: u8 = 0x01;

// Driver -> hook command tags.
const CMD_ACT: u8 = 0x01;
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
    Act(Action),
    Reset {
        seq: u32,
        params: PracticeParams,
        /// `Some(_)` iff the params carry `FLAG_RECORD`.
        record: Option<Box<StageRecord>>,
    },
    /// One action per frame, applied on the live stage frames that follow.
    Tape(Vec<Action>),
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
        // `u32` length prefix + tag byte
        const HEADER: usize = 5;

        buf.clear();
        buf.resize(HEADER, 0);
        Self { buf }
    }

    /// Returns the buffer to append the payload to.
    pub(crate) fn payload(&mut self) -> &mut Vec<u8> {
        self.buf
    }

    /// Patches the length/tag header over the placeholder and returns the sendable frame.
    fn finish(self, tag: u8) -> &'a [u8] {
        #[expect(clippy::cast_possible_truncation)]
        let len = (self.buf.len() - 4) as u32; // tag + payload
        if len > MAX_OBS {
            fatal!("outbound frame of {len} bytes exceeds MAX_OBS ({MAX_OBS})");
        }
        self.buf[..4].copy_from_slice(&len.to_le_bytes());
        self.buf[4] = tag;
        self.buf
    }
}

/// Sends the observation, then blocks until the driver sends a step-ending command and decodes it.
/// Returns `None` before the connection has been established. Aborts on any I/O error or protocol violation.
pub(crate) fn step(obs: ObsFrame<'_>, buf: &mut CommandBuf) -> Option<Command> {
    let stream = STREAM.get()?;

    let mut writer = stream;
    if writer
        .write_all(obs.finish(MSG_OBS))
        .and_then(|()| writer.flush())
        .is_err()
    {
        fatal!("send failed");
    }

    let body = &mut buf.0;
    let payload = if let Some(len) = recv_frame(stream, body) {
        &body[1..len]
    } else {
        fatal!("recv failed");
    };

    match body[0] {
        CMD_ACT => {
            let Ok(bytes) = <[u8; 4]>::try_from(payload) else {
                fatal!("bad ACT length");
            };
            let Some(action) = Action::from_wire(u32::from_le_bytes(bytes)) else {
                fatal!("ACT outside the action mask");
            };
            Some(Command::Act(action))
        }
        CMD_RESET => {
            let Some((seq, rest)) = payload.split_at_checked(4) else {
                fatal!("bad RESET length");
            };
            let seq = u32::from_le_bytes(seq.try_into().unwrap());
            let Some((params, blob)) = rest.split_at_checked(PARAMS_LEN) else {
                fatal!("bad RESET length");
            };
            let Some(params) = PracticeParams::parse(params) else {
                fatal!("RESET params invalid");
            };
            let record = if params.has_flag(FLAG_RECORD) {
                let Ok(record) = <&[u8; RECORD_LEN]>::try_from(blob) else {
                    fatal!("RESET with FLAG_RECORD must carry a {RECORD_LEN}-byte record");
                };
                Some(Box::new(StageRecord(*record)))
            } else {
                if !blob.is_empty() {
                    fatal!("bad RESET length");
                }
                None
            };
            Some(Command::Reset {
                seq,
                params,
                record,
            })
        }
        CMD_TAPE => {
            let Some((count, keys)) = payload.split_at_checked(4) else {
                fatal!("bad TAPE length");
            };
            let count = u32::from_le_bytes(count.try_into().unwrap()) as usize;
            if count == 0 || count > MAX_TAPE_FRAMES || keys.len() != 2 * count {
                fatal!("bad TAPE length");
            }
            let tape = keys
                .chunks_exact(2)
                .map(|k| {
                    let bits = u32::from(u16::from_le_bytes([k[0], k[1]]));
                    Action::from_wire(bits)
                        .unwrap_or_else(|| fatal!("TAPE frame outside the action mask"))
                })
                .collect();
            Some(Command::Tape(tape))
        }
        _ => fatal!("unknown command tag"),
    }
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
