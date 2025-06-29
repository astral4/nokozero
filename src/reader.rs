use anyhow::{Context, Result, anyhow};
use nix::errno::Errno;
use nix::unistd::Pid;
use std::fs::read_to_string;
use std::io::IoSliceMut;
use tap::Pipe;
use zerocopy::{FromBytes, Immutable, KnownLayout};

#[cfg(target_os = "linux")]
use nix::sys::uio::{RemoteIoVec, process_vm_readv};

// Offsets/locations for certain data in process memory
const PLAYER_PTR: usize = 0xe9bb8;
const PLAYER_POS: usize = 0x618;
const PLAYER_IS_FOCUSED: usize = 0x16240;
const PLAYER_HITBOX_RADIUS: usize = 0x2bfc8;

#[derive(Debug)]
pub struct StateReader {
    pid: Pid,
    base_addr: usize,
    ptrs: GameData<1>,
    data: GameData<3>,
}

#[derive(Debug)]
pub struct GameState {
    pub player: PlayerState,
}

#[derive(Debug)]
pub struct PlayerState {
    pub pos_x: f32,
    pub pos_y: f32,
    pub hitbox_radius: f32,
    pub is_focused: bool,
}

impl StateReader {
    /// Instantiates the game state reader based on the provided game process PID.
    ///
    /// # Errors
    /// This function returns an error if:
    /// - `/proc/<pid>/maps` could not be read
    /// - `/proc/<pid>/maps` was not in the expected format and could not be parsed
    pub fn new(pid: Pid) -> Result<Self> {
        let maps_path = format!("/proc/{pid}/maps");

        let base_addr = read_to_string(&maps_path)
            .with_context(|| format!("failed to read process memory maps from {maps_path:?}"))?
            .lines()
            .find(|line| line.contains("th15.exe"))
            .ok_or_else(|| anyhow!("Game executable not found in process memory maps"))?
            .split_whitespace()
            .next()
            .with_context(|| format!("unexpected format of {maps_path:?}"))?
            .split('-')
            .next()
            .with_context(|| format!("failed to parse address range from {maps_path:?}"))?
            .pipe(|addr| usize::from_str_radix(addr, 16))
            .with_context(|| format!("failed to parse base address from {maps_path:?}"))?;

        let ptrs = GameData::new([4]);

        let data = GameData::new([8, 4, 4]);

        Ok(Self {
            pid,
            base_addr,
            ptrs,
            data,
        })
    }

    /// Gets the current game state, including:
    /// - player position, focus state, and hitbox radius
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    pub fn get_state(&mut self) -> Result<Option<GameState>> {
        #[rustfmt::skip]
        let ptr_locations = [
            RemoteIoVec { base: self.base_addr + PLAYER_PTR, len: 4 }
        ];

        read(self.pid, self.ptrs.as_io_slices_mut(), &ptr_locations)?;

        let player_ptr = *self.ptrs.get::<u32>(0) as usize;

        if player_ptr == 0 {
            return Ok(None);
        }

        #[rustfmt::skip]
        let locations = [
            RemoteIoVec { base: player_ptr + PLAYER_POS, len: 8 },
            RemoteIoVec { base: player_ptr + PLAYER_IS_FOCUSED, len: 4 },
            RemoteIoVec { base: player_ptr + PLAYER_HITBOX_RADIUS, len: 4 },
        ];

        read(self.pid, self.data.as_io_slices_mut(), &locations)?;

        let pos: &[f32; 2] = self.data.get(0);
        let is_focused = self.data.get::<u32>(1) == &1;
        let hitbox_radius = *self.data.get(2);

        Ok(Some(GameState {
            player: PlayerState {
                pos_x: pos[0],
                pos_y: pos[1],
                hitbox_radius,
                is_focused,
            },
        }))
    }

    /// Gets the current game state.
    /// This function is currently unimplemented on non-Linux platforms and will panic when invoked.
    #[cfg(not(target_os = "linux"))]
    #[allow(clippy::missing_errors_doc)]
    pub fn get_state(&mut self) -> Result<Option<GameState>> {
        unimplemented!("game process memory reading is only supported on Linux");
    }
}

#[derive(Debug)]
struct GameData<const N: usize> {
    io_slices: [IoSliceMut<'static>; N],
    buffers: [Vec<u8>; N],
}

impl<const N: usize> GameData<N> {
    /// Instantiates a new group of buffers for storing data from game process memory.
    /// The group will contain `N` buffers, where `N` is the length of `sizes`.
    /// The size of each buffer depends on the contents of `sizes`.
    /// For example, `GameData::new([4, 2, 4])` creates three byte buffers of lengths 4, 2, and 4, respectively.
    fn new(sizes: [usize; N]) -> Self {
        let mut buffers = sizes.map(|size| vec![0u8; size]);

        let io_slices = buffers
            .iter_mut()
            .map(|buf| unsafe {
                // SAFETY: We're creating IoSliceMut with 'static lifetime, but we ensure
                // `buffers` lives at least as long as `io_slices` by storing them together in the GameData struct.
                // Also, the struct fields are ordered so that `io_slices` is dropped before `buffers`.
                IoSliceMut::new(std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self { io_slices, buffers }
    }

    /// Returns the list of buffers as `&mut [IoSliceMut<'_>]` for use with `process_vm_readv()`.
    fn as_io_slices_mut(&mut self) -> &mut [IoSliceMut<'static>] {
        self.io_slices.as_mut()
    }

    /// Interprets the byte buffer at the provided index as a value of type `T`,
    /// then returns a reference to the value.
    fn get<T: FromBytes + KnownLayout + Immutable>(&self, index: usize) -> &T {
        FromBytes::ref_from_bytes(&self.buffers[index]).unwrap()
    }
}

/// Utility method wrapping `process_vm_readv()` with more helpful error messages.
#[cfg(target_os = "linux")]
fn read(pid: Pid, buffers: &mut [IoSliceMut<'_>], locations: &[RemoteIoVec]) -> Result<()> {
    if let Err(errno) = process_vm_readv(pid, buffers, locations) {
        let err = match errno {
            Errno::EFAULT => Some(anyhow!(
                "a memory location is out of bounds for the game process"
            )),
            Errno::EPERM => Some(anyhow!(
                "this program does not have permission to access the address space of the game process"
            )),
            Errno::ESRCH => Some(anyhow!("no process with PID {pid} exists")),
            _ => None,
        };

        return Err(match err {
            Some(e) => e.context("failed to read game process memory"),
            None => anyhow!("failed to read game process memory"),
        });
    }

    Ok(())
}
