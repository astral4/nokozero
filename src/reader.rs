use anyhow::{Context, Result, anyhow};
use nix::errno::Errno;
use nix::unistd::Pid;
use std::fs::read_to_string;
use std::io::IoSliceMut;
use tap::Pipe;
use zerocopy::{FromBytes, Immutable, KnownLayout};

#[cfg(target_os = "linux")]
use nix::sys::uio::{RemoteIoVec, process_vm_readv};

const WORLD_WIDTH: f32 = 384.;
const WORLD_HEIGHT: f32 = 448.;

// Offsets/locations for certain data in process memory
const PLAYER_PTR: usize = 0xe9bb8;
const PLAYER_POS: usize = 0x618;
const PLAYER_IS_FOCUSED: usize = 0x16240;
const PLAYER_HITBOX_RADIUS: usize = 0x2bfc8;

const BULLETS_PTR: usize = 0xe9a6c;
const BULLETS_LIST: usize = 0x68;
const BULLET_NEXT_PTR: usize = 0x4;
const BULLET_POS: usize = 0xc38;
const BULLET_VEL: usize = 0xc44;
const BULLET_HITBOX_RADIUS: usize = 0xc58;
const BULLET_STATE: usize = 0xc8a;
// Reference: https://exphp.github.io/thpages/#/mods/bullet-cap
const BULLETS_CAP: usize = 2000;

const ITEMS_PTR: usize = 0xe9a9c;
const ITEMS_ARRAY: usize = 0x0;
const ITEM_STATE: usize = 0xc74;
const ITEM_TYPE: usize = 0xc78;
const ITEM_POS: usize = 0xc30;
const ITEM_VEL: usize = 0xc3c;
const ITEM_BYTE_LEN: usize = 0xc88;
// Reference: https://github.com/exphp-share/th-re-data/blob/41cd633354f3bbc4ff11b3d315ef7243c990f227/data/th15.v1.00b/type-structs-own.json#L1025
const ITEMS_CAP: usize = 600;

#[derive(Debug)]
pub struct StateReader {
    pid: Pid,
    base_addr: usize,
    ptrs: GameData<3>,
    player_data: GameData<3>,
    bullet_ptrs: GameData<2>,
    bullet_data: GameData<4>,
    item_data: GameData<4>,
}

#[derive(Debug)]
pub struct GameState {
    pub player: PlayerState,
    pub bullets: Vec<BulletState>,
    pub items: Vec<ItemState>,
}

#[derive(Debug)]
pub struct PlayerState {
    pub pos_x: f32,
    pub pos_y: f32,
    pub hitbox_radius: f32,
    pub is_focused: bool,
}

#[derive(Debug)]
pub struct BulletState {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub hitbox_radius: f32,
}

#[derive(Debug)]
pub struct ItemState {
    pub item_type: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
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

        Ok(Self {
            pid,
            base_addr,
            ptrs: GameData::new([4, 4, 4]),
            player_data: GameData::new([8, 4, 4]),
            bullet_ptrs: GameData::new([4, 4]),
            bullet_data: GameData::new([8, 8, 4, 2]),
            item_data: GameData::new([4, 4, 8, 8]),
        })
    }

    /// Gets the current state of the game, including the player, bullets, and items.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    pub fn get_state(&mut self) -> Result<Option<GameState>> {
        #[rustfmt::skip]
        let ptr_locations = [
            RemoteIoVec { base: self.base_addr + PLAYER_PTR, len: 4 },
            RemoteIoVec { base: self.base_addr + BULLETS_PTR, len: 4 },
            RemoteIoVec { base: self.base_addr + ITEMS_PTR, len: 4 },
        ];

        read(self.pid, self.ptrs.as_io_slices_mut(), &ptr_locations)?;

        let player_ptr = *self.ptrs.get::<u32>(0) as usize;
        let bullets_ptr = *self.ptrs.get::<u32>(1) as usize;
        let items_ptr = *self.ptrs.get::<u32>(2) as usize;

        if player_ptr == 0 || bullets_ptr == 0 || items_ptr == 0 {
            return Ok(None);
        }

        Ok(Some(GameState {
            player: self.get_player_state(player_ptr)?,
            bullets: self.get_bullets_state(bullets_ptr)?,
            items: self.get_items_state(items_ptr)?,
        }))
    }

    /// Gets the current state of the player.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_player_state(&mut self, player_ptr: usize) -> Result<PlayerState> {
        #[rustfmt::skip]
            let locations = [
                RemoteIoVec { base: player_ptr + PLAYER_POS, len: 8 },
                RemoteIoVec { base: player_ptr + PLAYER_IS_FOCUSED, len: 4 },
                RemoteIoVec { base: player_ptr + PLAYER_HITBOX_RADIUS, len: 4 },
            ];

        read(self.pid, self.player_data.as_io_slices_mut(), &locations)?;

        let &[pos_x, pos_y] = self.player_data.get(0);
        let is_focused = self.player_data.get::<u32>(1) == &1;
        let hitbox_radius = *self.player_data.get(2);

        Ok(PlayerState {
            pos_x,
            pos_y,
            hitbox_radius,
            is_focused,
        })
    }

    /// Gets the current state of all bullets on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_bullets_state(&mut self, bullets_ptr: usize) -> Result<Vec<BulletState>> {
        let mut bullets = Vec::new();
        let mut bullet_next_ptr = bullets_ptr + BULLETS_LIST;

        for _ in 0..BULLETS_CAP {
            #[rustfmt::skip]
            let ptr_locations = [
                RemoteIoVec { base: bullet_next_ptr, len: 4 },
                RemoteIoVec { base: bullet_next_ptr + BULLET_NEXT_PTR, len: 4 },
            ];

            read(
                self.pid,
                self.bullet_ptrs.as_io_slices_mut(),
                &ptr_locations,
            )?;

            let bullet_data_ptr = *self.bullet_ptrs.get::<u32>(0) as usize;
            bullet_next_ptr = *self.bullet_ptrs.get::<u32>(1) as usize;

            if bullet_data_ptr == 0 {
                if bullet_next_ptr == 0 {
                    break;
                }
                continue;
            }

            #[rustfmt::skip]
            let locations = [
                RemoteIoVec { base: bullet_data_ptr + BULLET_POS, len: 8 },
                RemoteIoVec { base: bullet_data_ptr + BULLET_VEL, len: 8 },
                RemoteIoVec { base: bullet_data_ptr + BULLET_HITBOX_RADIUS, len: 4 },
                RemoteIoVec { base: bullet_data_ptr + BULLET_STATE, len: 2 }
            ];

            read(self.pid, self.bullet_data.as_io_slices_mut(), &locations)?;

            // Check if bullet state is active
            if self.bullet_data.get::<u16>(3) == &1 {
                let &[pos_x, pos_y] = self.bullet_data.get(0);
                let hitbox_radius: f32 = *self.bullet_data.get(2);

                // Check if bullet is in bounds
                if pos_y >= -hitbox_radius
                    && pos_y <= WORLD_HEIGHT + hitbox_radius
                    && pos_x >= const { -WORLD_WIDTH / 2. } - hitbox_radius
                    && pos_x <= const { WORLD_WIDTH / 2. } + hitbox_radius
                {
                    let &[vel_x, vel_y] = self.bullet_data.get(1);

                    bullets.push(BulletState {
                        pos_x,
                        pos_y,
                        vel_x,
                        vel_y,
                        hitbox_radius,
                    });
                }
            }

            if bullet_next_ptr == 0 {
                break;
            }
        }

        Ok(bullets)
    }

    /// Gets the current state of all items on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_items_state(&mut self, items_ptr: usize) -> Result<Vec<ItemState>> {
        // Read items data
        let mut items = Vec::new();

        for i in 0..ITEMS_CAP {
            let item_base = items_ptr + ITEMS_ARRAY + i * ITEM_BYTE_LEN;

            #[rustfmt::skip]
            let locations = [
                RemoteIoVec { base: item_base + ITEM_STATE, len: 4 },
                RemoteIoVec { base: item_base + ITEM_TYPE, len: 4 },
                RemoteIoVec { base: item_base + ITEM_POS, len: 8 },
                RemoteIoVec { base: item_base + ITEM_VEL, len: 8 },
            ];

            read(self.pid, self.item_data.as_io_slices_mut(), &locations)?;

            // Check if item state is active
            if self.item_data.get::<u32>(0) == &0 {
                continue;
            }

            let item_type = *self.item_data.get(1);
            let &[pos_x, pos_y] = self.item_data.get(2);
            let &[vel_x, vel_y] = self.item_data.get(3);

            items.push(ItemState {
                item_type,
                pos_x,
                pos_y,
                vel_x,
                vel_y,
            });
        }

        Ok(items)
    }

    /// Gets the current state of the game, including the player, bullets, and items.
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

        let io_slices = buffers.each_mut().map(|buf| unsafe {
            // SAFETY: We're creating IoSliceMut with 'static lifetime, but we ensure
            // `buffers` lives at least as long as `io_slices` by storing them together in the GameData struct.
            // Also, the struct fields are ordered so that `io_slices` is dropped before `buffers`.
            IoSliceMut::new(std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()))
        });

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
        return Err(match errno {
            Errno::EFAULT => anyhow!(
                "a memory location is out of bounds for the game process"
            ),
            Errno::EINVAL => anyhow!("length of data to be read is too large"),
            Errno::ENOMEM => anyhow!("fatal error (out of memory)"),
            Errno::EPERM => anyhow!(
                "this program does not have permission to access the address space of the game process"
            ),
            Errno::ESRCH => anyhow!("no process with PID {pid} exists"),
            _ => anyhow!("unknown error (errno {errno}"),
        }.context("failed to read game process memory"));
    }

    Ok(())
}
