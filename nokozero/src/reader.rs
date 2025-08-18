use anyhow::{Context, Result, anyhow, bail};
use nix::errno::Errno;
use nix::unistd::Pid;
use std::array;
use std::fs::read_to_string;
use std::io::IoSliceMut;
use std::slice;
use tap::Pipe;
use zerocopy::{FromBytes, Immutable, KnownLayout};

#[cfg(target_os = "linux")]
use nix::sys::uio::{RemoteIoVec, process_vm_readv};

const WORLD_WIDTH: f32 = 384.;
const WORLD_HEIGHT: f32 = 448.;

// Locations and offsets for reading data in process memory
const BULLETS_PTR: usize = 0xe9a6c;
const BULLETS_LIST: usize = 0x68;
const BULLET_NEXT_PTR: usize = 0x4;
const BULLET_POS: usize = 0xc38;
const BULLET_VEL: usize = 0xc44;
const BULLET_HITBOX_RADIUS: usize = 0xc58;
const BULLET_STATE: usize = 0xc8a;

const ENEMIES_PTR: usize = 0xe9a80;
const ENEMIES_LIST: usize = 0x180;
const ENEMY_NEXT_PTR: usize = 0x4;
const ENEMY_POS: usize = 0x120c + 0x44;
const ENEMY_VEL: usize = 0x120c + 0x78;
const ENEMY_HITBOX_RADIUS: usize = 0x120c + 0x118;
const ENEMY_ANM_VM_ID: usize = 0x120c + 0x124;
const ENEMY_HP: usize = 0x120c + 0x3f74;
const ENEMY_MAX_HP: usize = 0x120c + 0x3f78;
const ENEMY_FLAGS: usize = 0x120c + 0x4060;
const ENEMY_BOSS_FLAG: u32 = 1 << 23;

const ITEMS_PTR: usize = 0xe9a9c;
const ITEMS_ARRAY: usize = 0x0;
const ITEM_POS: usize = 0xc30;
const ITEM_VEL: usize = 0xc3c;
const ITEM_STATE: usize = 0xc74;
const ITEM_TYPE: usize = 0xc78;
const ITEM_BYTE_LEN: usize = 0xc88;
// Reference: https://github.com/exphp-share/th-re-data/blob/41cd633354f3bbc4ff11b3d315ef7243c990f227/data/th15.v1.00b/type-structs-own.json#L1025
const ITEMS_CAP: usize = 600;

const LASERS_PTR: usize = 0xe9ba0;
const LASERS_LIST: usize = 0x5e0;
const LASER_NEXT_PTR: usize = 0x4;
const LASER_TYPE: usize = 0x14;
const LASER_POS: usize = 0x54;
const LASER_ANGLE: usize = 0x6c;
const LASER_LENGTH: usize = 0x70;
const LASER_WIDTH: usize = 0x74;
const LASER_SPEED: usize = 0x78;
const RAY_LASER_ORIGIN_VEL: usize = 0x5d4 + 0xc;
const RAY_LASER_ANGULAR_VEL: usize = 0x5d4 + 0x1c;
const CURVE_LASER_NUM_NODES: usize = 0x5d4 + 0x20;
const CURVE_LASER_NODES_ARRAY: usize = 0x5d4 + 0xf68;
const CURVE_LASER_NODE_POS: usize = 0x0;
const CURVE_LASER_NODE_ANGLE: usize = 0x18;
const CURVE_LASER_NODE_SPEED: usize = 0x1c;
const CURVE_LASER_NODE_BYTE_LEN: usize = 0x20;

const PLAYER_PTR: usize = 0xe9bb8;
const PLAYER_POS: usize = 0x618;
const PLAYER_IS_FOCUSED: usize = 0x16240;
const PLAYER_HITBOX_RADIUS: usize = 0x2bfc8;

#[derive(Debug)]
pub struct StateReader {
    pid: Pid,
    base_addr: usize,
    ptrs: Data<5>,                  // managers: bullet, enemy, item, laser, player
    single_ptr: Data<1>,            // single list pointer
    list_ptrs: Data<2>,             // list pointers: current, next
    bullet_data: Data<4>,           // pos, vel, radius, state
    enemy_data: Data<7>,            // pos, vel, radius, ANM VM ID, HP, max HP, flags
    item_data: Data<4>,             // pos, vel, state, type
    base_laser_data: Data<2>,       // type, width
    segment_laser_data: Data<4>,    // pos, angle, length, speed
    ray_laser_data: Data<4>,        // pos, angle, origin vel, angular vel
    curve_laser_data: Data<2>,      // node count, pointer to list head
    curve_laser_node_data: Data<3>, // pos, angle, speed
    player_data: Data<3>,           // pos, focus state, radius
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub bullets: Vec<Bullet>,
    pub enemies: Vec<Enemy>,
    pub power_items: Vec<PowerItem>,
    pub lasers: Lasers,
    pub player: Player,
}

#[derive(Debug, Clone, Copy)]
pub struct Bullet {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub hitbox_radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Enemy {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub hitbox_radius: f32,
    pub hp_ratio: f32,
    pub is_boss: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PowerItem {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
}

#[derive(Debug, Clone)]
pub struct Lasers {
    pub segments: Vec<SegmentLaser>,
    pub rays: Vec<RayLaser>,
    pub curves: Vec<CurveLaser>,
}

#[derive(Debug, Clone, Copy)]
pub struct SegmentLaser {
    pub head_pos_x: f32,
    pub head_pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub length: f32,
    pub width: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RayLaser {
    pub origin_pos_x: f32,
    pub origin_pos_y: f32,
    pub origin_vel_x: f32,
    pub origin_vel_y: f32,
    pub cos_angle: f32,
    pub sin_angle: f32,
    pub angular_vel: f32,
    pub width: f32,
}

#[derive(Debug, Clone)]
pub struct CurveLaser {
    pub nodes: Vec<CurveLaserNode>,
    pub width: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct CurveLaserNode {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Player {
    pub pos_x: f32,
    pub pos_y: f32,
    pub is_focused: bool,
    pub hitbox_radius: f32,
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
            ptrs: Data::new([4, 4, 4, 4, 4]),
            single_ptr: Data::new([4]),
            list_ptrs: Data::new([4, 4]),
            bullet_data: Data::new([8, 8, 4, 2]),
            enemy_data: Data::new([8, 8, 4, 4, 4, 4, 4]),
            item_data: Data::new([8, 8, 4, 4]),
            base_laser_data: Data::new([4, 4]),
            segment_laser_data: Data::new([8, 4, 4, 4]),
            ray_laser_data: Data::new([8, 4, 8, 4]),
            curve_laser_data: Data::new([4, 4]),
            curve_laser_node_data: Data::new([8, 4, 4]),
            player_data: Data::new([8, 4, 4]),
        })
    }

    /// Gets the current state of the game, including
    /// the player, bullets, enemies, lasers, and items.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    pub fn get_state(&mut self) -> Result<Option<GameState>> {
        #[rustfmt::skip]
        self.ptrs.read(self.pid, &[
            self.base_addr + BULLETS_PTR,
            self.base_addr + ENEMIES_PTR,
            self.base_addr + ITEMS_PTR,
            self.base_addr + LASERS_PTR,
            self.base_addr + PLAYER_PTR,
        ])?;

        let bullets_ptr = *self.ptrs.get::<u32>(0) as usize;
        let enemies_ptr = *self.ptrs.get::<u32>(1) as usize;
        let items_ptr = *self.ptrs.get::<u32>(2) as usize;
        let lasers_ptr = *self.ptrs.get::<u32>(3) as usize;
        let player_ptr = *self.ptrs.get::<u32>(4) as usize;

        if bullets_ptr == 0
            || enemies_ptr == 0
            || items_ptr == 0
            || lasers_ptr == 0
            || player_ptr == 0
        {
            return Ok(None);
        }

        Ok(Some(GameState {
            bullets: self.get_bullets(bullets_ptr)?,
            enemies: self.get_enemies(enemies_ptr)?,
            power_items: self.get_power_items(items_ptr)?,
            lasers: self.get_lasers(lasers_ptr)?,
            player: self.get_player(player_ptr)?,
        }))
    }

    /// Gets the current state of all bullets on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_bullets(&mut self, bullets_ptr: usize) -> Result<Vec<Bullet>> {
        let mut bullets = Vec::new();

        let mut bullet_next_ptr = bullets_ptr + BULLETS_LIST;

        while bullet_next_ptr != 0 {
            #[rustfmt::skip]
            self.list_ptrs.read(self.pid, &[
                bullet_next_ptr,
                bullet_next_ptr + BULLET_NEXT_PTR
            ])?;

            let bullet_data_ptr = *self.list_ptrs.get::<u32>(0) as usize;
            bullet_next_ptr = *self.list_ptrs.get::<u32>(1) as usize;

            if bullet_data_ptr == 0 {
                continue;
            }

            #[rustfmt::skip]
            self.bullet_data.read(self.pid, &[
                bullet_data_ptr + BULLET_POS,
                bullet_data_ptr + BULLET_VEL,
                bullet_data_ptr + BULLET_HITBOX_RADIUS,
                bullet_data_ptr + BULLET_STATE,
            ])?;

            let is_active = self.bullet_data.get::<u16>(3) == &1;

            if is_active {
                let &[pos_x, pos_y] = self.bullet_data.get(0);
                let hitbox_radius: f32 = *self.bullet_data.get(2);

                // Check if bullet is in bounds
                if pos_y >= -hitbox_radius
                    && pos_y <= WORLD_HEIGHT + hitbox_radius
                    && pos_x >= const { -WORLD_WIDTH / 2. } - hitbox_radius
                    && pos_x <= const { WORLD_WIDTH / 2. } + hitbox_radius
                {
                    let &[vel_x, vel_y] = self.bullet_data.get(1);

                    bullets.push(Bullet {
                        pos_x,
                        pos_y,
                        vel_x,
                        vel_y,
                        hitbox_radius,
                    });
                }
            }
        }

        Ok(bullets)
    }

    /// Gets the current state of all enemies on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_enemies(&mut self, enemies_ptr: usize) -> Result<Vec<Enemy>> {
        let mut enemies = Vec::new();

        // Get the pointer to the head of the linked list of enemies
        let mut enemy_next_ptr = self.read_single_ptr(enemies_ptr + ENEMIES_LIST)?;

        while enemy_next_ptr != 0 {
            #[rustfmt::skip]
            self.list_ptrs.read(self.pid, &[
                enemy_next_ptr,
                enemy_next_ptr + ENEMY_NEXT_PTR
            ])?;

            let enemy_data_ptr = *self.list_ptrs.get::<u32>(0) as usize;
            enemy_next_ptr = *self.list_ptrs.get::<u32>(1) as usize;

            if enemy_data_ptr == 0 {
                continue;
            }

            #[rustfmt::skip]
            self.enemy_data.read(self.pid, &[
                enemy_data_ptr + ENEMY_POS,
                enemy_data_ptr + ENEMY_VEL,
                enemy_data_ptr + ENEMY_HITBOX_RADIUS,
                enemy_data_ptr + ENEMY_ANM_VM_ID,
                enemy_data_ptr + ENEMY_HP,
                enemy_data_ptr + ENEMY_MAX_HP,
                enemy_data_ptr + ENEMY_FLAGS,
            ])?;

            let is_boss = self.enemy_data.get::<u32>(6) & ENEMY_BOSS_FLAG != 0;
            let has_anm_vm_id = self.enemy_data.get::<u32>(3) != &0;

            // Check if the enemy is "real"; i.e. is a boss or has an ANM VM ID set.
            // The game uses "fake" enemies to make certain bullet patterns easier to implement.
            // However, the player cannot interact with these enemies,
            // so they should not be counted as distinct entities.
            if is_boss || has_anm_vm_id {
                let &[pos_x, pos_y] = self.enemy_data.get(0);
                let &[vel_x, vel_y] = self.enemy_data.get(1);
                let hitbox_radius = *self.enemy_data.get(2);

                #[allow(clippy::cast_precision_loss)]
                let hp_ratio = {
                    let hp = *self.enemy_data.get::<i32>(4) as f32;
                    let max_hp = *self.enemy_data.get::<i32>(5) as f32;
                    (hp / max_hp).clamp(0., 1.)
                };

                enemies.push(Enemy {
                    pos_x,
                    pos_y,
                    vel_x,
                    vel_y,
                    hitbox_radius,
                    hp_ratio,
                    is_boss,
                });
            }
        }

        Ok(enemies)
    }

    /// Gets the current state of all power items on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_power_items(&mut self, items_ptr: usize) -> Result<Vec<PowerItem>> {
        let mut items = Vec::new();

        for i in 0..ITEMS_CAP {
            let item_base = items_ptr + ITEMS_ARRAY + i * ITEM_BYTE_LEN;

            #[rustfmt::skip]
            self.item_data.read(self.pid, &[
                item_base + ITEM_POS,
                item_base + ITEM_VEL,
                item_base + ITEM_STATE,
                item_base + ITEM_TYPE,
            ])?;

            let is_active = self.item_data.get::<u32>(2) != &0;
            let is_power_item = matches!(self.item_data.get::<u32>(3), 1 | 3 | 8);

            if is_active && is_power_item {
                let &[pos_x, pos_y] = self.item_data.get(0);
                let &[vel_x, vel_y] = self.item_data.get(1);

                items.push(PowerItem {
                    pos_x,
                    pos_y,
                    vel_x,
                    vel_y,
                });
            }
        }

        Ok(items)
    }

    /// Gets the current state of all lasers on the playing field.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_lasers(&mut self, lasers_ptr: usize) -> Result<Lasers> {
        let mut segment_lasers = Vec::new();
        let mut ray_lasers = Vec::new();
        let mut curve_lasers = Vec::new();

        // Get the pointer to the head of the linked list of lasers
        let mut laser_data_ptr = self.read_single_ptr(lasers_ptr + LASERS_LIST)?;

        loop {
            let laser_next_ptr = self.read_single_ptr(laser_data_ptr + LASER_NEXT_PTR)?;

            if laser_next_ptr == 0 {
                break;
            }

            #[rustfmt::skip]
            self.base_laser_data.read(self.pid, &[
                laser_data_ptr + LASER_TYPE,
                laser_data_ptr + LASER_WIDTH
            ])?;

            let laser_type: u32 = *self.base_laser_data.get(0);
            let width = *self.base_laser_data.get(1);

            match laser_type {
                0 => {
                    #[rustfmt::skip]
                    self.segment_laser_data.read(self.pid, &[
                        laser_data_ptr + LASER_POS,
                        laser_data_ptr + LASER_ANGLE,
                        laser_data_ptr + LASER_LENGTH,
                        laser_data_ptr + LASER_SPEED,
                    ])?;

                    let &[head_pos_x, head_pos_y] = self.segment_laser_data.get(0);
                    let (sin_angle, cos_angle) = self.segment_laser_data.get::<f32>(1).sin_cos();
                    let length = *self.segment_laser_data.get(2);
                    let speed = self.segment_laser_data.get(3);

                    segment_lasers.push(SegmentLaser {
                        head_pos_x,
                        head_pos_y,
                        vel_x: speed * cos_angle,
                        vel_y: speed * sin_angle,
                        length,
                        width,
                    });
                }
                1 => {
                    #[rustfmt::skip]
                    self.ray_laser_data.read(self.pid, &[
                        laser_data_ptr + LASER_POS,
                        laser_data_ptr + LASER_ANGLE,
                        laser_data_ptr + RAY_LASER_ORIGIN_VEL,
                        laser_data_ptr + RAY_LASER_ANGULAR_VEL,
                    ])?;

                    let &[origin_pos_x, origin_pos_y] = self.ray_laser_data.get(0);
                    let (sin_angle, cos_angle) = self.ray_laser_data.get::<f32>(1).sin_cos();
                    let &[origin_vel_x, origin_vel_y] = self.ray_laser_data.get(2);
                    let angular_vel = *self.ray_laser_data.get(3);

                    ray_lasers.push(RayLaser {
                        origin_pos_x,
                        origin_pos_y,
                        origin_vel_x,
                        origin_vel_y,
                        cos_angle,
                        sin_angle,
                        angular_vel,
                        width,
                    });
                }
                2 => {
                    let mut nodes = Vec::new();

                    #[rustfmt::skip]
                    self.curve_laser_data.read(self.pid, &[
                        laser_data_ptr + CURVE_LASER_NUM_NODES,
                        laser_data_ptr + CURVE_LASER_NODES_ARRAY,
                    ])?;

                    let num_nodes = *self.curve_laser_data.get::<u32>(0) as usize;
                    let node_data_ptr = *self.curve_laser_data.get::<u32>(1) as usize;

                    for i in 0..num_nodes {
                        let node_base = node_data_ptr + i * CURVE_LASER_NODE_BYTE_LEN;

                        #[rustfmt::skip]
                        self.curve_laser_node_data.read(self.pid, &[
                            node_base + CURVE_LASER_NODE_POS,
                            node_base + CURVE_LASER_NODE_ANGLE,
                            node_base + CURVE_LASER_NODE_SPEED,
                        ])?;

                        let &[pos_x, pos_y] = self.curve_laser_node_data.get(0);
                        let (sin_angle, cos_angle) =
                            self.curve_laser_node_data.get::<f32>(1).sin_cos();
                        let speed = self.curve_laser_node_data.get(2);

                        nodes.push(CurveLaserNode {
                            pos_x,
                            pos_y,
                            vel_x: speed * cos_angle,
                            vel_y: speed * sin_angle,
                        });
                    }

                    curve_lasers.push(CurveLaser { nodes, width });
                }
                id => bail!("unknown laser type (type {id})"),
            }

            laser_data_ptr = laser_next_ptr;
        }

        Ok(Lasers {
            segments: segment_lasers,
            rays: ray_lasers,
            curves: curve_lasers,
        })
    }

    /// Gets the current state of the player.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn get_player(&mut self, player_ptr: usize) -> Result<Player> {
        #[rustfmt::skip]
        self.player_data.read(self.pid, &[
            player_ptr + PLAYER_POS,
            player_ptr + PLAYER_IS_FOCUSED,
            player_ptr + PLAYER_HITBOX_RADIUS,
        ])?;

        let &[pos_x, pos_y] = self.player_data.get(0);
        let is_focused = self.player_data.get::<u32>(1) == &1;
        let hitbox_radius = *self.player_data.get(2);

        Ok(Player {
            pos_x,
            pos_y,
            is_focused,
            hitbox_radius,
        })
    }

    /// Utility method that reads a single `u32` at the specified location
    /// and interprets it as the address of a pointer.
    ///
    /// # Errors
    /// This function returns an error if the game process memory could not be read.
    #[cfg(target_os = "linux")]
    fn read_single_ptr(&mut self, location: usize) -> Result<usize> {
        self.single_ptr.read(self.pid, &[location])?;
        Ok(*self.single_ptr.get::<u32>(0) as usize)
    }

    /// Gets the current state of the game, including
    /// the player, bullets, enemies, lasers, and items.
    ///
    /// This function is currently unimplemented on non-Linux platforms
    /// and will panic when invoked.
    #[cfg(not(target_os = "linux"))]
    #[allow(clippy::missing_errors_doc)]
    pub fn get_state(&mut self) -> Result<Option<GameState>> {
        unimplemented!("game process memory reading is only supported on Linux");
    }
}

#[derive(Debug)]
struct Data<const N: usize> {
    sizes: [usize; N],
    io_slices: [IoSliceMut<'static>; N],
    buffers: [Vec<u8>; N],
}

impl<const N: usize> Data<N> {
    /// Instantiates a new group of buffers for storing data from process memory.
    /// The group will contain `N` buffers, where `N` is the length of `sizes`.
    /// The size of each buffer depends on the contents of `sizes`.
    /// For example, `Data::new([4, 2, 4])`
    /// creates three byte buffers of lengths 4, 2, and 4, respectively.
    fn new(sizes: [usize; N]) -> Self {
        let mut buffers = sizes.map(|size| vec![0u8; size]);

        let io_slices = buffers.each_mut().map(|buf| unsafe {
            // SAFETY: We're creating `IoSliceMut` with `'static` lifetime,
            // but we ensure `buffers` lives at least as long as `io_slices`
            // by storing them together in the `Data` struct.
            // Also, the struct fields are ordered so that `io_slices` is dropped before `buffers`.
            IoSliceMut::new(slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()))
        });

        Self {
            sizes,
            io_slices,
            buffers,
        }
    }

    /// Reads the memory of the process with the given PID at the given base addresses
    /// and copies the data into this instance of `Data`.
    ///
    /// # Errors
    /// This function returns an error if the process memory could not be read.
    #[cfg(target_os = "linux")]
    fn read(&mut self, pid: Pid, locations: &[usize; N]) -> Result<()> {
        // Construct `RemoteIoVec` from base address and stored size
        let locations: [_; N] = array::from_fn(|i| RemoteIoVec {
            base: locations[i],
            len: self.sizes[i],
        });

        if let Err(errno) = process_vm_readv(pid, self.io_slices.as_mut(), &locations) {
            return Err(match errno {
                Errno::EFAULT => anyhow!(
                    "a memory location is out of bounds for the process"
                ),
                Errno::EINVAL => anyhow!("length of data to be read is too large"),
                Errno::ENOMEM => anyhow!("fatal error (out of memory)"),
                Errno::EPERM => anyhow!(
                    "this program does not have permission to access the address space of the process"
                ),
                Errno::ESRCH => anyhow!("no process with PID {pid} exists"),
                _ => anyhow!("unknown error (errno {errno}"),
            }.context("failed to read process memory"));
        }

        Ok(())
    }

    /// Interprets the byte buffer at the given index as a value of type `T`,
    /// then returns a reference to the value.
    fn get<T: FromBytes + KnownLayout + Immutable>(&self, index: usize) -> &T {
        FromBytes::ref_from_bytes(&self.buffers[index]).unwrap()
    }
}
