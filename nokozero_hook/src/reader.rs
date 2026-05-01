use std::ptr::{NonNull, with_exposed_provenance_mut};

const WORLD_WIDTH: f32 = 384.;
const WORLD_HEIGHT: f32 = 448.;
const HALF_WORLD_WIDTH: f32 = WORLD_WIDTH / 2.;

// Locations and offsets for reading data in process memory
const BULLETS_PTR: usize = 0xe9a6c;
const BULLETS_LIST: usize = 0x68;
const BULLET_POS: usize = 0xc38;
const BULLET_VEL: usize = 0xc44;
const BULLET_HITBOX_RADIUS: usize = 0xc58;
const BULLET_STATE: usize = 0xc8a;

const ENEMIES_PTR: usize = 0xe9a80;
const ENEMIES_LIST: usize = 0x180;
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

const GAME_THREAD_PTR: usize = 0xe9a94;
const PLAYER_PTR: usize = 0xe9bb8;
const PLAYER_POS: usize = 0x618;
const PLAYER_IS_FOCUSED: usize = 0x16240;
const PLAYER_HITBOX_RADIUS: usize = 0x2bfc8;

#[derive(Debug, Clone, Copy)]
pub struct StateReader {
    base: GamePtr,
}

// SAFETY: The base pointer refers to the game executable's PE image,
// which is loaded for the lifetime of the game process.
// `StateReader` only reads memory and does not mutate it.
unsafe impl Send for StateReader {}
unsafe impl Sync for StateReader {}

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
    /// Creates a new reader with the given module base address.
    ///
    /// # Safety
    /// `base` must point to the start of the game's loaded PE image
    /// and must remain valid for the lifetime of the `StateReader`.
    #[must_use]
    pub unsafe fn new(base: NonNull<u8>) -> Self {
        Self {
            base: GamePtr(base),
        }
    }

    /// Returns `true` if the game's main loop is active, and `false` otherwise.
    #[must_use]
    pub fn is_game_active(&self) -> bool {
        (unsafe { self.base.read::<usize>(GAME_THREAD_PTR) }) != 0
    }

    /// Gets the current state of the game, including the player, bullets, enemies, lasers, and items.
    #[must_use]
    pub fn get_state(&self) -> Option<GameState> {
        let bullets_ptr = unsafe { self.base.read_ptr(BULLETS_PTR) }?;
        let enemies_ptr = unsafe { self.base.read_ptr(ENEMIES_PTR) }?;
        let items_ptr = unsafe { self.base.read_ptr(ITEMS_PTR) }?;
        let lasers_ptr = unsafe { self.base.read_ptr(LASERS_PTR) }?;
        let player_ptr = unsafe { self.base.read_ptr(PLAYER_PTR) }?;

        Some(GameState {
            bullets: get_bullets(bullets_ptr),
            enemies: get_enemies(enemies_ptr),
            power_items: get_power_items(items_ptr),
            lasers: get_lasers(lasers_ptr),
            player: get_player(player_ptr),
        })
    }
}

/// A non-null read-only pointer into process memory.
#[derive(Debug, Clone, Copy)]
struct GamePtr(NonNull<u8>);

impl GamePtr {
    /// Reconstitutes a `GamePtr` from a raw exposed-provenance address.
    /// Returns `None` if the address is null.
    fn from_addr(addr: usize) -> Option<Self> {
        NonNull::new(with_exposed_provenance_mut::<u8>(addr)).map(Self)
    }

    /// Calculates the offset from `self` in bytes.
    ///
    /// # Safety
    /// `self + offset` must remain within a valid allocation.
    unsafe fn byte_add(self, offset: usize) -> Self {
        Self(unsafe { self.0.byte_add(offset) })
    }

    /// Reads a value of type `T` at the given byte offset from `self`.
    ///
    /// # Safety
    /// `self + offset` must point to a valid, initialized `T`.
    unsafe fn read<T>(self, offset: usize) -> T {
        unsafe { self.0.byte_add(offset).cast::<T>().read_unaligned() }
    }

    /// Reads a 32-bit game pointer at the given byte offset from `self` and reconstitutes it.
    /// Returns `None` if the read pointer is null.
    ///
    /// # Safety
    /// `self + offset` must point to a valid `u32`.
    unsafe fn read_ptr(self, offset: usize) -> Option<Self> {
        let raw: u32 = unsafe { self.read(offset) };
        Self::from_addr(raw as usize)
    }
}

/// Traverses an indirect linked list, yielding data pointers.
/// The data pointer is at offset 0 and the next-entry pointer is at offset 4.
///
/// The iterator must be constructed with a pointer that is either `None`
/// or points at the first entry of a well-formed list.
struct List(Option<GamePtr>);

impl Iterator for List {
    type Item = GamePtr;

    fn next(&mut self) -> Option<GamePtr> {
        const NEXT: usize = 4;
        while let Some(node) = self.0 {
            let data = unsafe { node.read_ptr(0) };
            self.0 = unsafe { node.read_ptr(NEXT) };
            if let Some(data) = data {
                return Some(data);
            }
        }
        None
    }
}

/// Traverses a list of laser entities.
/// The list is terminated by a tail sentinel whose `next` field is null; the sentinel itself is not yielded.
///
/// The iterator must be constructed with a pointer to the head of such a list,
/// which may itself be the tail sentinel if the list is empty.
struct LaserList(GamePtr);

impl Iterator for LaserList {
    type Item = GamePtr;

    fn next(&mut self) -> Option<GamePtr> {
        let next = unsafe { self.0.read_ptr(LASER_NEXT_PTR) }?;
        let current = self.0;
        self.0 = next;
        Some(current)
    }
}

/// Gets the current state of all bullets on the playing field.
fn get_bullets(bullets_ptr: GamePtr) -> Vec<Bullet> {
    let mut bullets = Vec::new();

    let head = unsafe { bullets_ptr.byte_add(BULLETS_LIST) };
    for data in List(Some(head)) {
        let is_active = unsafe { data.read::<u16>(BULLET_STATE) } == 1;
        if !is_active {
            continue;
        }

        let [pos_x, pos_y] = unsafe { data.read::<[f32; 2]>(BULLET_POS) };
        let hitbox_radius = unsafe { data.read::<f32>(BULLET_HITBOX_RADIUS) };

        // Check if bullet is in bounds
        if pos_y >= -hitbox_radius
            && pos_y <= WORLD_HEIGHT + hitbox_radius
            && pos_x >= -HALF_WORLD_WIDTH - hitbox_radius
            && pos_x <= HALF_WORLD_WIDTH + hitbox_radius
        {
            let [vel_x, vel_y] = unsafe { data.read::<[f32; 2]>(BULLET_VEL) };

            bullets.push(Bullet {
                pos_x,
                pos_y,
                vel_x,
                vel_y,
                hitbox_radius,
            });
        }
    }

    bullets
}

/// Gets the current state of all enemies on the playing field.
fn get_enemies(enemies_ptr: GamePtr) -> Vec<Enemy> {
    let mut enemies = Vec::new();

    let head = unsafe { enemies_ptr.read_ptr(ENEMIES_LIST) };
    for data in List(head) {
        let flags = unsafe { data.read::<u32>(ENEMY_FLAGS) };
        let is_boss = flags & ENEMY_BOSS_FLAG != 0;
        let has_anm_vm_id = unsafe { data.read::<u32>(ENEMY_ANM_VM_ID) } != 0;

        // Check if the enemy is "real"; i.e. is a boss or has an ANM VM ID set.
        // The game uses "fake" enemies to make certain bullet patterns easier to implement.
        // However, the player cannot interact with these enemies,
        // so they should not be counted as distinct entities.
        if !(is_boss || has_anm_vm_id) {
            continue;
        }

        let [pos_x, pos_y] = unsafe { data.read::<[f32; 2]>(ENEMY_POS) };
        let [vel_x, vel_y] = unsafe { data.read::<[f32; 2]>(ENEMY_VEL) };
        let hitbox_radius = unsafe { data.read::<f32>(ENEMY_HITBOX_RADIUS) };

        #[allow(clippy::cast_precision_loss)]
        let hp_ratio = {
            let hp = unsafe { data.read::<i32>(ENEMY_HP) } as f32;
            let max_hp = unsafe { data.read::<i32>(ENEMY_MAX_HP) } as f32;
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

    enemies
}

/// Gets the current state of all power items on the playing field.
fn get_power_items(items_ptr: GamePtr) -> Vec<PowerItem> {
    let mut items = Vec::new();

    for i in 0..ITEMS_CAP {
        let item = unsafe { items_ptr.byte_add(ITEMS_ARRAY + i * ITEM_BYTE_LEN) };

        let state = unsafe { item.read::<u32>(ITEM_STATE) };
        let item_type = unsafe { item.read::<u32>(ITEM_TYPE) };

        let is_active = state != 0;
        let is_power_item = matches!(item_type, 1 | 3 | 8);

        if is_active && is_power_item {
            let [pos_x, pos_y] = unsafe { item.read::<[f32; 2]>(ITEM_POS) };
            let [vel_x, vel_y] = unsafe { item.read::<[f32; 2]>(ITEM_VEL) };

            items.push(PowerItem {
                pos_x,
                pos_y,
                vel_x,
                vel_y,
            });
        }
    }

    items
}

/// Gets the current state of all lasers on the playing field.
fn get_lasers(lasers_ptr: GamePtr) -> Lasers {
    let mut segment_lasers = Vec::new();
    let mut ray_lasers = Vec::new();
    let mut curve_lasers = Vec::new();

    if let Some(head) = unsafe { lasers_ptr.read_ptr(LASERS_LIST) } {
        for laser in LaserList(head) {
            let laser_type = unsafe { laser.read::<u32>(LASER_TYPE) };
            let width = unsafe { laser.read::<f32>(LASER_WIDTH) };

            match laser_type {
                0 => {
                    let [head_pos_x, head_pos_y] = unsafe { laser.read::<[f32; 2]>(LASER_POS) };
                    let (sin_angle, cos_angle) =
                        unsafe { laser.read::<f32>(LASER_ANGLE) }.sin_cos();
                    let length = unsafe { laser.read::<f32>(LASER_LENGTH) };
                    let speed = unsafe { laser.read::<f32>(LASER_SPEED) };

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
                    let [origin_pos_x, origin_pos_y] = unsafe { laser.read::<[f32; 2]>(LASER_POS) };
                    let (sin_angle, cos_angle) =
                        unsafe { laser.read::<f32>(LASER_ANGLE) }.sin_cos();
                    let [origin_vel_x, origin_vel_y] =
                        unsafe { laser.read::<[f32; 2]>(RAY_LASER_ORIGIN_VEL) };
                    let angular_vel = unsafe { laser.read::<f32>(RAY_LASER_ANGULAR_VEL) };

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
                    let num_nodes = unsafe { laser.read::<u32>(CURVE_LASER_NUM_NODES) } as usize;
                    let mut nodes = Vec::with_capacity(num_nodes);

                    if let Some(base) = unsafe { laser.read_ptr(CURVE_LASER_NODES_ARRAY) } {
                        for i in 0..num_nodes {
                            let node = unsafe { base.byte_add(i * CURVE_LASER_NODE_BYTE_LEN) };

                            let [pos_x, pos_y] =
                                unsafe { node.read::<[f32; 2]>(CURVE_LASER_NODE_POS) };
                            let (sin_angle, cos_angle) =
                                unsafe { node.read::<f32>(CURVE_LASER_NODE_ANGLE) }.sin_cos();
                            let speed = unsafe { node.read::<f32>(CURVE_LASER_NODE_SPEED) };

                            nodes.push(CurveLaserNode {
                                pos_x,
                                pos_y,
                                vel_x: speed * cos_angle,
                                vel_y: speed * sin_angle,
                            });
                        }
                    }

                    curve_lasers.push(CurveLaser { nodes, width });
                }
                id => panic!("unknown laser type (type {id})"),
            }
        }
    }

    Lasers {
        segments: segment_lasers,
        rays: ray_lasers,
        curves: curve_lasers,
    }
}

/// Gets the current state of the player.
fn get_player(player_ptr: GamePtr) -> Player {
    let [pos_x, pos_y] = unsafe { player_ptr.read::<[f32; 2]>(PLAYER_POS) };
    let is_focused = unsafe { player_ptr.read::<u32>(PLAYER_IS_FOCUSED) } == 1;
    let hitbox_radius = unsafe { player_ptr.read::<f32>(PLAYER_HITBOX_RADIUS) };

    Player {
        pos_x,
        pos_y,
        is_focused,
        hitbox_radius,
    }
}
