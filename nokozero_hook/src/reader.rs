use std::ptr::with_exposed_provenance;
use windows::Win32::System::LibraryLoader::GetModuleHandleA;

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

#[derive(Debug, Clone, Copy)]
pub struct StateReader {
    base: *const u8,
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
    /// Instantiates the game state reader.
    ///
    /// # Panics
    /// This function panics if the game module handle could not be obtained.
    #[must_use]
    pub fn new() -> Self {
        let module = unsafe { GetModuleHandleA(None) }.expect("game module handle should be valid");
        // The `windows` crate represents HMODULE as `isize`,
        // but the underlying value is the PE image base address.
        let base = with_exposed_provenance(module.0 as usize);
        Self { base }
    }

    /// Gets the current state of the game, including
    /// the player, bullets, enemies, lasers, and items.
    #[must_use]
    pub fn get_state(&self) -> Option<GameState> {
        // Note on provenance: reads from the PE image derive provenance
        // from the module handle. When following game pointers into
        // dynamically allocated memory, we reconstitute pointers via
        // `with_exposed_provenance`. This is inherent to DLL injection
        // and cannot be made fully sound under Rust's provenance model.
        // See also the note on provenance in `lib.rs`.
        let bullets_ptr = unsafe { read_ptr(self.base, BULLETS_PTR) };
        let enemies_ptr = unsafe { read_ptr(self.base, ENEMIES_PTR) };
        let items_ptr = unsafe { read_ptr(self.base, ITEMS_PTR) };
        let lasers_ptr = unsafe { read_ptr(self.base, LASERS_PTR) };
        let player_ptr = unsafe { read_ptr(self.base, PLAYER_PTR) };

        if bullets_ptr.is_null()
            || enemies_ptr.is_null()
            || items_ptr.is_null()
            || lasers_ptr.is_null()
            || player_ptr.is_null()
        {
            return None;
        }

        Some(GameState {
            bullets: get_bullets(bullets_ptr),
            enemies: get_enemies(enemies_ptr),
            power_items: get_power_items(items_ptr),
            lasers: get_lasers(lasers_ptr),
            player: get_player(player_ptr),
        })
    }
}

impl Default for StateReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Gets the current state of all bullets on the playing field.
fn get_bullets(bullets_ptr: *const u8) -> Vec<Bullet> {
    let mut bullets = Vec::new();

    let mut node = unsafe { bullets_ptr.add(BULLETS_LIST) };

    while !node.is_null() {
        let data = unsafe { read_ptr(node, 0) };
        node = unsafe { read_ptr(node, BULLET_NEXT_PTR) };

        if data.is_null() {
            continue;
        }

        let is_active = unsafe { read_field::<u16>(data, BULLET_STATE) } == 1;

        if is_active {
            let [pos_x, pos_y] = unsafe { read_field::<[f32; 2]>(data, BULLET_POS) };
            let hitbox_radius = unsafe { read_field::<f32>(data, BULLET_HITBOX_RADIUS) };

            // Check if bullet is in bounds
            if pos_y >= -hitbox_radius
                && pos_y <= WORLD_HEIGHT + hitbox_radius
                && pos_x >= const { -WORLD_WIDTH / 2. } - hitbox_radius
                && pos_x <= const { WORLD_WIDTH / 2. } + hitbox_radius
            {
                let [vel_x, vel_y] = unsafe { read_field::<[f32; 2]>(data, BULLET_VEL) };

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

    bullets
}

/// Gets the current state of all enemies on the playing field.
fn get_enemies(enemies_ptr: *const u8) -> Vec<Enemy> {
    let mut enemies = Vec::new();

    // Get the pointer to the head of the linked list of enemies
    let mut node = unsafe { read_ptr(enemies_ptr, ENEMIES_LIST) };

    while !node.is_null() {
        let data = unsafe { read_ptr(node, 0) };
        node = unsafe { read_ptr(node, ENEMY_NEXT_PTR) };

        if data.is_null() {
            continue;
        }

        let flags = unsafe { read_field::<u32>(data, ENEMY_FLAGS) };
        let is_boss = flags & ENEMY_BOSS_FLAG != 0;
        let has_anm_vm_id = unsafe { read_field::<u32>(data, ENEMY_ANM_VM_ID) } != 0;

        // Check if the enemy is "real"; i.e. is a boss or has an ANM VM ID set.
        // The game uses "fake" enemies to make certain bullet patterns easier to implement.
        // However, the player cannot interact with these enemies,
        // so they should not be counted as distinct entities.
        if is_boss || has_anm_vm_id {
            let [pos_x, pos_y] = unsafe { read_field::<[f32; 2]>(data, ENEMY_POS) };
            let [vel_x, vel_y] = unsafe { read_field::<[f32; 2]>(data, ENEMY_VEL) };
            let hitbox_radius = unsafe { read_field::<f32>(data, ENEMY_HITBOX_RADIUS) };

            #[allow(clippy::cast_precision_loss)]
            let hp_ratio = {
                let hp = unsafe { read_field::<i32>(data, ENEMY_HP) } as f32;
                let max_hp = unsafe { read_field::<i32>(data, ENEMY_MAX_HP) } as f32;
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

    enemies
}

/// Gets the current state of all power items on the playing field.
fn get_power_items(items_ptr: *const u8) -> Vec<PowerItem> {
    let mut items = Vec::new();

    for i in 0..ITEMS_CAP {
        let item = unsafe { items_ptr.add(ITEMS_ARRAY + i * ITEM_BYTE_LEN) };

        let state = unsafe { read_field::<u32>(item, ITEM_STATE) };
        let item_type = unsafe { read_field::<u32>(item, ITEM_TYPE) };

        let is_active = state != 0;
        let is_power_item = matches!(item_type, 1 | 3 | 8);

        if is_active && is_power_item {
            let [pos_x, pos_y] = unsafe { read_field::<[f32; 2]>(item, ITEM_POS) };
            let [vel_x, vel_y] = unsafe { read_field::<[f32; 2]>(item, ITEM_VEL) };

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
fn get_lasers(lasers_ptr: *const u8) -> Lasers {
    let mut segment_lasers = Vec::new();
    let mut ray_lasers = Vec::new();
    let mut curve_lasers = Vec::new();

    // Get the pointer to the head of the linked list of lasers
    let mut laser = unsafe { read_ptr(lasers_ptr, LASERS_LIST) };

    loop {
        let next = unsafe { read_ptr(laser, LASER_NEXT_PTR) };

        if next.is_null() {
            break;
        }

        let laser_type = unsafe { read_field::<u32>(laser, LASER_TYPE) };
        let width = unsafe { read_field::<f32>(laser, LASER_WIDTH) };

        match laser_type {
            0 => {
                let [head_pos_x, head_pos_y] = unsafe { read_field::<[f32; 2]>(laser, LASER_POS) };
                let (sin_angle, cos_angle) =
                    unsafe { read_field::<f32>(laser, LASER_ANGLE) }.sin_cos();
                let length = unsafe { read_field::<f32>(laser, LASER_LENGTH) };
                let speed = unsafe { read_field::<f32>(laser, LASER_SPEED) };

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
                let [origin_pos_x, origin_pos_y] =
                    unsafe { read_field::<[f32; 2]>(laser, LASER_POS) };
                let (sin_angle, cos_angle) =
                    unsafe { read_field::<f32>(laser, LASER_ANGLE) }.sin_cos();
                let [origin_vel_x, origin_vel_y] =
                    unsafe { read_field::<[f32; 2]>(laser, RAY_LASER_ORIGIN_VEL) };
                let angular_vel = unsafe { read_field::<f32>(laser, RAY_LASER_ANGULAR_VEL) };

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
                let num_nodes = unsafe { read_field::<u32>(laser, CURVE_LASER_NUM_NODES) } as usize;
                let nodes_base = unsafe { read_ptr(laser, CURVE_LASER_NODES_ARRAY) };

                let mut nodes = Vec::with_capacity(num_nodes);

                for i in 0..num_nodes {
                    let node = unsafe { nodes_base.add(i * CURVE_LASER_NODE_BYTE_LEN) };

                    let [pos_x, pos_y] =
                        unsafe { read_field::<[f32; 2]>(node, CURVE_LASER_NODE_POS) };
                    let (sin_angle, cos_angle) =
                        unsafe { read_field::<f32>(node, CURVE_LASER_NODE_ANGLE) }.sin_cos();
                    let speed = unsafe { read_field::<f32>(node, CURVE_LASER_NODE_SPEED) };

                    nodes.push(CurveLaserNode {
                        pos_x,
                        pos_y,
                        vel_x: speed * cos_angle,
                        vel_y: speed * sin_angle,
                    });
                }

                curve_lasers.push(CurveLaser { nodes, width });
            }
            id => panic!("unknown laser type (type {id})"),
        }

        laser = next;
    }

    Lasers {
        segments: segment_lasers,
        rays: ray_lasers,
        curves: curve_lasers,
    }
}

/// Gets the current state of the player.
fn get_player(player_ptr: *const u8) -> Player {
    let [pos_x, pos_y] = unsafe { read_field::<[f32; 2]>(player_ptr, PLAYER_POS) };
    let is_focused = unsafe { read_field::<u32>(player_ptr, PLAYER_IS_FOCUSED) } == 1;
    let hitbox_radius = unsafe { read_field::<f32>(player_ptr, PLAYER_HITBOX_RADIUS) };

    Player {
        pos_x,
        pos_y,
        is_focused,
        hitbox_radius,
    }
}

/// Reads a value of type `T` at the given byte offset from `base`.
///
/// # Safety
/// `base.add(offset)` must point to a valid, initialized `T`.
unsafe fn read_field<T>(base: *const u8, offset: usize) -> T {
    unsafe { base.add(offset).cast::<T>().read_unaligned() }
}

/// Reads a 32-bit game pointer at the given byte offset from `base`
/// and reconstitutes it as a Rust pointer via [`with_exposed_provenance`].
///
/// # Safety
/// `base.add(offset)` must point to a valid `u32`.
unsafe fn read_ptr(base: *const u8, offset: usize) -> *const u8 {
    let raw = unsafe { read_field::<u32>(base, offset) };
    with_exposed_provenance(raw as usize)
}
