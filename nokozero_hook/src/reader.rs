//! Logic for reading game entity data from process memory.

use crate::addrs::{
    BOMB_FRAGMENTS_VA, BOMBS_VA, BULLET_MANAGER_PTR_VA, ENEMY_MANAGER_PTR_VA, GAME_TICK_VA,
    GRAZE_VA, ITEM_MANAGER_PTR_VA, LASER_MANAGER_PTR_VA, LIFE_FRAGMENTS_VA, LIVES_VA,
    PLAYER_PTR_VA, POWER_VA, SCORE_DIV10_VA, VALUE_VA,
};
use crate::log::fatal;
use crate::mem::{game_live, read};
use std::ptr::{NonNull, with_exposed_provenance_mut};

const BULLETS_LIST: usize = 0x68;
const BULLET_POS: usize = 0xc38;
const BULLET_VEL: usize = 0xc44;
const BULLET_HITBOX_RADIUS: usize = 0xc58;
const BULLET_STATE: usize = 0xc8a;

const ENEMIES_LIST: usize = 0x180;
const ENEMY_POS: usize = 0x120c + 0x44;
const ENEMY_VEL: usize = 0x120c + 0x78;
const ENEMY_HITBOX_RADIUS: usize = 0x120c + 0x118;
const ENEMY_ANM_VM_ID: usize = 0x120c + 0x124;
const ENEMY_HP: usize = 0x120c + 0x3f74;
const ENEMY_MAX_HP: usize = 0x120c + 0x3f78;
const ENEMY_INVULN_TIMER: usize = 0x120c + 0x3ffc;
const ENEMY_FLAGS: usize = 0x120c + 0x4060;
const ENEMY_BOSS_FLAG: u32 = 1 << 23;
/// Bomb shield raised, the ECL invincibility flag, and hidden, respectively.
/// Any of the three makes the enemy undamageable on its own, independently of [`ENEMY_INVULN_TIMER`].
const ENEMY_INVULN_FLAGS: u32 = (1 << 0) | (1 << 4) | (1 << 5);

const ITEMS_ARRAY: usize = 0x0;
const ITEM_POS: usize = 0xc30;
const ITEM_VEL: usize = 0xc3c;
const ITEM_STATE: usize = 0xc74;
const ITEM_TYPE: usize = 0xc78;
const ITEM_BYTE_LEN: usize = 0xc88;
// Reference: https://github.com/exphp-share/th-re-data/blob/41cd633354f3bbc4ff11b3d315ef7243c990f227/data/th15.v1.00b/type-structs-own.json#L1025
const ITEMS_CAP: usize = 600;

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
const CURVE_LASER_MAX_NODES: usize = 128;
const CURVE_LASER_NODES_ARRAY: usize = 0x5d4 + 0xf68;
const CURVE_LASER_NODE_POS: usize = 0x0;
const CURVE_LASER_NODE_ANGLE: usize = 0x18;
const CURVE_LASER_NODE_SPEED: usize = 0x1c;
const CURVE_LASER_NODE_BYTE_LEN: usize = 0x20;

const PLAYER_POS: usize = 0x618;
const PLAYER_IS_FOCUSED: usize = 0x16240;
const PLAYER_HITBOX_RADIUS: usize = 0x2bfc8;

pub(crate) struct GameState {
    pub(crate) bullets: Vec<Bullet>,
    pub(crate) enemies: Vec<Enemy>,
    pub(crate) items: Vec<Item>,
    pub(crate) lasers: Lasers,
    pub(crate) player: Player,
}

impl GameState {
    pub(crate) fn new() -> Self {
        Self {
            bullets: Vec::with_capacity(1024),
            enemies: Vec::new(),
            items: Vec::new(),
            lasers: Lasers {
                segments: Vec::new(),
                rays: Vec::new(),
                curve_points: Vec::new(),
            },
            player: Player::default(),
        }
    }

    /// Reads the current game state, returning it if a stage is fully live and returning `None` otherwise.
    #[must_use]
    pub(crate) fn read(&mut self) -> Option<&Self> {
        self.clear();

        if !unsafe { game_live() } {
            return None;
        }
        let (
            Some(bullets_ptr),
            Some(enemies_ptr),
            Some(items_ptr),
            Some(lasers_ptr),
            Some(player_ptr),
        ) = (unsafe {
            (
                GamePtr::read_global(BULLET_MANAGER_PTR_VA),
                GamePtr::read_global(ENEMY_MANAGER_PTR_VA),
                GamePtr::read_global(ITEM_MANAGER_PTR_VA),
                GamePtr::read_global(LASER_MANAGER_PTR_VA),
                GamePtr::read_global(PLAYER_PTR_VA),
            )
        })
        else {
            return None;
        };

        get_bullets(bullets_ptr, &mut self.bullets);
        get_enemies(enemies_ptr, &mut self.enemies);
        get_items(items_ptr, &mut self.items);
        get_lasers(lasers_ptr, &mut self.lasers);
        self.player = get_player(player_ptr);
        Some(self)
    }

    fn clear(&mut self) {
        self.bullets.clear();
        self.enemies.clear();
        self.items.clear();
        self.lasers.segments.clear();
        self.lasers.rays.clear();
        self.lasers.curve_points.clear();
        self.player = Player::default();
    }
}

pub(crate) struct Bullet {
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) vel_x: f32,
    pub(crate) vel_y: f32,
    pub(crate) hitbox_radius: f32,
}

pub(crate) struct Enemy {
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) vel_x: f32,
    pub(crate) vel_y: f32,
    pub(crate) hitbox_radius: f32,
    pub(crate) hp_ratio: f32,
    pub(crate) max_hp: i32,
    pub(crate) is_boss: bool,
    pub(crate) is_invulnerable: bool,
    pub(crate) invuln_frames: i32,
}

pub(crate) struct Item {
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) vel_x: f32,
    pub(crate) vel_y: f32,
    pub(crate) kind: u32,
}

pub(crate) struct Lasers {
    pub(crate) segments: Vec<SegmentLaser>,
    pub(crate) rays: Vec<RayLaser>,
    /// Curve lasers flattened to their nodes.
    pub(crate) curve_points: Vec<CurvePoint>,
}

pub(crate) struct SegmentLaser {
    pub(crate) head_pos_x: f32,
    pub(crate) head_pos_y: f32,
    pub(crate) vel_x: f32,
    pub(crate) vel_y: f32,
    pub(crate) length: f32,
    pub(crate) width: f32,
}

pub(crate) struct RayLaser {
    pub(crate) origin_pos_x: f32,
    pub(crate) origin_pos_y: f32,
    pub(crate) origin_vel_x: f32,
    pub(crate) origin_vel_y: f32,
    pub(crate) cos_angle: f32,
    pub(crate) sin_angle: f32,
    pub(crate) angular_vel: f32,
    pub(crate) width: f32,
}

pub(crate) struct CurvePoint {
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) vel_x: f32,
    pub(crate) vel_y: f32,
    pub(crate) width: f32,
}

#[derive(Default)]
pub(crate) struct Player {
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) is_focused: bool,
    pub(crate) hitbox_radius: f32,
}

pub(crate) struct Resources {
    /// Frames since the stage started.
    pub(crate) game_tick: u32,
    pub(crate) score_div10: u32,
    pub(crate) graze: i32,
    pub(crate) value_x100: i32,
    pub(crate) power: i32,
    pub(crate) lives: i32,
    pub(crate) life_fragments: i32,
    pub(crate) bombs: i32,
    pub(crate) bomb_fragments: i32,
}

impl Resources {
    #[must_use]
    pub(crate) fn read() -> Self {
        unsafe {
            Self {
                game_tick: read(GAME_TICK_VA),
                score_div10: read(SCORE_DIV10_VA),
                graze: read(GRAZE_VA),
                value_x100: read(VALUE_VA),
                power: read(POWER_VA),
                lives: read(LIVES_VA),
                life_fragments: read(LIFE_FRAGMENTS_VA),
                bombs: read(BOMBS_VA),
                bomb_fragments: read(BOMB_FRAGMENTS_VA),
            }
        }
    }
}

/// A non-null read-only pointer into process memory.
#[derive(Clone, Copy)]
struct GamePtr(NonNull<u8>);

impl GamePtr {
    fn from_addr(addr: usize) -> Option<Self> {
        NonNull::new(with_exposed_provenance_mut(addr)).map(Self)
    }

    /// # Safety
    ///
    /// `addr` must point to a readable, mapped `u32`.
    unsafe fn read_global(addr: usize) -> Option<Self> {
        Self::from_addr(unsafe { read::<u32>(addr) } as usize)
    }

    /// # Safety
    ///
    /// `self + offset` must remain within a valid allocation.
    unsafe fn byte_add(self, offset: usize) -> Self {
        Self(unsafe { self.0.byte_add(offset) })
    }

    /// # Safety
    ///
    /// `self + offset` must point to a valid, initialized `T`.
    unsafe fn read<T>(self, offset: usize) -> T {
        unsafe { self.0.byte_add(offset).cast().read_unaligned() }
    }

    /// # Safety
    ///
    /// `self + offset` must point to a valid, initialized `u32`.
    unsafe fn read_ptr(self, offset: usize) -> Option<Self> {
        let raw: u32 = unsafe { self.read(offset) };
        Self::from_addr(raw as usize)
    }
}

/// An indirect linked list. Iterating yields each data pointer.
struct List {
    node: Option<GamePtr>,
}

impl List {
    /// `head` must be `None` or point to the first entry of a well-formed list.
    fn new(head: Option<GamePtr>) -> Self {
        Self { node: head }
    }
}

impl Iterator for List {
    type Item = GamePtr;

    fn next(&mut self) -> Option<GamePtr> {
        const NEXT: usize = 4;

        while let Some(node) = self.node {
            let data = unsafe { node.read_ptr(0) };
            self.node = unsafe { node.read_ptr(NEXT) };
            if let Some(data) = data {
                return Some(data);
            }
        }
        None
    }
}

/// A linked list of laser entities terminated by a tail sentinel whose next node is null.
struct LaserList {
    node: GamePtr,
}

impl LaserList {
    fn new(head: GamePtr) -> Self {
        Self { node: head }
    }
}

impl Iterator for LaserList {
    type Item = GamePtr;

    fn next(&mut self) -> Option<GamePtr> {
        let next = unsafe { self.node.read_ptr(LASER_NEXT_PTR) }?;
        let current = self.node;
        self.node = next;
        Some(current)
    }
}

fn get_bullets(bullets_ptr: GamePtr, bullets: &mut Vec<Bullet>) {
    let head = unsafe { bullets_ptr.byte_add(BULLETS_LIST) };
    for data in List::new(Some(head)) {
        let state = unsafe { data.read::<u16>(BULLET_STATE) };
        let is_lethal = state == 1 || state == 2;
        if !is_lethal {
            continue;
        }

        let [pos_x, pos_y] = unsafe { data.read::<[f32; 2]>(BULLET_POS) };
        let [vel_x, vel_y] = unsafe { data.read::<[f32; 2]>(BULLET_VEL) };
        let hitbox_radius = unsafe { data.read::<f32>(BULLET_HITBOX_RADIUS) };

        bullets.push(Bullet {
            pos_x,
            pos_y,
            vel_x,
            vel_y,
            hitbox_radius,
        });
    }
}

fn get_enemies(enemies_ptr: GamePtr, enemies: &mut Vec<Enemy>) {
    let head = unsafe { enemies_ptr.read_ptr(ENEMIES_LIST) };
    for data in List::new(head) {
        let flags = unsafe { data.read::<u32>(ENEMY_FLAGS) };
        let is_boss = flags & ENEMY_BOSS_FLAG != 0;
        let has_anm_vm_id = unsafe { data.read::<u32>(ENEMY_ANM_VM_ID) } != 0;

        // Check if the enemy is "real" (i.e. is a boss or has an ANM VM ID set).
        // The game uses "fake" enemies to make certain bullet patterns easier to implement.
        // However, the player cannot interact with these enemies, so they should not be counted as distinct entities.
        if !(is_boss || has_anm_vm_id) {
            continue;
        }

        let [pos_x, pos_y] = unsafe { data.read::<[f32; 2]>(ENEMY_POS) };
        let [vel_x, vel_y] = unsafe { data.read::<[f32; 2]>(ENEMY_VEL) };
        let hitbox_radius = unsafe { data.read::<f32>(ENEMY_HITBOX_RADIUS) };

        let max_hp = unsafe { data.read::<i32>(ENEMY_MAX_HP) };
        #[expect(clippy::cast_precision_loss)]
        let hp_ratio = if max_hp > 0 {
            let hp = unsafe { data.read::<i32>(ENEMY_HP) } as f32;
            (hp / max_hp as f32).clamp(0., 1.)
        } else {
            0.
        };

        let invuln_frames = unsafe { data.read::<i32>(ENEMY_INVULN_TIMER) }.max(0);
        let is_invulnerable = flags & ENEMY_INVULN_FLAGS != 0 || invuln_frames > 0;

        enemies.push(Enemy {
            pos_x,
            pos_y,
            vel_x,
            vel_y,
            hitbox_radius,
            hp_ratio,
            max_hp,
            is_boss,
            is_invulnerable,
            invuln_frames,
        });
    }
}

fn get_items(items_ptr: GamePtr, items: &mut Vec<Item>) {
    for i in 0..ITEMS_CAP {
        let item = unsafe { items_ptr.byte_add(ITEMS_ARRAY + i * ITEM_BYTE_LEN) };

        if unsafe { item.read::<u32>(ITEM_STATE) } == 0 {
            continue;
        }

        let [pos_x, pos_y] = unsafe { item.read::<[f32; 2]>(ITEM_POS) };
        let [vel_x, vel_y] = unsafe { item.read::<[f32; 2]>(ITEM_VEL) };
        let kind = unsafe { item.read::<u32>(ITEM_TYPE) };

        items.push(Item {
            pos_x,
            pos_y,
            vel_x,
            vel_y,
            kind,
        });
    }
}

fn get_lasers(lasers_ptr: GamePtr, lasers: &mut Lasers) {
    if let Some(head) = unsafe { lasers_ptr.read_ptr(LASERS_LIST) } {
        for laser in LaserList::new(head) {
            let laser_type = unsafe { laser.read::<u32>(LASER_TYPE) };
            let width = unsafe { laser.read::<f32>(LASER_WIDTH) };

            match laser_type {
                0 => {
                    let [head_pos_x, head_pos_y] = unsafe { laser.read::<[f32; 2]>(LASER_POS) };
                    let (sin_angle, cos_angle) =
                        unsafe { laser.read::<f32>(LASER_ANGLE) }.sin_cos();
                    let length = unsafe { laser.read::<f32>(LASER_LENGTH) };
                    let speed = unsafe { laser.read::<f32>(LASER_SPEED) };

                    lasers.segments.push(SegmentLaser {
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

                    lasers.rays.push(RayLaser {
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
                    if num_nodes > CURVE_LASER_MAX_NODES {
                        fatal!("unexpected curve laser node count {num_nodes}");
                    }
                    if let Some(base) = unsafe { laser.read_ptr(CURVE_LASER_NODES_ARRAY) } {
                        for i in 0..num_nodes {
                            let node = unsafe { base.byte_add(i * CURVE_LASER_NODE_BYTE_LEN) };

                            let [pos_x, pos_y] =
                                unsafe { node.read::<[f32; 2]>(CURVE_LASER_NODE_POS) };
                            let (sin_angle, cos_angle) =
                                unsafe { node.read::<f32>(CURVE_LASER_NODE_ANGLE) }.sin_cos();
                            let speed = unsafe { node.read::<f32>(CURVE_LASER_NODE_SPEED) };

                            lasers.curve_points.push(CurvePoint {
                                pos_x,
                                pos_y,
                                vel_x: speed * cos_angle,
                                vel_y: speed * sin_angle,
                                width,
                            });
                        }
                    }
                }
                _ => fatal!("unexpected laser type {laser_type}"),
            }
        }
    }
}

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
