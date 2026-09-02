//! Serialization of game state into observation payloads.
//!
//! A payload is a [`META_WORDS`]-word meta block followed by six entity sections. Each section has a `count` field of type `u32`
//! followed by `count` rows of the matching [`SECTION_WIDTHS`] entry. Every field is 4 bytes little-endian so the entire payload
//! can parse as an array with 4-byte elements. Enemy `max_hp` / `invuln_frames` and item `kind` are `f32` so rows are homogenous
//! and parsing is easier. These values are still exact below 2^24. Flag words are split into their low and high 16 bits,
//! each exact as an `f32`.

use crate::practice::WireMeta;
use crate::reader::{
    Bullet, CurvePoint, Enemy, GameState, Item, RayLaser, Resources, SegmentLaser,
};

const META_WORDS: usize = 32;
/// Bullets; enemies; items; segment lasers; ray lasers; curve laser points.
const SECTION_WIDTHS: [usize; 6] = [
    Bullet::WIDTH,
    Enemy::WIDTH,
    Item::WIDTH,
    SegmentLaser::WIDTH,
    RayLaser::WIDTH,
    CurvePoint::WIDTH,
];

/// Bit in the frame flags word to indicate that the hook rewrote the controller's action on at least one frame
/// since the previous exchange. This is set on the observation following the overridden frames.
const FLAG_INPUT_OVERRIDDEN: u32 = 1 << 31;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scene {
    Other = 0,
    Menu = 1,
    InGame = 2,
}

pub(crate) struct Meta {
    pub(crate) step: u32,
    pub(crate) scene: Scene,
    pub(crate) wire: WireMeta,
    /// Sets [`FLAG_INPUT_OVERRIDDEN`] in the frame flags word.
    pub(crate) overrode_input: bool,
}

/// Appends the observation payload to `buf`.
pub(crate) fn build(buf: &mut Vec<u8>, state: Option<&GameState>, meta: &Meta, res: &Resources) {
    let rows = state.map_or(0, |s| {
        s.bullets.len() * Bullet::WIDTH
            + s.enemies.len() * Enemy::WIDTH
            + s.items.len() * Item::WIDTH
            + s.lasers.segments.len() * SegmentLaser::WIDTH
            + s.lasers.rays.len() * RayLaser::WIDTH
            + s.lasers.curve_points.len() * CurvePoint::WIDTH
    });
    buf.reserve(4 * (META_WORDS + SECTION_WIDTHS.len() + rows));

    let start = buf.len();
    put_meta(buf, state, meta, res);
    debug_assert_eq!(
        buf.len() - start,
        4 * META_WORDS,
        "meta block width must match META_WORDS",
    );

    let Some(state) = state else {
        for _ in SECTION_WIDTHS {
            put_count(buf, 0);
        }
        return;
    };

    put_section(buf, &state.bullets, Bullet::cells);
    put_section(buf, &state.enemies, Enemy::cells);
    put_section(buf, &state.items, Item::cells);
    put_section(buf, &state.lasers.segments, SegmentLaser::cells);
    put_section(buf, &state.lasers.rays, RayLaser::cells);
    put_section(buf, &state.lasers.curve_points, CurvePoint::cells);
}

#[expect(clippy::cast_possible_truncation)]
fn split_flags(flags: u32) -> [f32; 2] {
    [f32::from(flags as u16), f32::from((flags >> 16) as u16)]
}

impl Bullet {
    const WIDTH: usize = 9;

    fn cells(&self) -> [f32; Self::WIDTH] {
        let [flags_lo, flags_hi] = split_flags(self.flags);
        [
            self.pos_x,
            self.pos_y,
            self.vel_x,
            self.vel_y,
            self.size_w,
            self.size_h,
            self.scale,
            flags_lo,
            flags_hi,
        ]
    }
}

impl Enemy {
    const WIDTH: usize = 17;

    #[expect(clippy::cast_precision_loss)]
    fn cells(&self) -> [f32; Self::WIDTH] {
        let [flags_lo, flags_hi] = split_flags(self.flags);
        [
            self.pos_x,
            self.pos_y,
            self.vel_x,
            self.vel_y,
            self.hitbox_w,
            self.hitbox_h,
            self.hurtbox_w,
            self.hurtbox_h,
            self.hp_ratio,
            self.max_hp as f32,
            f32::from(u8::from(self.is_boss)),
            f32::from(u8::from(self.is_invulnerable)),
            self.invuln_frames as f32,
            f32::from(u8::from(self.is_lethal)),
            self.no_hitbox_frames as f32,
            flags_lo,
            flags_hi,
        ]
    }
}

impl Item {
    const WIDTH: usize = 5;

    #[expect(clippy::cast_precision_loss)]
    fn cells(&self) -> [f32; Self::WIDTH] {
        [
            self.pos_x,
            self.pos_y,
            self.vel_x,
            self.vel_y,
            self.kind as f32,
        ]
    }
}

impl SegmentLaser {
    const WIDTH: usize = 6;

    fn cells(&self) -> [f32; Self::WIDTH] {
        [
            self.head_pos_x,
            self.head_pos_y,
            self.vel_x,
            self.vel_y,
            self.length,
            self.width,
        ]
    }
}

impl RayLaser {
    const WIDTH: usize = 8;

    fn cells(&self) -> [f32; Self::WIDTH] {
        [
            self.origin_pos_x,
            self.origin_pos_y,
            self.origin_vel_x,
            self.origin_vel_y,
            self.cos_angle,
            self.sin_angle,
            self.angular_vel,
            self.width,
        ]
    }
}

impl CurvePoint {
    const WIDTH: usize = 5;

    fn cells(&self) -> [f32; Self::WIDTH] {
        [self.pos_x, self.pos_y, self.vel_x, self.vel_y, self.width]
    }
}

fn put_meta(buf: &mut Vec<u8>, state: Option<&GameState>, meta: &Meta, res: &Resources) {
    put_u32(buf, meta.step);
    put_u32(buf, meta.scene as u32);
    put_u32(buf, state.is_some().into());
    put_u32(buf, meta.wire.load_generation.to_wire());
    put_u32(buf, meta.wire.reset_seq);
    put_u32(buf, meta.wire.reset_outcome as u32);
    put_u32(buf, meta.wire.applied_section);
    put_u32(buf, meta.wire.hits);
    let frame_flags = if meta.overrode_input {
        FLAG_INPUT_OVERRIDDEN
    } else {
        0
    };
    put_u32(buf, frame_flags);
    put_u32(buf, res.game_tick);
    put_u32(buf, res.score_div10);
    put_i32(buf, res.graze);
    put_i32(buf, res.value_x100);
    put_i32(buf, res.power);
    put_i32(buf, res.lives);
    put_i32(buf, res.life_fragments);
    put_i32(buf, res.bombs);
    put_i32(buf, res.bomb_fragments);
    put_u32(buf, res.rng_state);
    put_u32(buf, res.rng_count);
    put_u32(buf, res.chapter);
    put_u32(buf, res.time_in_chapter);
    put_i32(buf, res.spell_id);
    put_i32(buf, res.miss_count);
    put_i32(buf, res.spell_timer);
    put_u32(buf, res.spell_flags);
    put_u32(buf, res.mode_flags);
    match state {
        Some(s) => put_f32s(
            buf,
            [
                s.player.pos_x,
                s.player.pos_y,
                f32::from(u8::from(s.player.is_focused)),
                s.player.hitbox_radius,
                s.player.hit_radius,
            ],
        ),
        None => put_f32s(buf, [0.; 5]),
    }
}

/// Appends one entity section consisting of a `u32` row count followed by each row's cells.
fn put_section<T, const N: usize>(buf: &mut Vec<u8>, rows: &[T], cells: impl Fn(&T) -> [f32; N]) {
    put_count(buf, rows.len());
    for row in rows {
        put_f32s(buf, cells(row));
    }
}

fn put_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_f32s<const N: usize>(buf: &mut Vec<u8>, values: [f32; N]) {
    buf.extend_from_slice(values.map(f32::to_le_bytes).as_flattened());
}

fn put_count(buf: &mut Vec<u8>, n: usize) {
    #[expect(clippy::cast_possible_truncation)]
    put_u32(buf, n as u32);
}

#[cfg(test)]
mod tests {
    use super::{FLAG_INPUT_OVERRIDDEN, META_WORDS, Meta, Resources, SECTION_WIDTHS, Scene, build};
    use crate::practice::{Generation, Outcome, WireMeta};
    use crate::reader::{Bullet, CurvePoint, Enemy, GameState, Item, Lasers, Player, SegmentLaser};

    fn resources() -> Resources {
        Resources {
            game_tick: 1234,
            score_div10: 567,
            graze: 89,
            value_x100: 500_000,
            power: 350,
            lives: 5,
            life_fragments: 1,
            bombs: 3,
            bomb_fragments: 2,
            rng_state: 0xBEEF,
            rng_count: 44,
            chapter: 3,
            time_in_chapter: 77,
            spell_id: -1,
            miss_count: 2,
            spell_timer: -5,
            spell_flags: 0x8000_0001,
            mode_flags: 0x18,
        }
    }

    fn payload_len(counts: [usize; 6]) -> usize {
        4 * (META_WORDS
            + counts
                .iter()
                .zip(SECTION_WIDTHS)
                .map(|(count, width)| 1 + count * width)
                .sum::<usize>())
    }

    fn wire_for_test() -> WireMeta {
        WireMeta {
            load_generation: Generation::for_test(0),
            reset_seq: 0,
            reset_outcome: Outcome::Idle,
            applied_section: 0,
            hits: 0,
        }
    }

    fn meta_for_test() -> Meta {
        Meta {
            step: 0,
            scene: Scene::Menu,
            wire: wire_for_test(),
            overrode_input: false,
        }
    }

    fn f32_at(buf: &[u8], word: usize) -> f32 {
        f32::from_le_bytes(buf[(word * 4)..(word * 4 + 4)].try_into().unwrap())
    }

    fn u32_at(buf: &[u8], word: usize) -> u32 {
        u32::from_le_bytes(buf[(word * 4)..(word * 4 + 4)].try_into().unwrap())
    }

    fn i32_at(buf: &[u8], word: usize) -> i32 {
        i32::from_le_bytes(buf[(word * 4)..(word * 4 + 4)].try_into().unwrap())
    }

    #[test]
    #[expect(clippy::float_cmp)]
    fn outside_stage_layout() {
        let mut buf = Vec::new();
        build(
            &mut buf,
            None,
            &Meta {
                step: 42,
                scene: Scene::Menu,
                wire: WireMeta {
                    load_generation: Generation::for_test(9),
                    reset_seq: 7,
                    reset_outcome: Outcome::Pending,
                    applied_section: 1201,
                    hits: 3,
                },
                overrode_input: true,
            },
            &resources(),
        );
        assert_eq!(buf.len(), payload_len([0; 6]));
        assert_eq!(u32_at(&buf, 0), 42); // step
        assert_eq!(u32_at(&buf, 1), Scene::Menu as u32); // scene
        assert_eq!(u32_at(&buf, 2), 0); // in_stage
        assert_eq!(u32_at(&buf, 3), 9); // load_generation
        assert_eq!(u32_at(&buf, 4), 7); // reset_seq
        assert_eq!(u32_at(&buf, 5), 1); // reset_outcome
        assert_eq!(u32_at(&buf, 6), 1201); // applied_section
        assert_eq!(u32_at(&buf, 7), 3); // hits
        assert_eq!(u32_at(&buf, 8), FLAG_INPUT_OVERRIDDEN); // frame flags
        assert_eq!(u32_at(&buf, 9), 1234); // game_tick
        assert_eq!(u32_at(&buf, 10), 567); // score_div10
        assert_eq!(i32_at(&buf, 11), 89); // graze
        assert_eq!(i32_at(&buf, 12), 500_000); // value_x100
        assert_eq!(i32_at(&buf, 17), 2); // bomb_fragments
        assert_eq!(u32_at(&buf, 18), 0xBEEF); // rng_state
        assert_eq!(u32_at(&buf, 19), 44); // rng_count
        assert_eq!(u32_at(&buf, 20), 3); // chapter
        assert_eq!(u32_at(&buf, 21), 77); // time_in_chapter
        assert_eq!(i32_at(&buf, 22), -1); // spell_id
        assert_eq!(i32_at(&buf, 23), 2); // miss_count
        assert_eq!(i32_at(&buf, 24), -5); // spell_timer
        assert_eq!(u32_at(&buf, 25), 0x8000_0001); // spell_flags
        assert_eq!(u32_at(&buf, 26), 0x18); // mode_flags
        // player block
        for word in 27..32 {
            assert_eq!(f32_at(&buf, word), 0.);
        }
        for section in 0..6 {
            assert_eq!(u32_at(&buf, META_WORDS + section), 0);
        }
    }

    #[test]
    #[expect(clippy::float_cmp, clippy::too_many_lines)]
    fn inside_stage_layout() {
        let state = GameState {
            bullets: vec![Bullet {
                pos_x: 1.,
                pos_y: 2.,
                vel_x: 3.,
                vel_y: 4.,
                size_w: 5.,
                size_h: 6.,
                scale: 7.,
                flags: 0x0001_0012,
            }],
            enemies: vec![Enemy {
                pos_x: 10.,
                pos_y: 11.,
                vel_x: 12.,
                vel_y: 13.,
                hitbox_w: 14.,
                hitbox_h: 15.,
                hurtbox_w: 16.,
                hurtbox_h: 17.,
                hp_ratio: 0.5,
                max_hp: 60000,
                is_boss: true,
                is_invulnerable: true,
                invuln_frames: 42,
                is_lethal: false,
                no_hitbox_frames: 9,
                flags: 0x0400_0002,
            }],
            items: vec![Item {
                pos_x: 20.,
                pos_y: 21.,
                vel_x: 22.,
                vel_y: 23.,
                kind: 8,
            }],
            lasers: Lasers {
                segments: vec![SegmentLaser {
                    head_pos_x: 30.,
                    head_pos_y: 31.,
                    vel_x: 32.,
                    vel_y: 33.,
                    length: 34.,
                    width: 35.,
                }],
                rays: vec![],
                curve_points: vec![
                    CurvePoint {
                        pos_x: 40.,
                        pos_y: 41.,
                        vel_x: 42.,
                        vel_y: 43.,
                        width: 48.,
                    },
                    CurvePoint {
                        pos_x: 44.,
                        pos_y: 45.,
                        vel_x: 46.,
                        vel_y: 47.,
                        width: 48.,
                    },
                ],
            },
            player: Player {
                pos_x: -100.5,
                pos_y: 200.25,
                is_focused: true,
                hitbox_radius: 2.,
                hit_radius: 3.,
            },
        };

        let mut buf = Vec::new();
        build(
            &mut buf,
            Some(&state),
            &Meta {
                step: 6,
                scene: Scene::InGame,
                wire: WireMeta {
                    load_generation: Generation::for_test(2),
                    hits: 1,
                    ..wire_for_test()
                },
                ..meta_for_test()
            },
            &resources(),
        );
        assert_eq!(buf.len(), payload_len([1, 1, 1, 1, 0, 2]));
        assert_eq!(u32_at(&buf, 2), 1); // in_stage
        assert_eq!(u32_at(&buf, 3), 2); // load_generation
        assert_eq!(u32_at(&buf, 7), 1); // hits
        assert_eq!(u32_at(&buf, 8), 0); // frame flags
        assert_eq!(u32_at(&buf, 18), 0xBEEF); // rng_state
        // player block
        assert_eq!(f32_at(&buf, 27), -100.5);
        assert_eq!(f32_at(&buf, 28), 200.25);
        assert_eq!(f32_at(&buf, 29), 1.);
        assert_eq!(f32_at(&buf, 30), 2.);
        assert_eq!(f32_at(&buf, 31), 3.);
        // bullets
        let mut word = META_WORDS;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 1), 1.);
        assert_eq!(f32_at(&buf, word + 5), 5.);
        assert_eq!(f32_at(&buf, word + 6), 6.);
        assert_eq!(f32_at(&buf, word + 7), 7.);
        assert_eq!(f32_at(&buf, word + 8), 18.); // flags low half: 0x0012
        assert_eq!(f32_at(&buf, word + 9), 1.); // flags high half: 0x0001
        // enemies
        word += 1 + 9;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 5), 14.);
        assert_eq!(f32_at(&buf, word + 6), 15.);
        assert_eq!(f32_at(&buf, word + 7), 16.);
        assert_eq!(f32_at(&buf, word + 8), 17.);
        assert_eq!(f32_at(&buf, word + 9), 0.5);
        assert_eq!(f32_at(&buf, word + 10), 60000.);
        assert_eq!(f32_at(&buf, word + 11), 1.);
        assert_eq!(f32_at(&buf, word + 12), 1.);
        assert_eq!(f32_at(&buf, word + 13), 42.);
        assert_eq!(f32_at(&buf, word + 14), 0.); // is_lethal
        assert_eq!(f32_at(&buf, word + 15), 9.); // no_hitbox_frames
        assert_eq!(f32_at(&buf, word + 16), 2.); // flags low half
        assert_eq!(f32_at(&buf, word + 17), 1024.); // flags high half: 0x0400
        // items
        word += 1 + 17;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 5), 8.);
        // segment lasers
        word += 1 + 5;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 6), 35.);
        // ray lasers
        word += 1 + 6;
        assert_eq!(u32_at(&buf, word), 0);
        // curve laser points
        word += 1;
        assert_eq!(u32_at(&buf, word), 2);
        assert_eq!(f32_at(&buf, word + 1), 40.);
        assert_eq!(f32_at(&buf, word + 5), 48.);
        assert_eq!(f32_at(&buf, word + 6), 44.);
        assert_eq!(f32_at(&buf, word + 10), 48.);
    }
}
