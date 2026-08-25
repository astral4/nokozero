//! Serialization of game state into observation payloads.
//!
//! A payload is a [`META_WORDS`]-word meta block followed by six entity sections. Each section has a `count` field of type `u32`
//! followed by `count` rows of the matching [`SECTION_WIDTHS`] entry. Every field is 4 bytes little-endian so the entire payload
//! can parse as an array with 4-byte elements. Enemy `max_hp` and item `kind` are `f32` so rows are homogenous and parsing is easier.
//! These values are still exact below 2^24.

use crate::practice::WireMeta;
use crate::reader::{GameState, Resources};

const META_WORDS: usize = 22;
/// Bullets; enemies; items; segment lasers; ray lasers; curve laser points.
const SECTION_WIDTHS: [usize; 6] = [5, 8, 5, 6, 8, 5];

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
        s.bullets.len() * SECTION_WIDTHS[0]
            + s.enemies.len() * SECTION_WIDTHS[1]
            + s.items.len() * SECTION_WIDTHS[2]
            + s.lasers.segments.len() * SECTION_WIDTHS[3]
            + s.lasers.rays.len() * SECTION_WIDTHS[4]
            + s.lasers.curve_points.len() * SECTION_WIDTHS[5]
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

    put_count(buf, state.bullets.len());
    for b in &state.bullets {
        put_f32s(buf, [b.pos_x, b.pos_y, b.vel_x, b.vel_y, b.hitbox_radius]);
    }

    put_count(buf, state.enemies.len());
    for e in &state.enemies {
        #[expect(clippy::cast_precision_loss)]
        put_f32s(
            buf,
            [
                e.pos_x,
                e.pos_y,
                e.vel_x,
                e.vel_y,
                e.hitbox_radius,
                e.hp_ratio,
                e.max_hp as f32,
                f32::from(u8::from(e.is_boss)),
            ],
        );
    }

    put_count(buf, state.items.len());
    for i in &state.items {
        #[expect(clippy::cast_precision_loss)]
        put_f32s(buf, [i.pos_x, i.pos_y, i.vel_x, i.vel_y, i.kind as f32]);
    }

    put_count(buf, state.lasers.segments.len());
    for l in &state.lasers.segments {
        put_f32s(
            buf,
            [
                l.head_pos_x,
                l.head_pos_y,
                l.vel_x,
                l.vel_y,
                l.length,
                l.width,
            ],
        );
    }

    put_count(buf, state.lasers.rays.len());
    for l in &state.lasers.rays {
        put_f32s(
            buf,
            [
                l.origin_pos_x,
                l.origin_pos_y,
                l.origin_vel_x,
                l.origin_vel_y,
                l.cos_angle,
                l.sin_angle,
                l.angular_vel,
                l.width,
            ],
        );
    }

    put_count(buf, state.lasers.curve_points.len());
    for p in &state.lasers.curve_points {
        put_f32s(buf, [p.pos_x, p.pos_y, p.vel_x, p.vel_y, p.width]);
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
    match state {
        Some(s) => put_f32s(
            buf,
            [
                s.player.pos_x,
                s.player.pos_y,
                f32::from(u8::from(s.player.is_focused)),
                s.player.hitbox_radius,
            ],
        ),
        None => put_f32s(buf, [0.; 4]),
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
        // player block
        for word in 18..22 {
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
                hitbox_radius: 5.,
            }],
            enemies: vec![Enemy {
                pos_x: 10.,
                pos_y: 11.,
                vel_x: 12.,
                vel_y: 13.,
                hitbox_radius: 14.,
                hp_ratio: 0.5,
                max_hp: 60000,
                is_boss: true,
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
        // player block
        assert_eq!(f32_at(&buf, 18), -100.5);
        assert_eq!(f32_at(&buf, 19), 200.25);
        assert_eq!(f32_at(&buf, 20), 1.);
        assert_eq!(f32_at(&buf, 21), 2.);
        // bullets
        let mut word = META_WORDS;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 1), 1.);
        assert_eq!(f32_at(&buf, word + 5), 5.);
        // enemies
        word += 1 + 5;
        assert_eq!(u32_at(&buf, word), 1);
        assert_eq!(f32_at(&buf, word + 6), 0.5);
        assert_eq!(f32_at(&buf, word + 7), 60000.);
        assert_eq!(f32_at(&buf, word + 8), 1.);
        // items
        word += 1 + 8;
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
