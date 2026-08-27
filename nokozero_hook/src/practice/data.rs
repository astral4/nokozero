//! ECL warp catalog data.

// Warp target IDs. Named sections are `stage * 1000 + category * 100`, where the category is 1 for midbosses and 2 for bosses.
// Chapter warps are `10000 + stage * 100 + portion`.
const ST1_MID1: u32 = 1101;
const ST1_MID2: u32 = 1102;
const ST1_MID3: u32 = 1103;
const ST1_BOSS1: u32 = 1201;
const ST1_BOSS2: u32 = 1202;
const ST1_BOSS3: u32 = 1203;
const ST1_BOSS4: u32 = 1204;
const ST2_MID1: u32 = 2101;
const ST2_BOSS1: u32 = 2201;
const ST2_BOSS2: u32 = 2202;
const ST2_BOSS3: u32 = 2203;
const ST2_BOSS4: u32 = 2204;
const ST2_BOSS5: u32 = 2205;
const ST2_BOSS6: u32 = 2206;
const ST3_MID1: u32 = 3101;
const ST3_MID2: u32 = 3102;
const ST3_BOSS1: u32 = 3201;
const ST3_BOSS2: u32 = 3202;
const ST3_BOSS3: u32 = 3203;
const ST3_BOSS4: u32 = 3204;
const ST3_BOSS5: u32 = 3205;
const ST3_BOSS6: u32 = 3206;
const ST3_BOSS7: u32 = 3207;
const ST4_MID1: u32 = 4101;
const ST4_BOSS1: u32 = 4201;
const ST4_BOSS2: u32 = 4202;
const ST4_BOSS3: u32 = 4203;
const ST4_BOSS4: u32 = 4204;
const ST4_BOSS5: u32 = 4205;
const ST4_BOSS6: u32 = 4206;
const ST4_BOSS7: u32 = 4207;
// `phase` selects the wave to start at (0 = default, 1-6 = later waves).
const ST5_MID1: u32 = 5101;
const ST5_BOSS1: u32 = 5201;
const ST5_BOSS2: u32 = 5202;
const ST5_BOSS3: u32 = 5203;
const ST5_BOSS4: u32 = 5204;
const ST5_BOSS5: u32 = 5205;
const ST5_BOSS6: u32 = 5206;
const ST5_BOSS7: u32 = 5207;
const ST5_BOSS8: u32 = 5208;
const ST6_STARS: u32 = 6101;
const ST6_BOSS1: u32 = 6201;
const ST6_BOSS2: u32 = 6202;
const ST6_BOSS3: u32 = 6203;
const ST6_BOSS4: u32 = 6204;
const ST6_BOSS5: u32 = 6205;
const ST6_BOSS6: u32 = 6206;
const ST6_BOSS7: u32 = 6207;
const ST6_BOSS8: u32 = 6208;
const ST6_BOSS9: u32 = 6209;
const ST6_BOSS10: u32 = 6210;
// `phase` 1-3 lowers the starting HP (5500/3500/1200).
const ST6_BOSS11: u32 = 6211;
const ST7_MID1: u32 = 7101;
const ST7_MID2: u32 = 7102;
const ST7_MID3: u32 = 7103;
const ST7_NS1: u32 = 7201;
const ST7_S1: u32 = 7202;
const ST7_NS2: u32 = 7203;
const ST7_S2: u32 = 7204;
const ST7_NS3: u32 = 7205;
const ST7_S3: u32 = 7206;
const ST7_NS4: u32 = 7207;
const ST7_S4: u32 = 7208;
const ST7_NS5: u32 = 7209;
const ST7_S5: u32 = 7210;
const ST7_NS6: u32 = 7211;
const ST7_S6: u32 = 7212;
const ST7_NS7: u32 = 7213;
const ST7_S7: u32 = 7214;
const ST7_NS8: u32 = 7215;
const ST7_S8: u32 = 7216;
const ST7_S9: u32 = 7217;
const ST7_S10: u32 = 7218;

// Offsets for the extra stage.
const ST7_START: u32 = 0x8f6c;
const ST7_BOSS_CREATE_CALL: u32 = 0x93cc;
const ST7_BS_MOVE_POS_X: u32 = 0x4a8;
const ST7_BS_MOVE_POS_Y: u32 = 0x4ac;
const ST7_BS_ANM_SELECT: u32 = 0x57c;
const ST7_BS_ANM_SET_MAIN_INS: u32 = 0x584;
const ST7_BS_ANM_SET_MAIN_ARG2: u32 = 0x594;
const ST7_BS_SET_SPRITE_INS: u32 = 0x5d8;
const ST7_BS_MOVE_LIMIT_INS: u32 = 0x644;
const ST7_BS_ANM_SET_SPRITE1_ARG2: u32 = 0x688;
const ST7_BS_ANM_SET_SPRITE2_ARG2: u32 = 0x6a0;
const ST7_BS_ANM_SELECT2: u32 = 0x6b4;

// Offsets for stage scripts. `*_START` is the stage script's entry-jump site.
// `*_SPELL_START` / `*_SPELL` are each stage's spell practice jump pair.
const ST1_START: u32 = 0x7220;
const ST1_SPELL_START: u32 = 0x3b8;
const ST1_SPELL: u32 = 0x4a0;
const ST2_START: u32 = 0x7444;
const ST2_SPELL_START: u32 = 0x40c;
const ST2_SPELL: u32 = 0x4f4;
const ST3_START: u32 = 0xa258;
const ST3_SPELL_START: u32 = 0x448;
const ST3_SPELL: u32 = 0x530;
const ST4_START: u32 = 0x860c;
const ST4_SPELL_START: u32 = 0x41e0;
const ST4_SPELL: u32 = 0x42c8;
const ST5_START: u32 = 0x9f0c;
const ST5_SPELL_START: u32 = 0x528;
const ST5_SPELL: u32 = 0x610;
const ST6_START: u32 = 0x91d0;
const ST6_SPELL_START: u32 = 0x5a0;
const ST6_SPELL: u32 = 0x688;
// The "boss change effect" block used by some stage 6 sections.
const ST6_EFFECT_BLOCK: u32 = 0x63c;

/// `(entry jump start, boss section destination, boss ECL file)` for stages 1–6.
const BOSS_ENTRIES: [(u32, u32, usize); 6] = [
    (ST1_START, 0x768c, 2),
    (ST2_START, 0x7834, 2),
    (ST3_START, 0xa614, 2),
    (ST4_START, 0x89b0, 2),
    (ST5_START, 0xa220, 3),
    (ST6_START, 0x9500, 2),
];

/// `ST5_MID1` wave starts for `phase` 1–6. `0` runs the section's default wave.
const ST5_WAVE_DESTS: [u32; 6] = [0x7100, 0x716c, 0x71ec, 0x726c, 0x72ec, 0x736c];

/// The wave selector jump patched by `ST5_MID1`'s waves and chapter warps (5,6)–(5,10).
const ST5_WAVE_SRC: u32 = 0x70ec;

/// `ST6_BOSS11` starting HP tiers for `phase` 1-3. `0` keeps the full 8000.
const ST6_BOSS11_HEALTH: [i32; 3] = [5500, 3500, 1200];

const ST7_S10_HECATIA_AT: [u32; 3] = [0x8f30, 0x8f7c, 0x8fc8];
const ST7_S10_HECATIA_AT_FRAME: i32 = 94;
const ST7_S10_JUNKO_AT: [u32; 3] = [0x0b08, 0x0b54, 0x0ba0];
const ST7_S10_JUNKO_AT_FRAME: i32 = 0;

const _: () = assert!(
    ST7_S10_HECATIA_AT.len() == ST7_S10_JUNKO_AT.len(),
    "both bosses must expose the same number of ST7_S10 point indices",
);

#[derive(Clone, Copy)]
enum Phase {
    /// For sections without phases. The wire value must be `0`.
    None,
    /// `ST5_MID1` with the wave to start at. `0` = the default wave.
    Wave(u32),
    /// `ST6_BOSS11` with the starting HP tier. `0` = full HP.
    Health(u32),
    /// `ST7_S9` phase. `0` = blue phase, 1 = green phase, 2 = red phase.
    S9(u32),
    /// `ST7_S10` with each boss's starting point index.
    S10 { hecatia: usize, junko: usize },
}

impl Phase {
    /// Returns `None` if `phase` is not a valid value for the provided section.
    const fn parse(section: u32, phase: u32) -> Option<Self> {
        let resolved = match section {
            ST5_MID1 if phase as usize <= ST5_WAVE_DESTS.len() => Self::Wave(phase),
            ST6_BOSS11 if phase as usize <= ST6_BOSS11_HEALTH.len() => Self::Health(phase),
            ST7_S9 if phase <= 2 => Self::S9(phase),
            ST7_S10 if phase == 0 => Self::S10 {
                hecatia: 0,
                junko: 0,
            },
            ST7_S10 => {
                let (Some(hecatia), Some(junko)) =
                    (s10_start_index(phase / 10), s10_start_index(phase % 10))
                else {
                    return None;
                };
                Self::S10 { hecatia, junko }
            }
            _ if phase == 0 => Self::None,
            _ => return None,
        };
        Some(resolved)
    }
}

const fn s10_start_index(digit: u32) -> Option<usize> {
    if let Some(index) = (digit as usize).checked_sub(1)
        && index < ST7_S10_HECATIA_AT.len()
    {
        Some(index)
    } else {
        None
    }
}

enum SectionId {
    Chapter { stage: u32, portion: u32 },
    Named { stage: u32 },
}

const STAGES: u32 = 7;
pub(super) const EXTRA_STAGE: u32 = 7;
pub(super) const EXTRA_DIFFICULTY: u32 = 4;
/// The highest selectable character index.
pub(crate) const MAX_CHARACTER: u32 = 3;

impl SectionId {
    const fn classify(section: u32) -> Self {
        if section >= 10000 && section < 20000 {
            Self::Chapter {
                stage: (section - 10000) / 100,
                portion: (section - 10000) % 100,
            }
        } else {
            Self::Named {
                stage: section / 1000,
            }
        }
    }

    const fn stage(self) -> Option<u32> {
        let stage = match self {
            Self::Chapter { stage, .. } | Self::Named { stage } => stage,
        };
        if stage >= 1 && stage <= STAGES {
            Some(stage)
        } else {
            None
        }
    }
}

pub(super) const fn section_stage(section: u32) -> Option<u32> {
    SectionId::classify(section).stage()
}

/// Returns whether the provided `section` and `phase` correspond to a dispatchable warp.
pub(super) fn section_mapped(section: u32, phase: u32) -> bool {
    expand_section(section, phase, &mut |_| {})
}

/// Emits the full op stream for `(section, phase)` into `emit`. Returns whether the pair mapped. `false` means nothing was emitted.
pub(super) fn expand_section(section: u32, phase: u32, emit: &mut Emit<'_>) -> bool {
    let Some(phase) = Phase::parse(section, phase) else {
        return false;
    };
    match SectionId::classify(section) {
        SectionId::Chapter { stage, portion } => expand_chapter(stage, portion, emit),
        SectionId::Named { .. } => match warp_index(section) {
            Some(index) => {
                WARPS[index].expand(phase, emit);
                true
            }
            None => false,
        },
    }
}

/// Returns whether `stage` can run at `difficulty`.
pub(super) const fn rank_matches_stage(stage: u32, difficulty: u32) -> bool {
    (stage == EXTRA_STAGE) == (difficulty == EXTRA_DIFFICULTY)
}

#[derive(Clone, Copy)]
struct Nonspell {
    /// Offset of the ASCII digit inside the nonspell-name string, plus the digit to write.
    select: (u32, i8),
    /// The instruction opcode to zero to stop the nonspell's item drops.
    item_drops: u32,
    /// The instruction opcode to zero to silence the boss change sound effect.
    sound_effect: u32,
    /// Per-boss tweaks for move/wait/invulnerability windows that skip the boss' approach animation.
    timings: &'static [(u32, i32)],
}

/// A stored patch operation.
#[derive(Clone, Copy)]
enum Op {
    /// Selects ECL file slot `n`, resetting the cursor.
    File(usize),
    /// Moves the ECL cursor. Pairs with a following [`Op::Seq`].
    Pos(u32),
    /// Performs an ECL jump, verifying the ID is `expect` at `start`, then overwrites with a 24-byte jump.
    Jump {
        start: u32,
        expect: u16,
        dest: u32,
        at_frame: i32,
        ecl_time: i32,
    },
    /// Verifies the ID is `expect` at `pos`, then writes `words` from there.
    SeqAt {
        pos: u32,
        expect: u16,
        words: &'static [u32],
    },
    /// Writes `words` at the cursor, continuing a [`Op::Pos`]/[`Op::SeqAt`] stream.
    Seq {
        words: &'static [u32],
    },
    I16 {
        at: u32,
        v: i16,
    },
    I32 {
        at: u32,
        v: i32,
    },
    U32 {
        at: u32,
        v: u32,
    },
    /// Sets `ChapterIntent::skip_remaining`.
    Skip(i32),
    /// Sets `ChapterIntent::st7_bonus`.
    St7Bonus,
    /// Jumps to the boss section of the specified stage script, selects the boss ECL file, and skips the chapters left behind.
    /// Leaves the boss file selected.
    BossEntry {
        stage: usize,
        skips: i32,
    },
    /// Jumps to a spell where the jump target is the spell base. If it isn't, [`Op::SpellFields`] should be used.
    EnterSpell {
        start: u32,
        dest: u32,
        at_frame: i32,
        health: i32,
        ordinal: i8,
    },
    SpellFields {
        base: u32,
        health: i32,
        ordinal: i8,
    },
    Nonspell {
        stage: usize,
        ns: Nonspell,
    },
    St6BossEffect {
        pos_str: u32,
    },
    St7BossEntry,
    St7HideSubboss,
    St7EnterSpell {
        health: u32,
        ordinal: u32,
        junko: bool,
    },
    St7Mid {
        health: i32,
        ordinal: i8,
    },
}

type Emit<'e> = dyn FnMut(PrimOp<'_>) + 'e;

/// A primitive patch operation emitted by [`Op::expand`].
#[derive(Clone, Copy)]
pub(super) enum PrimOp<'a> {
    File(usize),
    Pos(u32),
    Jump {
        start: u32,
        expect: u16,
        dest: u32,
        at_frame: i32,
        ecl_time: i32,
    },
    SeqAt {
        pos: u32,
        expect: u16,
        words: &'a [u32],
    },
    Seq {
        words: &'a [u32],
    },
    I8 {
        at: u32,
        v: i8,
    },
    I16 {
        at: u32,
        v: i16,
    },
    I32 {
        at: u32,
        v: i32,
    },
    U32 {
        at: u32,
        v: u32,
    },
    Skip(i32),
    SetChapter(i32),
    St7Bonus,
}

impl Op {
    /// Flattens a stored op to primitive ops.
    #[expect(clippy::too_many_lines)]
    fn expand(self, emit: &mut Emit<'_>) {
        match self {
            Op::File(n) => emit(PrimOp::File(n)),
            Op::Pos(p) => emit(PrimOp::Pos(p)),
            Op::Jump {
                start,
                expect,
                dest,
                at_frame,
                ecl_time,
            } => {
                emit(PrimOp::Jump {
                    start,
                    expect,
                    dest,
                    at_frame,
                    ecl_time,
                });
            }
            Op::SeqAt { pos, expect, words } => emit(PrimOp::SeqAt { pos, expect, words }),
            Op::Seq { words } => emit(PrimOp::Seq { words }),
            Op::I16 { at, v } => emit(PrimOp::I16 { at, v }),
            Op::I32 { at, v } => emit(PrimOp::I32 { at, v }),
            Op::U32 { at, v } => emit(PrimOp::U32 { at, v }),
            Op::Skip(n) => emit(PrimOp::Skip(n)),
            Op::St7Bonus => emit(PrimOp::St7Bonus),
            Op::BossEntry { stage, skips } => {
                let (start, dest, file) = BOSS_ENTRIES[stage - 1];
                // Every `ST*_START` holds an id-0x17 instruction.
                emit(PrimOp::Jump {
                    start,
                    expect: 0x17,
                    dest,
                    at_frame: 60,
                    ecl_time: 0,
                });
                emit(PrimOp::File(file));
                emit(PrimOp::Skip(skips));
            }
            Op::EnterSpell {
                start,
                dest,
                at_frame,
                health,
                ordinal,
            } => {
                // Every `*_SPELL_START` holds an id-0x2a instruction.
                emit(PrimOp::Jump {
                    start,
                    expect: 0x2A,
                    dest,
                    at_frame,
                    ecl_time: 0,
                });
                Op::SpellFields {
                    base: dest,
                    health,
                    ordinal,
                }
                .expand(emit);
            }
            Op::SpellFields {
                base,
                health,
                ordinal,
            } => {
                emit(PrimOp::I32 {
                    at: base + 0x10,
                    v: health,
                });
                emit(PrimOp::I8 {
                    at: base + 0x30,
                    v: ordinal,
                });
            }
            Op::Nonspell { stage, ns } => {
                Op::BossEntry { stage, skips: 2 }.expand(emit);
                emit(PrimOp::I8 {
                    at: ns.select.0,
                    v: ns.select.1,
                });
                emit(PrimOp::I16 {
                    at: ns.item_drops,
                    v: 0,
                });
                emit(PrimOp::I16 {
                    at: ns.sound_effect,
                    v: 0,
                });
                for &(at, v) in ns.timings {
                    emit(PrimOp::I32 { at, v });
                }
            }
            Op::St6BossEffect { pos_str } => {
                emit(PrimOp::SeqAt {
                    pos: ST6_EFFECT_BLOCK,
                    expect: 0x2A,
                    words: &[
                        0,
                        0x0014_012E,
                        0x01FF_0000,
                        0,
                        3,
                        0,
                        0x0018_012F,
                        0x02FF_0000,
                        0,
                        3,
                        6,
                    ],
                });
                let tail = [
                    0,
                    0x0020_000F,
                    0x01FF_0000,
                    0,
                    0xC,
                    0x7373_6F42,
                    pos_str,
                    0x0000_0073,
                ];
                emit(PrimOp::Seq { words: &tail });
            }
            Op::St7BossEntry => {
                emit(PrimOp::Jump {
                    start: ST7_START,
                    expect: 0x17,
                    dest: ST7_BOSS_CREATE_CALL,
                    at_frame: 60,
                    ecl_time: 0,
                });
                emit(PrimOp::Skip(2));
                Op::St7HideSubboss.expand(emit);
            }
            Op::St7HideSubboss => {
                for &inner in ST7_HIDE_SUBBOSS {
                    inner.expand(emit);
                }
            }
            Op::St7EnterSpell {
                health,
                ordinal,
                junko,
            } => {
                emit(PrimOp::File(3));
                let head = [0, 0x0014_01ff, 0x01ff_0000, 0, health];
                emit(PrimOp::SeqAt {
                    pos: 0x6d0,
                    expect: 0x16,
                    words: &head,
                });
                if junko {
                    emit(PrimOp::Seq {
                        words: &[
                            0,
                            0x0014_012E,
                            0x01FF_0000,
                            0,
                            5,
                            0,
                            0x0018_012F,
                            0x02FF_0000,
                            0,
                            3,
                            6,
                        ],
                    });
                    emit(PrimOp::Seq {
                        words: &[
                            0,
                            0x0020_000F,
                            0x01FF_0000,
                            0,
                            0xC,
                            0x7373_6F42,
                            0x6F70_5F34,
                            0x0000_0073,
                        ],
                    });
                }
                let card = [
                    0,
                    0x0020_000b,
                    0x01ff_0000,
                    0,
                    0xc,
                    0x7373_6f42,
                    0x6472_6143,
                    ordinal,
                ];
                emit(PrimOp::Seq { words: &card });
            }
            Op::St7Mid { health, ordinal } => {
                emit(PrimOp::File(2));
                // invulnerability window
                emit(PrimOp::SeqAt {
                    pos: 0x34c,
                    expect: 0x1F8,
                    words: &[0, 0x0020_0203, 0x01FF_0000, 0, 60],
                });
                // boss movement restriction
                emit(PrimOp::SeqAt {
                    pos: 0x37c,
                    expect: 0x202,
                    words: &[
                        0,
                        0x002C_01F8,
                        0x04FF_0000,
                        0,
                        0,
                        0x4300_0000,
                        0x438C_0000,
                        0x4380_0000,
                    ],
                });
                emit(PrimOp::I16 { at: 0x33c, v: 0 }); // void wait
                emit(PrimOp::I32 { at: 0x36c, v: 60 }); // wait time
                emit(PrimOp::I32 { at: 0x328, v: 60 }); // move time
                // spell practice jump
                emit(PrimOp::Jump {
                    start: 0x48c,
                    expect: 0x203,
                    dest: 0x508,
                    at_frame: 0,
                    ecl_time: 0,
                });
                emit(PrimOp::I32 {
                    at: 0x2fc,
                    v: health,
                });
                emit(PrimOp::I8 {
                    at: 0x525,
                    v: ordinal,
                });
            }
        }
    }
}

struct Warp {
    section: u32,
    /// Operations applied for every phase.
    ops: &'static [Op],
    /// Operations applied depending on the phase.
    phase_ops: Option<fn(Phase, &mut Emit<'_>)>,
}

impl Warp {
    const fn new(section: u32, ops: &'static [Op]) -> Self {
        Self {
            section,
            ops,
            phase_ops: None,
        }
    }

    const fn with_phase(
        section: u32,
        ops: &'static [Op],
        phase_ops: fn(Phase, &mut Emit<'_>),
    ) -> Self {
        Self {
            section,
            ops,
            phase_ops: Some(phase_ops),
        }
    }

    /// Emits the full stream of primitive ops for a named section.
    fn expand(&self, phase: Phase, emit: &mut Emit<'_>) {
        for &op in self.ops {
            op.expand(emit);
        }
        if let Some(phase_ops) = self.phase_ops {
            phase_ops(phase, emit);
        }
    }
}

const fn warp_index(section: u32) -> Option<usize> {
    let mut i = 0;
    while i < WARPS.len() {
        if WARPS[i].section == section {
            return Some(i);
        }
        i += 1;
    }
    None
}

const WARPS: &[Warp] = &[
    Warp::new(
        ST1_MID1,
        &[
            Op::Jump {
                start: ST1_START,
                expect: 0x17,
                dest: 0x74d8,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(
        ST1_MID2,
        &[
            Op::Jump {
                start: ST1_START,
                expect: 0x17,
                dest: 0x758c,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(
        ST1_MID3,
        &[
            Op::Jump {
                start: ST1_START,
                expect: 0x17,
                dest: 0x75b4,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::File(4),
            Op::Skip(1),
            // spell practice jump
            Op::Jump {
                start: 0x444,
                expect: 0x17,
                dest: 0x560,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::I32 { at: 0x434, v: 60 },
            Op::I32 { at: 0x570, v: 60 },   // move speed
            Op::I32 { at: 0x588, v: 9999 }, // suppress the spellcard
            // invulnerability window
            Op::SeqAt {
                pos: 0x384,
                expect: 0x20F,
                words: &[0, 0x001C_0203, 0x01FF_0000, 0, 60],
            },
        ],
    ),
    Warp::new(ST1_BOSS1, &[Op::BossEntry { stage: 1, skips: 1 }]),
    Warp::new(
        ST1_BOSS2,
        &[
            Op::BossEntry { stage: 1, skips: 2 },
            Op::EnterSpell {
                start: ST1_SPELL_START,
                dest: ST1_SPELL,
                at_frame: 1,
                health: 1700,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST1_BOSS3,
        &[Op::Nonspell {
            stage: 1,
            ns: Nonspell {
                select: (0x5d0, 0x32),
                item_drops: 0x13ec,
                sound_effect: 0x151c,
                timings: &[],
            },
        }],
    ),
    Warp::new(
        ST1_BOSS4,
        &[
            Op::BossEntry { stage: 1, skips: 2 },
            Op::EnterSpell {
                start: ST1_SPELL_START,
                dest: ST1_SPELL,
                at_frame: 1,
                health: 1900,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST2_MID1,
        &[
            Op::Jump {
                start: ST2_START,
                expect: 0x17,
                dest: 0x7720,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(ST2_BOSS1, &[Op::BossEntry { stage: 2, skips: 1 }]),
    Warp::new(
        ST2_BOSS2,
        &[
            Op::BossEntry { stage: 2, skips: 2 },
            Op::EnterSpell {
                start: ST2_SPELL_START,
                dest: ST2_SPELL,
                at_frame: 1,
                health: 2200,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST2_BOSS3,
        &[Op::Nonspell {
            stage: 2,
            ns: Nonspell {
                select: (0x644, 0x32),
                item_drops: 0x1214,
                sound_effect: 0x1344,
                timings: &[(0x14bc, 59), (0x1508, 0), (0x1120, 60)],
            },
        }],
    ),
    Warp::new(
        ST2_BOSS4,
        &[
            Op::BossEntry { stage: 2, skips: 2 },
            Op::EnterSpell {
                start: ST2_SPELL_START,
                dest: ST2_SPELL,
                at_frame: 1,
                health: 2400,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST2_BOSS5,
        &[Op::Nonspell {
            stage: 2,
            ns: Nonspell {
                select: (0x644, 0x33),
                item_drops: 0x1ea8,
                sound_effect: 0x1fd8,
                timings: &[(0x2150, 59), (0x219c, 0), (0x1db4, 60)],
            },
        }],
    ),
    Warp::new(
        ST2_BOSS6,
        &[
            Op::BossEntry { stage: 2, skips: 2 },
            Op::EnterSpell {
                start: ST2_SPELL_START,
                dest: ST2_SPELL,
                at_frame: 1,
                health: 2500,
                ordinal: 0x33,
            },
        ],
    ),
    Warp::new(
        ST3_MID1,
        &[
            Op::Jump {
                start: ST3_START,
                expect: 0x17,
                dest: 0xa500,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(
        ST3_MID2,
        &[
            Op::Jump {
                start: ST3_START,
                expect: 0x17,
                dest: 0xa528,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::File(3),
            Op::Skip(1),
            // spell practice jump
            Op::Jump {
                start: 0x360,
                expect: 0x17,
                dest: 0x47c,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::I32 { at: 0x350, v: 60 }, // move speed
            Op::I32 { at: 0x48c, v: 60 },
            Op::I32 { at: 0x4a4, v: 9999 }, // suppress the spellcard
            // invulnerability window
            Op::SeqAt {
                pos: 0x2a4,
                expect: 0x20F,
                words: &[0, 0x001C_0203, 0x01FF_0000, 0, 60],
            },
        ],
    ),
    Warp::new(ST3_BOSS1, &[Op::BossEntry { stage: 3, skips: 1 }]),
    Warp::new(
        ST3_BOSS2,
        &[
            Op::BossEntry { stage: 3, skips: 2 },
            Op::EnterSpell {
                start: ST3_SPELL_START,
                dest: ST3_SPELL,
                at_frame: 1,
                health: 2000,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST3_BOSS3,
        &[Op::Nonspell {
            stage: 3,
            ns: Nonspell {
                select: (0x680, 0x32),
                item_drops: 0x15d8,
                sound_effect: 0x1708,
                timings: &[(0x18ac, 0), (0x14e4, 60)],
            },
        }],
    ),
    Warp::new(
        ST3_BOSS4,
        &[
            Op::BossEntry { stage: 3, skips: 2 },
            Op::EnterSpell {
                start: ST3_SPELL_START,
                dest: ST3_SPELL,
                at_frame: 1,
                health: 2100,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST3_BOSS5,
        &[Op::Nonspell {
            stage: 3,
            ns: Nonspell {
                select: (0x680, 0x33),
                item_drops: 0x27b0,
                sound_effect: 0x28e0,
                timings: &[(0x2a98, 0), (0x2a58, 60)],
            },
        }],
    ),
    Warp::new(
        ST3_BOSS6,
        &[
            Op::BossEntry { stage: 3, skips: 2 },
            Op::EnterSpell {
                start: ST3_SPELL_START,
                dest: ST3_SPELL,
                at_frame: 1,
                health: 2500,
                ordinal: 0x33,
            },
        ],
    ),
    Warp::new(
        ST3_BOSS7,
        &[
            Op::BossEntry { stage: 3, skips: 2 },
            Op::EnterSpell {
                start: ST3_SPELL_START,
                dest: ST3_SPELL,
                at_frame: 1,
                health: 3000,
                ordinal: 0x34,
            },
        ],
    ),
    Warp::new(
        ST4_MID1,
        &[
            Op::Jump {
                start: ST4_START,
                expect: 0x17,
                dest: 0x88ec,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(ST4_BOSS1, &[Op::BossEntry { stage: 4, skips: 1 }]),
    Warp::new(
        ST4_BOSS2,
        &[
            Op::BossEntry { stage: 4, skips: 2 },
            Op::EnterSpell {
                start: ST4_SPELL_START,
                dest: ST4_SPELL,
                at_frame: 1,
                health: 2300,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST4_BOSS3,
        &[Op::Nonspell {
            stage: 4,
            ns: Nonspell {
                select: (0x4418, 0x32),
                item_drops: 0x4f24,
                sound_effect: 0x5054,
                timings: &[(0x51cc, 20), (0x4e30, 0)],
            },
        }],
    ),
    Warp::new(
        ST4_BOSS4,
        &[
            Op::BossEntry { stage: 4, skips: 2 },
            Op::EnterSpell {
                start: ST4_SPELL_START,
                dest: ST4_SPELL,
                at_frame: 1,
                health: 3100,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST4_BOSS5,
        &[Op::Nonspell {
            stage: 4,
            ns: Nonspell {
                select: (0x4418, 0x33),
                item_drops: 0x5d40,
                sound_effect: 0x5e70,
                timings: &[(0x5ffc, 20), (0x5fe8, 0)],
            },
        }],
    ),
    Warp::new(
        ST4_BOSS6,
        &[
            Op::BossEntry { stage: 4, skips: 2 },
            Op::EnterSpell {
                start: ST4_SPELL_START,
                dest: ST4_SPELL,
                at_frame: 1,
                health: 2500,
                ordinal: 0x33,
            },
        ],
    ),
    Warp::new(
        ST4_BOSS7,
        &[
            Op::BossEntry { stage: 4, skips: 2 },
            Op::Jump {
                start: ST4_SPELL_START,
                expect: 0x2A,
                dest: 0x42b4,
                at_frame: 1,
                ecl_time: 0,
            },
            Op::Pos(0x42b4),
            Op::Seq {
                words: &[1, 0x0014_0279, 0x01FF_0000, 0, 1],
            },
            Op::SpellFields {
                base: ST4_SPELL,
                health: 3400,
                ordinal: 0x34,
            },
        ],
    ),
    Warp::with_phase(ST5_MID1, &[Op::Skip(1)], st5_mid1_ops),
    Warp::new(ST5_BOSS1, &[Op::BossEntry { stage: 5, skips: 1 }]),
    Warp::new(
        ST5_BOSS2,
        &[
            Op::BossEntry { stage: 5, skips: 2 },
            Op::EnterSpell {
                start: ST5_SPELL_START,
                dest: ST5_SPELL,
                at_frame: 0,
                health: 1,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST5_BOSS3,
        &[Op::Nonspell {
            stage: 5,
            ns: Nonspell {
                select: (0x760, 0x32),
                item_drops: 0x1394,
                sound_effect: 0x14c4,
                timings: &[(0x163c, 59), (0x165c, 0), (0x12ac, 0)],
            },
        }],
    ),
    Warp::new(
        ST5_BOSS4,
        &[
            Op::BossEntry { stage: 5, skips: 2 },
            Op::EnterSpell {
                start: ST5_SPELL_START,
                dest: ST5_SPELL,
                at_frame: 0,
                health: 3400,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST5_BOSS5,
        &[Op::Nonspell {
            stage: 5,
            ns: Nonspell {
                select: (0x760, 0x33),
                item_drops: 0x2744,
                sound_effect: 0x2874,
                timings: &[(0x29ec, 59), (0x2a0c, 0)],
            },
        }],
    ),
    Warp::new(
        ST5_BOSS6,
        &[
            Op::BossEntry { stage: 5, skips: 2 },
            Op::EnterSpell {
                start: ST5_SPELL_START,
                dest: ST5_SPELL,
                at_frame: 0,
                health: 2320,
                ordinal: 0x33,
            },
        ],
    ),
    Warp::new(
        ST5_BOSS7,
        &[
            Op::BossEntry { stage: 5, skips: 2 },
            Op::EnterSpell {
                start: ST5_SPELL_START,
                dest: ST5_SPELL,
                at_frame: 0,
                health: 3001,
                ordinal: 0x34,
            },
        ],
    ),
    Warp::new(
        ST5_BOSS8,
        &[
            Op::BossEntry { stage: 5, skips: 2 },
            Op::EnterSpell {
                start: ST5_SPELL_START,
                dest: ST5_SPELL,
                at_frame: 0,
                health: 1,
                ordinal: 0x35,
            },
        ],
    ),
    Warp::new(
        ST6_STARS,
        &[
            Op::Jump {
                start: ST6_START,
                expect: 0x17,
                dest: 0x93e0,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::Jump {
                start: 0x2b34,
                expect: 0xB,
                dest: 0x2c40,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
        ],
    ),
    Warp::new(ST6_BOSS1, &[Op::BossEntry { stage: 6, skips: 1 }]),
    Warp::new(
        ST6_BOSS2,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::EnterSpell {
                start: ST6_SPELL_START,
                dest: ST6_SPELL,
                at_frame: 0,
                health: 4000,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST6_BOSS3,
        &[Op::Nonspell {
            stage: 6,
            ns: Nonspell {
                select: (0x7d8, 0x32),
                item_drops: 0x1274,
                sound_effect: 0x13a4,
                timings: &[(0x1528, 0), (0x1178, 0)],
            },
        }],
    ),
    Warp::new(
        ST6_BOSS4,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::EnterSpell {
                start: ST6_SPELL_START,
                dest: ST6_SPELL,
                at_frame: 0,
                health: 2600,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST6_BOSS5,
        &[Op::Nonspell {
            stage: 6,
            ns: Nonspell {
                select: (0x7d8, 0x33),
                item_drops: 0x1f84,
                sound_effect: 0x20b4,
                timings: &[],
            },
        }],
    ),
    Warp::new(
        ST6_BOSS6,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::Jump {
                start: ST6_SPELL_START,
                expect: 0x2A,
                dest: ST6_EFFECT_BLOCK,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::SpellFields {
                base: ST6_SPELL,
                health: 4000,
                ordinal: 0x33,
            },
            Op::St6BossEffect {
                pos_str: 0x6f70_5f33,
            },
        ],
    ),
    Warp::new(
        ST6_BOSS7,
        &[Op::Nonspell {
            stage: 6,
            ns: Nonspell {
                select: (0x7d8, 0x34),
                item_drops: 0x302c,
                sound_effect: 0x315c,
                timings: &[],
            },
        }],
    ),
    Warp::new(
        ST6_BOSS8,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::Jump {
                start: ST6_SPELL_START,
                expect: 0x2A,
                dest: ST6_EFFECT_BLOCK,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::SpellFields {
                base: ST6_SPELL,
                health: 3700,
                ordinal: 0x34,
            },
            Op::St6BossEffect {
                pos_str: 0x6f70_5f34,
            },
        ],
    ),
    Warp::new(
        ST6_BOSS9,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::EnterSpell {
                start: ST6_SPELL_START,
                dest: ST6_SPELL,
                at_frame: 0,
                health: 5000,
                ordinal: 0x35,
            },
        ],
    ),
    Warp::new(
        ST6_BOSS10,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::EnterSpell {
                start: ST6_SPELL_START,
                dest: ST6_SPELL,
                at_frame: 0,
                health: 6000,
                ordinal: 0x36,
            },
        ],
    ),
    Warp::with_phase(
        ST6_BOSS11,
        &[
            Op::BossEntry { stage: 6, skips: 2 },
            Op::Jump {
                start: ST6_SPELL_START,
                expect: 0x2A,
                dest: ST6_EFFECT_BLOCK,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::SpellFields {
                base: ST6_SPELL,
                health: 8000,
                ordinal: 0x37,
            },
            Op::St6BossEffect {
                pos_str: 0x6f70_5f34,
            },
        ],
        st6_boss11_ops,
    ),
    Warp::new(
        ST7_MID1,
        &[
            Op::Jump {
                start: ST7_START,
                expect: 0x17,
                dest: 0x92a8,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
            Op::St7Mid {
                health: 2200,
                ordinal: 0x31,
            },
        ],
    ),
    Warp::new(
        ST7_MID2,
        &[
            Op::Jump {
                start: ST7_START,
                expect: 0x17,
                dest: 0x92a8,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
            Op::St7Mid {
                health: 2200,
                ordinal: 0x32,
            },
        ],
    ),
    Warp::new(
        ST7_MID3,
        &[
            Op::Jump {
                start: ST7_START,
                expect: 0x17,
                dest: 0x92a8,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(1),
            Op::St7Mid {
                health: 3000,
                ordinal: 0x33,
            },
        ],
    ),
    Warp::new(
        ST7_NS1,
        &[
            Op::St7Bonus,
            Op::Jump {
                start: ST7_START,
                expect: 0x17,
                dest: ST7_BOSS_CREATE_CALL,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::St7HideSubboss,
            Op::Skip(1),
        ],
    ),
    Warp::new(
        ST7_S1,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3000,
                ordinal: 0x31,
                junko: false,
            },
        ],
    ),
    Warp::new(
        ST7_NS2,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x32 }, // change nonspell
            Op::I16 { at: 0x16f4, v: 0 },
            Op::I16 { at: 0x1824, v: 0 },
            Op::I32 { at: 0x199c, v: 59 },
            Op::I32 { at: 0x19bc, v: 0 },
            Op::I32 { at: 0x15dc, v: 0 },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            }, // void 303-0
            Op::I16 { at: 0x191c, v: 0 }, // void 306-0
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 14,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 120,
            },
            // jump over sprite change
            Op::Jump {
                start: 0x1a90,
                expect: 0x204,
                dest: 0x1ae8,
                at_frame: 24,
                ecl_time: 0,
            },
        ],
    ),
    Warp::new(
        ST7_S2,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3400,
                ordinal: 0x32,
                junko: false,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 14,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 120,
            },
        ],
    ),
    Warp::new(
        ST7_NS3,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x33 },
            Op::I16 { at: 0x2430, v: 0 },
            Op::I16 { at: 0x2560, v: 0 },
            Op::I32 { at: 0x26d8, v: 59 },
            Op::I32 { at: 0x26f8, v: 0 },
            Op::I32 { at: 0x2318, v: 0 },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I16 { at: 0x2658, v: 0 },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 7,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 132,
            },
            Op::Jump {
                start: 0x276c,
                expect: 0x204,
                dest: 0x2798,
                at_frame: 24,
                ecl_time: 0,
            },
        ],
    ),
    Warp::new(
        ST7_S3,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3000,
                ordinal: 0x33,
                junko: false,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 7,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 132,
            },
        ],
    ),
    Warp::new(
        ST7_NS4,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x34 },
            Op::I16 { at: 0x32a0, v: 0 }, // don't activate sub-boss
            Op::I16 { at: 0x32e8, v: 0 },
            Op::I16 { at: 0x3418, v: 0 },
            // skip move
            Op::Jump {
                start: 0x3548,
                expect: 0x191,
                dest: 0x35b0,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::I32 { at: 0x31c4, v: 10 }, // invulnerability time
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I16 { at: 0x3510, v: 0 },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 116,
            },
            Op::I32 {
                at: ST7_BS_ANM_SELECT,
                v: 5,
            }, // change 302
        ],
    ),
    Warp::new(
        ST7_S4,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 2000,
                ordinal: 0x34,
                junko: true,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 116,
            },
            Op::I32 {
                at: ST7_BS_ANM_SELECT,
                v: 5,
            },
        ],
    ),
    Warp::new(
        ST7_NS5,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x35 },
            Op::I16 { at: 0x4464, v: 0 },
            Op::I16 { at: 0x44c0, v: 0 },
            Op::I16 { at: 0x45f0, v: 0 },
            Op::Jump {
                start: 0x4720,
                expect: 0x191,
                dest: 0x4788,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::I32 { at: 0x4388, v: 0 },
            Op::I16 { at: 0x46e8, v: 0 }, // void 306-0
        ],
    ),
    Warp::new(
        ST7_S5,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3500,
                ordinal: 0x35,
                junko: false,
            },
        ],
    ),
    Warp::new(
        ST7_NS6,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x36 },
            Op::I16 { at: 0x5368, v: 0 },
            Op::I16 { at: 0x5498, v: 0 },
            Op::I32 { at: 0x5610, v: 59 },
            Op::I32 { at: 0x5630, v: 0 },
            Op::I32 { at: 0x5250, v: 0 },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I16 { at: 0x5590, v: 0 },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 14,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 120,
            },
            Op::Jump {
                start: 0x5694,
                expect: 0x204,
                dest: 0x56ec,
                at_frame: 24,
                ecl_time: 0,
            },
        ],
    ),
    Warp::new(
        ST7_S6,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3500,
                ordinal: 0x36,
                junko: false,
            },
            Op::I16 { at: 0xd898, v: 0 }, // void 316
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 14,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 120,
            },
        ],
    ),
    Warp::new(
        ST7_NS7,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x37 },
            Op::I16 { at: 0x6198, v: 0 },
            Op::I16 { at: 0x62c8, v: 0 },
            Op::I32 { at: 0x6440, v: 59 },
            Op::I32 { at: 0x6460, v: 0 },
            Op::I32 { at: 0x6080, v: 0 },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I16 { at: 0x63c0, v: 0 },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 7,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 132,
            },
            Op::Jump {
                start: 0x64d4,
                expect: 0x204,
                dest: 0x652c,
                at_frame: 24,
                ecl_time: 0,
            },
        ],
    ),
    Warp::new(
        ST7_S7,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3000,
                ordinal: 0x37,
                junko: false,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 7,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 132,
            },
        ],
    ),
    Warp::new(
        ST7_NS8,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 { at: 0x708, v: 0x38 },
            Op::I16 { at: 0x7004, v: 0 },
            Op::I16 { at: 0x708c, v: 0 },
            Op::I16 { at: 0x71bc, v: 0 },
            Op::Jump {
                start: 0x72ec,
                expect: 0x191,
                dest: 0x7354,
                at_frame: 0,
                ecl_time: 0,
            },
            Op::I32 { at: 0x6f54, v: 0 },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I16 { at: 0x72b4, v: 0 },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 116,
            },
            Op::I32 {
                at: ST7_BS_ANM_SELECT,
                v: 5,
            },
        ],
    ),
    Warp::new(
        ST7_S8,
        &[
            Op::St7BossEntry,
            Op::St7EnterSpell {
                health: 3000,
                ordinal: 0x38,
                junko: true,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_MAIN_ARG2,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: 116,
            },
            Op::I32 {
                at: ST7_BS_ANM_SELECT,
                v: 5,
            },
        ],
    ),
    Warp::with_phase(
        ST7_S9,
        &[
            Op::St7BossEntry,
            Op::File(3),
            Op::I32 {
                at: ST7_BS_MOVE_POS_X,
                v: 0,
            }, // center position
            Op::U32 {
                at: ST7_BS_MOVE_POS_Y,
                v: 0x4360_0000,
            },
            Op::I16 {
                at: ST7_BS_SET_SPRITE_INS,
                v: 0,
            }, // cancel boss sprite change
            Op::I32 {
                at: ST7_BS_ANM_SELECT,
                v: 5,
            }, // change selected anim
            Op::I32 {
                at: ST7_BS_ANM_SELECT2,
                v: 5,
            },
            Op::I16 {
                at: ST7_BS_ANM_SET_MAIN_INS,
                v: 0,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE1_ARG2,
                v: -1,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: -1,
            },
            Op::I16 {
                at: ST7_BS_MOVE_LIMIT_INS,
                v: 0,
            }, // remove move limits
            Op::SeqAt {
                pos: 0x6d0,
                expect: 0x16,
                words: &[
                    0,
                    0x0018_0132,
                    0x02FF_0000,
                    0,
                    0,
                    0, // 306
                    0,
                    0x0014_01F6,
                    0x01FF_0000,
                    0,
                    0x20, // flag
                    0,
                    0x0014_01FF,
                    0x01FF_0000,
                    0,
                    3000, // health
                    0,
                    0x0020_000B,
                    0x01FF_0000,
                    0,
                    0xC,
                    0x7373_6F42,
                    0x6472_6143,
                    0x39, // "BossCard9"
                ],
            },
        ],
        st7_s9_ops,
    ),
    Warp::with_phase(
        ST7_S10,
        &[
            Op::Jump {
                start: ST7_START,
                expect: 0x17,
                dest: ST7_BOSS_CREATE_CALL,
                at_frame: 60,
                ecl_time: 0,
            },
            Op::Skip(2),
            // Hecatia
            Op::File(3),
            Op::U32 {
                at: ST7_BS_MOVE_POS_X,
                v: 0xc280_0000,
            },
            Op::U32 {
                at: ST7_BS_MOVE_POS_Y,
                v: 0x4300_0000,
            },
            Op::I16 {
                at: ST7_BS_MOVE_LIMIT_INS,
                v: 0,
            }, // void 504
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE1_ARG2,
                v: -1,
            },
            Op::I32 {
                at: ST7_BS_ANM_SET_SPRITE2_ARG2,
                v: -1,
            },
            Op::SeqAt {
                pos: 0x6d0,
                expect: 0x16,
                words: &[
                    5,
                    0x0014_012E,
                    0x01FF_0000,
                    0,
                    5, // effect
                    5,
                    0x0018_012F,
                    0x02FF_0000,
                    0,
                    3,
                    6,
                    5,
                    0x0020_000F,
                    0x01FF_0000,
                    0,
                    0xC,
                    0x7373_6F42,
                    0x6F70_5F34,
                    0x0000_0073, // "Boss4_pos"
                    5,
                    0x0014_012E,
                    0x01FF_0000,
                    0,
                    3,
                    5,
                    0x0014_01FF,
                    0x01FF_0000,
                    0,
                    7000, // health
                    6,
                    0x0020_000B,
                    0x01FF_0000,
                    0,
                    0xC,
                    0x7373_6F42,
                    0x6472_6143,
                    0x3031, // "BossCard10"
                ],
            },
            // Junko
            Op::File(4),
            Op::U32 {
                at: 0x184,
                v: 0x4280_0000,
            },
            Op::U32 {
                at: 0x188,
                v: 0x4300_0000,
            },
            Op::I16 { at: 0x328, v: 0 },  // void 504
            Op::I32 { at: 0x36c, v: -1 }, // change 303-1
            Op::I32 { at: 0x384, v: -1 }, // change 303-2
            Op::I16 { at: 0x2b4, v: 0 },  // void 401
            Op::SeqAt {
                pos: 0x3b4,
                expect: 0xB,
                words: &[
                    5,
                    0x0014_01FF,
                    0x01FF_0000,
                    0,
                    7000, // health
                    5,
                    0x0014_0203,
                    0x01FF_0000,
                    0,
                    14, // invulnerability
                    9999,
                    0x0010_0000,
                    0x00FF_0000,
                    0, // stall
                ],
            },
        ],
        st7_s10_ops,
    ),
];

fn st5_mid1_ops(phase: Phase, emit: &mut Emit<'_>) {
    if let Phase::Wave(wave @ 1..) = phase {
        emit(PrimOp::Jump {
            start: ST5_START,
            expect: 0x17,
            dest: 0xa1a0,
            at_frame: 60,
            ecl_time: 0,
        });
        emit(PrimOp::Jump {
            start: ST5_WAVE_SRC,
            expect: 0x20C,
            dest: ST5_WAVE_DESTS[wave as usize - 1],
            at_frame: 0,
            ecl_time: 0,
        });
    } else {
        emit(PrimOp::Jump {
            start: ST5_START,
            expect: 0x17,
            dest: 0xa180,
            at_frame: 60,
            ecl_time: 0,
        });
        emit(PrimOp::Jump {
            start: 0x6ed8,
            expect: 0xB,
            dest: 0x7074,
            at_frame: 0,
            ecl_time: 0,
        });
    }
}

fn st6_boss11_ops(phase: Phase, emit: &mut Emit<'_>) {
    if let Phase::Health(tier @ 1..) = phase {
        emit(PrimOp::I32 {
            at: ST6_SPELL + 0x10,
            v: ST6_BOSS11_HEALTH[tier as usize - 1],
        });
    }
}

fn st7_s9_ops(phase: Phase, emit: &mut Emit<'_>) {
    const S9_DURATION: u32 = 0x10408 + 0x18;
    const S9_TR_START: u32 = 0x104b8;
    const S9_BLUE_PHASE_TIME: i32 = 120 * 3 + 180 * 5 + 280;

    // `S9(0)` starts from the blue phase, i.e. leaves the script as loaded.
    match phase {
        Phase::S9(1) => {
            // This compensates the duration and p2 -> p3 transition for the removed blue phase.
            emit(PrimOp::I32 {
                at: S9_DURATION,
                v: 5400 - S9_BLUE_PHASE_TIME + 90,
            });
            emit(PrimOp::I32 {
                at: 0x10bb0 + 0x10,
                v: 55 - (S9_BLUE_PHASE_TIME - 90) / 60,
            });
            emit(PrimOp::Jump {
                start: S9_TR_START,
                expect: 0x12C,
                dest: 0x10a50,
                at_frame: 0,
                ecl_time: 0,
            });
        }
        Phase::S9(2) => {
            emit(PrimOp::I32 {
                at: S9_DURATION,
                v: 5400 - (55 * 60) - (90 + 18),
            });
            emit(PrimOp::Jump {
                start: S9_TR_START,
                expect: 0x12C,
                dest: 0x10c7c,
                at_frame: 0,
                ecl_time: 0,
            });
        }
        Phase::None | Phase::Wave(_) | Phase::Health(_) | Phase::S9(_) | Phase::S10 { .. } => {}
    }
}

fn st7_s10_ops(phase: Phase, emit: &mut Emit<'_>) {
    if let Phase::S10 { hecatia, junko } = phase {
        if hecatia > 0 {
            emit(PrimOp::File(3));
            emit(PrimOp::Jump {
                start: ST7_S10_HECATIA_AT[0],
                expect: 0xB,
                dest: ST7_S10_HECATIA_AT[hecatia],
                at_frame: ST7_S10_HECATIA_AT_FRAME,
                ecl_time: 0,
            });
        }
        if junko > 0 {
            emit(PrimOp::File(4));
            emit(PrimOp::Jump {
                start: ST7_S10_JUNKO_AT[0],
                expect: 0xB,
                dest: ST7_S10_JUNKO_AT[junko],
                at_frame: ST7_S10_JUNKO_AT_FRAME,
                ecl_time: 0,
            });
        }
    }
}

const ST7_HIDE_SUBBOSS: &[Op] = &[
    Op::File(4),
    Op::I16 { at: 0x190, v: 0 },
    Op::I16 { at: 0x1dc, v: 0 },
    Op::I16 { at: 0x228, v: 0 },
    Op::Jump {
        start: 0x2a0,
        expect: 0x207,
        dest: 0x2f4,
        at_frame: 5,
        ecl_time: 0,
    },
    Op::SeqAt {
        pos: 0x480,
        expect: 0x191,
        words: &[
            0,
            0x0010_01F9,
            0x00FF_0000,
            0,
            0,
            0x0034_0190,
            0x02FF_0000,
            0,
            0x4340_0000,
            0xC280_0000,
        ],
    },
];

/// Chapter-warp entry-jump start offsets for each stage.
const CHAPTER_STARTS: [u32; 7] = [0x7248, 0x74a0, 0xa280, 0x8634, 0x9f34, 0x91f8, 0x8f94];

enum ChapterEffect {
    None,
    /// Defer `ECLSetChapter(n)` via the chapter intent.
    SetChapter(i32),
    /// Zero an `i16` at this file offset.
    ZeroI16(u32),
    /// Zero an `i32` at this file offset.
    ZeroI32(u32),
}

struct ChapterWarp {
    dest: u32,
    time: i32,
    /// An optional second jump in the same file.
    sub: Option<(u32, u16, u32)>,
    // An optional non-jump side effect.
    effect: ChapterEffect,
}

impl ChapterWarp {
    const fn new(dest: u32, time: i32) -> Self {
        Self {
            dest,
            time,
            sub: None,
            effect: ChapterEffect::None,
        }
    }

    const fn sub(mut self, from: u32, expect: u16, to: u32) -> Self {
        self.sub = Some((from, expect, to));
        self
    }

    const fn effect(mut self, effect: ChapterEffect) -> Self {
        self.effect = effect;
        self
    }
}

#[rustfmt::skip]
const CHAPTER_WARPS: &[(u32, u32, ChapterWarp)] = &[
    (1, 2, ChapterWarp::new(0x7494, 90).sub(0x3f4c, 0xB, 0x3fa8)),
    (1, 3, ChapterWarp::new(0x7544, 90).effect(ChapterEffect::SetChapter(2))),
    (1, 4, ChapterWarp::new(0x7544, 90).sub(0x4098, 0xF, 0x40fc)),
    (1, 5, ChapterWarp::new(0x75f8, 90).effect(ChapterEffect::SetChapter(5))),
    (1, 6, ChapterWarp::new(0x75f8, 90).sub(0x4150, 0xF, 0x41b8)),
    (2, 2, ChapterWarp::new(0x76ec, 90).sub(0x42b4, 0xB, 0x4354)),
    (2, 3, ChapterWarp::new(0x76ec, 90).sub(0x42b4, 0xB, 0x43d4)),
    (2, 4, ChapterWarp::new(0x778c, 29).effect(ChapterEffect::SetChapter(4))),
    (2, 5, ChapterWarp::new(0x778c, 0).sub(0x443c, 0xF, 0x448c)),
    (2, 6, ChapterWarp::new(0x778c, 90).sub(0x443c, 0xF, 0x450c)),
    (3, 2, ChapterWarp::new(0xa4cc, 90).sub(0x5b6c, 0xB, 0x5c00)),
    (3, 3, ChapterWarp::new(0xa4cc, 30).sub(0x5b6c, 0xB, 0x5c6c)),
    (3, 4, ChapterWarp::new(0xa56c, 50).effect(ChapterEffect::SetChapter(4))),
    (3, 5, ChapterWarp::new(0xa56c, 20).sub(0x5d0c, 0xF, 0x5d70)),
    (3, 6, ChapterWarp::new(0xa56c, 90).sub(0x5d0c, 0xF, 0x5df0)),
    (4, 2, ChapterWarp::new(0x8880, 90).sub(0x406c, 0xB, 0x40ec)),
    (4, 3, ChapterWarp::new(0x8880, 90).sub(0x406c, 0xB, 0x4158).effect(ChapterEffect::ZeroI32(0x5cc4))),
    (4, 4, ChapterWarp::new(0x8930, 90).effect(ChapterEffect::ZeroI32(0x6bc0))),
    (4, 5, ChapterWarp::new(0x8930, 90).sub(0x41e0, 0xF, 0x4244).effect(ChapterEffect::ZeroI32(0x6e74))),
    (4, 6, ChapterWarp::new(0x8930, 90).sub(0x41e0, 0xF, 0x42c4)),
    (5, 2, ChapterWarp::new(0xa180, 90).sub(0x6ed8, 0xB, 0x6f58)),
    (5, 3, ChapterWarp::new(0xa180, 90).sub(0x6ed8, 0xB, 0x7040).effect(ChapterEffect::ZeroI32(0x7ba0))),
    (5, 4, ChapterWarp::new(0xa180, 90).sub(0x6ed8, 0xB, 0x7098)),
    (5, 5, ChapterWarp::new(0xa1a0, 90).effect(ChapterEffect::ZeroI16(0x70f0))),
    (5, 6, ChapterWarp::new(0xa1a0, 90).sub(ST5_WAVE_SRC, 0x20C, ST5_WAVE_DESTS[1]).effect(ChapterEffect::ZeroI32(0x81c0))),
    (5, 7, ChapterWarp::new(0xa1a0, 90).sub(ST5_WAVE_SRC, 0x20C, ST5_WAVE_DESTS[2]).effect(ChapterEffect::ZeroI32(0x845c))),
    (5, 8, ChapterWarp::new(0xa1a0, 90).sub(ST5_WAVE_SRC, 0x20C, ST5_WAVE_DESTS[3]).effect(ChapterEffect::ZeroI32(0x87e0))),
    (5, 9, ChapterWarp::new(0xa1a0, 90).sub(ST5_WAVE_SRC, 0x20C, ST5_WAVE_DESTS[4]).effect(ChapterEffect::ZeroI32(0x8ce8))),
    (5, 10, ChapterWarp::new(0xa1a0, 90).sub(ST5_WAVE_SRC, 0x20C, ST5_WAVE_DESTS[5]).effect(ChapterEffect::ZeroI32(0x9a50))),
    (6, 2, ChapterWarp::new(0x93e0, 90).sub(0x2b34, 0xB, 0x2be8).effect(ChapterEffect::ZeroI32(0x38d0))),
    (6, 3, ChapterWarp::new(0x93e0, 90).sub(0x2b34, 0xB, 0x2c68).effect(ChapterEffect::ZeroI32(0x40ac))),
    (6, 4, ChapterWarp::new(0x93e0, 90).sub(0x2b34, 0xB, 0x2ce8).effect(ChapterEffect::ZeroI32(0x43b0))),
    (7, 2, ChapterWarp::new(0x9208, 90).sub(0x5d08, 0xB, 0x5d88)),
    (7, 3, ChapterWarp::new(0x9208, 90).sub(0x5d08, 0xB, 0x5e08)),
    (7, 4, ChapterWarp::new(0x9208, 90).sub(0x5d08, 0xB, 0x5e9c)),
    (7, 5, ChapterWarp::new(0x9208, 90).sub(0x5d08, 0xB, 0x5f30)),
    (7, 6, ChapterWarp::new(0x9338, 90)),
    (7, 7, ChapterWarp::new(0x9338, 10).sub(0x5f84, 0xB, 0x6018)),
    (7, 8, ChapterWarp::new(0x9338, 90).sub(0x5f84, 0xB, 0x60c0)),
    (7, 9, ChapterWarp::new(0x9338, 90).sub(0x5f84, 0xB, 0x6154)),
];

/// What a `(stage, portion)` pair dispatches to.
enum ChapterDispatch {
    StageStart,
    Warp(&'static ChapterWarp),
    Unmapped,
}

impl ChapterDispatch {
    const fn classify(stage: u32, portion: u32) -> Self {
        if portion == 1 {
            if stage >= 1 && stage <= STAGES {
                return Self::StageStart;
            }
            return Self::Unmapped;
        }
        let mut i = 0;
        while i < CHAPTER_WARPS.len() {
            if CHAPTER_WARPS[i].0 == stage && CHAPTER_WARPS[i].1 == portion {
                return Self::Warp(&CHAPTER_WARPS[i].2);
            }
            i += 1;
        }
        Self::Unmapped
    }
}

/// Emits the full stream of primitive ops for a chapter warp.
/// Returns whether the stage and portion mapped to a valid chapter warp. `false` means nothing was emitted.
fn expand_chapter(stage: u32, portion: u32, emit: &mut Emit<'_>) -> bool {
    let row = match ChapterDispatch::classify(stage, portion) {
        ChapterDispatch::StageStart => return true,
        ChapterDispatch::Warp(row) => row,
        ChapterDispatch::Unmapped => return false,
    };
    // Every `CHAPTER_STARTS` offset holds an id-0x2a instruction.
    emit(PrimOp::Jump {
        start: CHAPTER_STARTS[stage as usize - 1],
        expect: 0x2A,
        dest: row.dest,
        at_frame: 60,
        ecl_time: row.time,
    });
    if let Some((from, expect, to)) = row.sub {
        emit(PrimOp::Jump {
            start: from,
            expect,
            dest: to,
            at_frame: 0,
            ecl_time: 0,
        });
    }
    match row.effect {
        ChapterEffect::None => {}
        ChapterEffect::SetChapter(n) => emit(PrimOp::SetChapter(n)),
        ChapterEffect::ZeroI16(at) => emit(PrimOp::I16 { at, v: 0 }),
        ChapterEffect::ZeroI32(at) => emit(PrimOp::I32 { at, v: 0 }),
    }
    true
}

const _: () = {
    /// Returns whether the provided section accepts any nonzero phase value.
    const fn takes_phase(section: u32) -> bool {
        let mut v = 1;
        while v < 100 {
            if Phase::parse(section, v).is_some() {
                return true;
            }
            v += 1;
        }
        false
    }

    let ids = [
        ST1_MID1, ST1_MID2, ST1_MID3, ST1_BOSS1, ST1_BOSS2, ST1_BOSS3, ST1_BOSS4, ST2_MID1,
        ST2_BOSS1, ST2_BOSS2, ST2_BOSS3, ST2_BOSS4, ST2_BOSS5, ST2_BOSS6, ST3_MID1, ST3_MID2,
        ST3_BOSS1, ST3_BOSS2, ST3_BOSS3, ST3_BOSS4, ST3_BOSS5, ST3_BOSS6, ST3_BOSS7, ST4_MID1,
        ST4_BOSS1, ST4_BOSS2, ST4_BOSS3, ST4_BOSS4, ST4_BOSS5, ST4_BOSS6, ST4_BOSS7, ST5_MID1,
        ST5_BOSS1, ST5_BOSS2, ST5_BOSS3, ST5_BOSS4, ST5_BOSS5, ST5_BOSS6, ST5_BOSS7, ST5_BOSS8,
        ST6_STARS, ST6_BOSS1, ST6_BOSS2, ST6_BOSS3, ST6_BOSS4, ST6_BOSS5, ST6_BOSS6, ST6_BOSS7,
        ST6_BOSS8, ST6_BOSS9, ST6_BOSS10, ST6_BOSS11, ST7_MID1, ST7_MID2, ST7_MID3, ST7_NS1,
        ST7_S1, ST7_NS2, ST7_S2, ST7_NS3, ST7_S3, ST7_NS4, ST7_S4, ST7_NS5, ST7_S5, ST7_NS6,
        ST7_S6, ST7_NS7, ST7_S7, ST7_NS8, ST7_S8, ST7_S9, ST7_S10,
    ];
    assert!(
        ids.len() == WARPS.len(),
        "declared section IDs and catalog rows drifted"
    );
    let mut i = 0;
    while i < ids.len() {
        assert!(
            section_stage(ids[i]).is_some(),
            "section ID doesn't decode to a stage"
        );
        let Some(row) = warp_index(ids[i]) else {
            panic!("declared section ID doesn't have a catalog row");
        };
        assert!(
            takes_phase(ids[i]) == WARPS[row].phase_ops.is_some(),
            "phase validity (Phase::parse) and phase dispatch (phase_ops) drifted"
        );
        i += 1;
    }

    let mut c = 0;
    while c < CHAPTER_WARPS.len() {
        let (stage, portion, _) = CHAPTER_WARPS[c];
        assert!(
            stage >= 1 && stage <= STAGES && portion >= 2,
            "chapter row must specify a real stage and a warpable portion",
        );
        c += 1;
    }

    let mut w = 0;
    while w < WARPS.len() {
        let ops = WARPS[w].ops;
        let mut o = 0;
        while o < ops.len() {
            if let Op::BossEntry { stage, .. } | Op::Nonspell { stage, .. } = ops[o] {
                assert!(
                    stage >= 1 && stage <= BOSS_ENTRIES.len(),
                    "BossEntry/Nonspell stage must index BOSS_ENTRIES"
                );
            }
            o += 1;
        }
        w += 1;
    }
};
