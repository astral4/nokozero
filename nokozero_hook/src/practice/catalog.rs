//! Dispatch and interpreter over live ECL buffers.

use super::chapter::ChapterIntent;
use super::data::{PrimOp, expand_section};
use super::ecl::Ecl;

/// Applies the warp corresponding to the provided `section` and `phase`. Returns the chapter effects to execute.
///
/// # Safety
///
/// A stage must be loaded and `ecl` must be newly created for this stage load.
pub(super) unsafe fn apply_section(ecl: &mut Ecl, section: u32, phase: u32) -> ChapterIntent {
    let mut intent = ChapterIntent::NONE;
    assert!(
        expand_section(section, phase, &mut |op| {
            unsafe { apply_op(ecl, &mut intent, op) };
        }),
        "parse-validated warp target failed to dispatch"
    );
    intent
}

/// # Safety
///
/// A stage must be loaded and `ecl` must be newly created for this stage load.
unsafe fn apply_op(ecl: &mut Ecl, intent: &mut ChapterIntent, op: PrimOp<'_>) {
    unsafe {
        match op {
            PrimOp::File(n) => ecl.set_file(n),
            PrimOp::Pos(p) => ecl.set_pos(p),
            PrimOp::Jump {
                start,
                expect,
                dest,
                at_frame,
                ecl_time,
            } => ecl.jump(start, expect, dest, at_frame, ecl_time),
            PrimOp::SeqAt { pos, expect, words } => ecl.write_seq_at(pos, expect, words),
            PrimOp::Seq { words } => ecl.write_seq(words),
            PrimOp::I8 { at, v } => ecl.write_at(at, v),
            PrimOp::I16 { at, v } => ecl.write_at(at, v),
            PrimOp::I32 { at, v } => ecl.write_at(at, v),
            PrimOp::U32 { at, v } => ecl.write_at(at, v),
            PrimOp::Skip(n) => intent.skip_remaining = n,
            PrimOp::SetChapter(n) => intent.set_chapter = Some(n),
            PrimOp::St7Bonus => intent.st7_bonus = true,
        }
    }
}
