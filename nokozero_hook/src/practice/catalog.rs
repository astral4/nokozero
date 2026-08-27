//! Dispatch and interpreter over live ECL buffers.

use super::data::{Phase, PrimOp, SectionId, WARPS, expand_chapter, warp_index};
use super::ecl::Ecl;

/// Chapter effects requested by the dispatched section.
#[derive(Default)]
pub(super) struct ChapterIntent {
    pub(super) skip: i32,
    pub(super) set: Option<i32>,
    pub(super) st7_bonus: bool,
}

impl ChapterIntent {
    fn skip_chapters(&mut self, times: i32) {
        self.skip = times;
    }

    fn set_chapter(&mut self, value: i32) {
        self.set = Some(value);
    }

    fn request_st7_chapter_bonus(&mut self) {
        self.st7_bonus = true;
    }
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
            PrimOp::Skip(n) => intent.skip_chapters(n),
            PrimOp::SetChapter(n) => intent.set_chapter(n),
            PrimOp::St7Bonus => intent.request_st7_chapter_bonus(),
        }
    }
}

/// Applies the warp corresponding to the provided `section` and `phase`. Returns the chapter effects to execute.
/// `None` means the section and phase didn't match a warp, and nothing was patched.
///
/// # Safety
///
/// A stage must be loaded and `ecl` must be newly created for this stage load.
pub(super) unsafe fn apply_section(
    ecl: &mut Ecl,
    section: u32,
    phase: u32,
) -> Option<ChapterIntent> {
    let phase = Phase::parse(section, phase)?;
    let mut intent = ChapterIntent::default();
    let mut emit = |op: PrimOp<'_>| {
        unsafe { apply_op(ecl, &mut intent, op) };
    };
    let mapped = match SectionId::classify(section) {
        SectionId::Chapter { stage, portion } => expand_chapter(stage, portion, &mut emit),
        SectionId::Named { .. } => {
            let index = warp_index(section)?;
            WARPS[index].expand(phase, &mut emit);
            true
        }
    };
    mapped.then_some(intent)
}
