//! Prevents the game's ANM VM pool from overflowing.

use crate::patch::NearBranchSite;

const CANCEL_EFFECT_SKIP: NearBranchSite =
    NearBranchSite::new(0x0041_e2d9, 0x88, 0x0041_e3ad, "bullet-cancel effect skip");

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(crate) unsafe fn install() {
    unsafe {
        const { CANCEL_EFFECT_SKIP.force() }.apply();
    }
}
