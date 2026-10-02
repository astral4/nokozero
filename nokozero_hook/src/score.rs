//! Prevents the game from writing to its score file (`scoreth15.dat`).
//! Instances killed mid-write would leave the file truncated, causing the next boot to read garbage data.

use crate::patch::{CallSite, Site};

extern "fastcall" fn no_score_file(_path: *const u8, _size: *mut u32, _flag: u32) -> u32 {
    0
}

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(crate) unsafe fn install() {
    unsafe {
        CallSite::new(0x0045_dbfd, 0x0040_2db0, "score load skip")
            .retarget(no_score_file as *mut ());
        Site::new(
            0x0045_df50,
            [0x55, 0x8B, 0xEC, 0x83, 0xEC],
            "score save skip",
        )
        .patch([0x83, 0xC8, 0xFF, 0xC3, 0x90])
        .apply();
    }
}
