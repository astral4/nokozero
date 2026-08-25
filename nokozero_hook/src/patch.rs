//! In-process byte patching primitives.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::process::abort;
use std::ptr::{copy_nonoverlapping, with_exposed_provenance, with_exposed_provenance_mut};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, PAGE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The length of a rel32 branch (opcode plus disp32).
const REL32_LEN: usize = 5;

/// The length of a near conditional branch (`0F`-prefixed opcode plus disp32).
const NEAR_LEN: usize = 6;

/// When a byte mismatch was caught.
#[derive(Clone, Copy)]
enum Stage {
    /// The site does not hold its expected bytes.
    PreCheck,
    /// The site does not hold the patched bytes after writing.
    PostWrite,
}

impl Stage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PreCheck => "pre-check",
            Self::PostWrite => "post-write",
        }
    }
}

/// Utility for formatting byte slices as space-separated hexadecimal values.
struct Hex<'a>(&'a [u8]);

impl Display for Hex<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for (i, b) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Returns the little-endian disp32 of a branch whose next instruction starts at `next`, targeting `target`.
const fn disp32(next: u32, target: u32) -> [u8; 4] {
    target.wrapping_sub(next).to_le_bytes()
}

/// Returns the bytes of a rel32 branch at `site` targeting `target`.
const fn rel32(opcode: u8, site: u32, target: u32) -> [u8; 5] {
    // The displacement is relative to the byte after the branch.
    #[expect(clippy::cast_possible_truncation)]
    let d = disp32(site.wrapping_add(REL32_LEN as u32), target);
    [opcode, d[0], d[1], d[2], d[3]]
}

/// A patch site.
#[derive(Clone, Copy)]
pub(crate) struct Site<const N: usize> {
    addr: u32,
    expected: [u8; N],
    name: &'static str,
}

impl<const N: usize> Site<N> {
    #[must_use]
    pub(crate) const fn new(addr: u32, expected: [u8; N], name: &'static str) -> Self {
        Self {
            addr,
            expected,
            name,
        }
    }

    /// Constructs a [`Patch`] replacing the site's bytes with `replacement`.
    pub(crate) const fn patch(self, replacement: [u8; N]) -> Patch<N> {
        Patch {
            site: self,
            replacement,
        }
    }

    unsafe fn write_relative_branch(&self, hook: *mut (), opcode: u8) {
        const { assert!(N >= REL32_LEN, "rel32 branch needs at least 5 bytes") };

        #[expect(clippy::cast_possible_truncation)]
        let hook_addr = hook.expose_provenance() as u32;

        let mut bytes = [0x90u8; N];
        bytes[..REL32_LEN].copy_from_slice(&rel32(opcode, self.addr, hook_addr));

        unsafe {
            expect_bytes(self.addr, &self.expected, self.name, Stage::PreCheck);
            patch_bytes(self.addr, &bytes, self.name);
        }

        let actual = unsafe { read_at::<N>(self.addr) };
        let read_disp = i32::from_le_bytes([actual[1], actual[2], actual[3], actual[4]]);
        #[expect(clippy::cast_possible_truncation)]
        let resolved = self
            .addr
            .wrapping_add(REL32_LEN as u32)
            .wrapping_add_signed(read_disp);

        let ok = actual[0] == opcode
            && resolved == hook_addr
            && actual[REL32_LEN..] == bytes[REL32_LEN..];
        if !ok {
            eprintln!(
                "nokozero_hook: install failed at {}: post-write mismatch at {:#010x}: \
                wrote [{}] found [{}], branch lands at {resolved:#010x}, hook is at {hook_addr:#010x}",
                self.name,
                self.addr,
                Hex(&bytes),
                Hex(&actual),
            );
            abort();
        }
    }
}

/// An `E8 rel32` direct call.
pub(crate) struct CallSite(Site<REL32_LEN>);

impl CallSite {
    #[must_use]
    pub(crate) const fn new(addr: u32, callee: u32, name: &'static str) -> Self {
        Self(Site::new(addr, rel32(0xe8, addr, callee), name))
    }

    /// Rewrites the call to enter `hook` in place of the original callee. `hook` returns into the instruction after the call.
    ///
    /// # Safety
    ///
    /// The site must be mapped, and no other thread may execute or write its page for the duration.
    /// In practice, this means this function should be called during `DLL_PROCESS_ATTACH`.
    /// `hook` must have the original callee's calling convention and signature.
    pub(crate) unsafe fn retarget(&self, hook: *mut ()) {
        unsafe { self.0.write_relative_branch(hook, 0xe8) };
    }
}

/// A 6-byte near conditional branch `0F cc rel32`.
pub(crate) struct NearBranchSite {
    site: Site<NEAR_LEN>,
}

impl NearBranchSite {
    #[must_use]
    pub(crate) const fn new(addr: u32, opcode: u8, taken_target: u32, name: &'static str) -> Self {
        assert!(opcode & 0xf0 == 0x80, "not a near conditional branch");
        #[expect(clippy::cast_possible_truncation)]
        let d = disp32(addr.wrapping_add(NEAR_LEN as u32), taken_target);
        Self {
            site: Site::new(addr, [0x0f, opcode, d[0], d[1], d[2], d[3]], name),
        }
    }

    /// Constructs a [`Patch`] forcing the branch to always be taken.
    pub(crate) const fn force(self) -> Patch<NEAR_LEN> {
        let e = self.site.expected;
        self.site.patch([0x90, 0xe9, e[2], e[3], e[4], e[5]])
    }
}

/// A fixed-length byte replacement at a [`Site`].
#[must_use]
pub(crate) struct Patch<const N: usize> {
    site: Site<N>,
    replacement: [u8; N],
}

impl<const N: usize> Patch<N> {
    /// Verifies the site holds its expected bytes, writes the replacement, and verifies the site holds the patched bytes after writing.
    ///
    /// # Safety
    ///
    /// The site must be mapped, and no other thread may execute or write its page for the duration.
    /// In practice, this means this function should be called during `DLL_PROCESS_ATTACH`.
    pub(crate) unsafe fn apply(&self) {
        let Site {
            addr,
            expected,
            name,
        } = self.site;
        unsafe {
            expect_bytes(addr, &expected, name, Stage::PreCheck);
            patch_bytes(addr, &self.replacement, name);
            expect_bytes(addr, &self.replacement, name, Stage::PostWrite);
        }
    }
}

unsafe fn patch_bytes(addr: u32, src: &[u8], name: &str) {
    let dst = with_exposed_provenance_mut(addr as usize);
    let written = unsafe {
        with_writable(dst, src.len(), PAGE_READWRITE, |p| {
            copy_nonoverlapping(src.as_ptr(), p, src.len());
        })
    };
    if written.is_none() {
        eprintln!("nokozero_hook: install failed at {name}");
        abort();
    }
}

/// Aborts if the live bytes at `addr` do not equal `expected`.
///
/// # Safety
///
/// `addr` must be readable for `N` bytes.
unsafe fn expect_bytes<const N: usize>(addr: u32, expected: &[u8; N], name: &str, stage: Stage) {
    let actual = unsafe { read_at::<N>(addr) };
    if actual != *expected {
        eprintln!(
            "nokozero_hook: install failed at {name}: {} mismatch at {addr:#010x}: expected [{}] found [{}]",
            stage.as_str(),
            Hex(expected),
            Hex(&actual),
        );
        abort();
    }
}

/// # Safety
///
/// `addr` must be readable for `N` bytes.
unsafe fn read_at<const N: usize>(addr: u32) -> [u8; N] {
    let mut buf = [0u8; N];
    let src = with_exposed_provenance::<u8>(addr as usize);
    unsafe { copy_nonoverlapping(src, buf.as_mut_ptr(), N) };
    buf
}

/// Reprotects a region to `prot` for the duration of `f`, restores the original protection, and flushes the instruction cache.
/// `None` means the initial protection change failed and `f` was not called.
///
/// # Safety
///
/// If `prot` excludes EXECUTE, no other thread may execute anywhere on the affected pages.
/// Writes through `f` must stay within `[addr, addr + len)`.
#[must_use]
unsafe fn with_writable<R>(
    addr: *mut u8,
    len: usize,
    prot: PAGE_PROTECTION_FLAGS,
    f: impl FnOnce(*mut u8) -> R,
) -> Option<R> {
    let target = addr.cast();
    let mut saved = 0;
    if unsafe { VirtualProtect(target, len, prot, &raw mut saved) } == 0 {
        return None;
    }
    let result = f(addr);
    unsafe {
        VirtualProtect(target, len, saved, &raw mut saved);
        FlushInstructionCache(GetCurrentProcess(), target.cast_const(), len);
    }
    Some(result)
}
