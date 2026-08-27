//! Implementation of `DirectSound` for headless execution.

// `#[implement]` triggers these lints.
#![expect(trivial_casts, clippy::inline_always, clippy::ref_as_ptr)]

use super::backing::Backing;
use super::out::{get_zeroed, get_zeroed_array, put};
use crate::iat::{ImportRef, hook_import};
use std::cmp::min;
use std::convert::identity;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Foundation::{E_FAIL, E_NOTIMPL, HWND, S_OK};
use windows::Win32::Media::Audio::DirectSound::{
    DS_CERTIFIED, DSBCAPS, DSBPOSITIONNOTIFY, DSBSTATUS_LOOPING, DSBSTATUS_PLAYING, DSBUFFERDESC,
    DSCAPS, DSEFFECTDESC, DSSPEAKER_STEREO, IDirectSound, IDirectSound_Impl, IDirectSound8,
    IDirectSound8_Impl, IDirectSoundBuffer, IDirectSoundBuffer_Impl, IDirectSoundBuffer8,
    IDirectSoundBuffer8_Impl, IDirectSoundNotify, IDirectSoundNotify_Impl,
};
use windows::Win32::Media::Audio::{WAVE_FORMAT_PCM, WAVEFORMATEX};
use windows::core::{
    Error, GUID, HRESULT, IUnknown, Interface as _, OutRef, Ref, Result, implement,
};
use windows_sys::Win32::Foundation::HMODULE;

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(super) unsafe fn install(game: HMODULE) {
    unsafe {
        hook_import(
            game,
            ImportRef::Ordinal {
                dll: "DSOUND.dll",
                ordinal: 11,
            },
            hook_directsound_create8 as *mut (),
        );
    }
}

unsafe extern "system" fn hook_directsound_create8(
    _device_guid: *const c_void,
    out_ds8: *mut *mut c_void,
    _unk_outer: *mut c_void,
) -> HRESULT {
    if out_ds8.is_null() {
        return E_FAIL;
    }
    let obj: IDirectSound8 = FakeDirectSound.into();
    unsafe { out_ds8.write(obj.into_raw()) };
    S_OK
}

#[implement(IDirectSound8)]
struct FakeDirectSound;

impl IDirectSound_Impl for FakeDirectSound_Impl {
    fn CreateSoundBuffer(
        &self,
        pcdsbufferdesc: *const DSBUFFERDESC,
        ppdsbuffer: OutRef<'_, IDirectSoundBuffer>,
        _punkouter: Ref<'_, IUnknown>,
    ) -> Result<()> {
        let bytes = if pcdsbufferdesc.is_null() {
            0
        } else {
            unsafe { (*pcdsbufferdesc).dwBufferBytes }
        };
        let buffer8: IDirectSoundBuffer8 = FakeSoundBuffer::new(bytes).into();
        ppdsbuffer.write(Some(buffer8.cast()?))
    }

    fn GetCaps(&self, pdscaps: *mut DSCAPS) -> Result<()> {
        if !pdscaps.is_null() {
            unsafe {
                (*pdscaps).dwFlags = 0;
                (*pdscaps).dwMinSecondarySampleRate = 100;
                (*pdscaps).dwMaxSecondarySampleRate = 200_000;
                (*pdscaps).dwPrimaryBuffers = 1;
            }
        }
        Ok(())
    }

    fn DuplicateSoundBuffer(
        &self,
        pdsbufferoriginal: Ref<'_, IDirectSoundBuffer>,
    ) -> Result<IDirectSoundBuffer> {
        let bytes = pdsbufferoriginal.as_ref().map_or(0, |orig| {
            let mut caps = DSBCAPS {
                #[expect(clippy::cast_possible_truncation)]
                dwSize: size_of::<DSBCAPS>() as u32,
                ..Default::default()
            };
            drop(unsafe { orig.GetCaps(&raw mut caps) });
            caps.dwBufferBytes
        });
        let buffer8: IDirectSoundBuffer8 = FakeSoundBuffer::new(bytes).into();
        buffer8.cast()
    }

    fn SetCooperativeLevel(&self, _hwnd: HWND, _dwlevel: u32) -> Result<()> {
        Ok(())
    }

    fn Compact(&self) -> Result<()> {
        Ok(())
    }

    fn GetSpeakerConfig(&self) -> Result<u32> {
        Ok(DSSPEAKER_STEREO)
    }

    fn SetSpeakerConfig(&self, _dwspeakerconfig: u32) -> Result<()> {
        Ok(())
    }

    fn Initialize(&self, _pcguiddevice: *const GUID) -> Result<()> {
        Ok(())
    }
}

impl IDirectSound8_Impl for FakeDirectSound_Impl {
    fn VerifyCertification(&self) -> Result<u32> {
        Ok(DS_CERTIFIED)
    }
}

#[implement(IDirectSoundBuffer8, IDirectSoundNotify)]
struct FakeSoundBuffer {
    backing: Backing,
    requested: u32,
    cursor: AtomicU32,
    volume: AtomicU32,
}

impl FakeSoundBuffer {
    fn new(bytes: u32) -> Self {
        Self {
            backing: Backing::new(bytes as usize),
            requested: bytes,
            cursor: AtomicU32::new(0),
            volume: AtomicU32::new(0),
        }
    }

    #[expect(clippy::cast_possible_truncation)]
    fn len(&self) -> u32 {
        self.backing.len() as u32
    }
}

impl IDirectSoundNotify_Impl for FakeSoundBuffer_Impl {
    fn SetNotificationPositions(
        &self,
        _dwpositionnotifies: u32,
        _pcpositionnotifies: *const DSBPOSITIONNOTIFY,
    ) -> Result<()> {
        Ok(())
    }
}

impl IDirectSoundBuffer_Impl for FakeSoundBuffer_Impl {
    fn GetCaps(&self, pdsbuffercaps: *mut DSBCAPS) -> Result<()> {
        if !pdsbuffercaps.is_null() {
            unsafe {
                (*pdsbuffercaps).dwFlags = 0;
                (*pdsbuffercaps).dwBufferBytes = self.requested;
                (*pdsbuffercaps).dwUnlockTransferRate = 0;
                (*pdsbuffercaps).dwPlayCpuOverhead = 0;
            }
        }
        Ok(())
    }

    fn GetCurrentPosition(
        &self,
        pdwcurrentplaycursor: *mut u32,
        pdwcurrentwritecursor: *mut u32,
    ) -> Result<()> {
        // 44100 Hz, 2 channels (stereo), 16-bit
        const BYTES_PER_TICK: u32 = 44_100 * 2 * 2 / 60;

        let len = self.len();
        let play = self
            .cursor
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cursor| {
                Some((cursor + BYTES_PER_TICK) % len)
            })
            .unwrap_or_else(identity); // the closure never returns `None`
        unsafe {
            put(pdwcurrentplaycursor, play).unwrap();
            put(pdwcurrentwritecursor, (play + BYTES_PER_TICK) % len).unwrap();
        }
        Ok(())
    }

    fn GetFormat(
        &self,
        pwfxformat: *mut WAVEFORMATEX,
        dwsizeallocated: u32,
        pdwsizewritten: *mut u32,
    ) -> Result<()> {
        if !pwfxformat.is_null() && dwsizeallocated as usize >= size_of::<WAVEFORMATEX>() {
            unsafe {
                pwfxformat.write(WAVEFORMATEX {
                    #[expect(clippy::cast_possible_truncation)]
                    wFormatTag: WAVE_FORMAT_PCM as u16,
                    nChannels: 2,
                    nSamplesPerSec: 44_100,
                    nAvgBytesPerSec: 176_400,
                    nBlockAlign: 4,
                    wBitsPerSample: 16,
                    cbSize: 0,
                });
            }
        }
        #[expect(clippy::cast_possible_truncation)]
        let size = size_of::<WAVEFORMATEX>() as u32;
        unsafe { put(pdwsizewritten, size) }
    }

    fn GetVolume(&self) -> Result<i32> {
        Ok(self.volume.load(Ordering::Relaxed).cast_signed())
    }

    fn GetPan(&self) -> Result<i32> {
        Ok(0)
    }

    fn GetFrequency(&self) -> Result<u32> {
        Ok(44_100)
    }

    fn GetStatus(&self) -> Result<u32> {
        Ok(DSBSTATUS_PLAYING | DSBSTATUS_LOOPING)
    }

    fn Initialize(
        &self,
        _pdirectsound: Ref<'_, IDirectSound>,
        _pcdsbufferdesc: *const DSBUFFERDESC,
    ) -> Result<()> {
        Ok(())
    }

    fn Lock(
        &self,
        dwoffset: u32,
        dwbytes: u32,
        ppvaudioptr1: *mut *mut c_void,
        pdwaudiobytes1: *mut u32,
        ppvaudioptr2: *mut *mut c_void,
        pdwaudiobytes2: *mut u32,
        _dwflags: u32,
    ) -> Result<()> {
        let len = self.len();
        let offset = dwoffset % len;
        let first = min(dwbytes, len.saturating_sub(offset));
        let second = min(dwbytes - first, len);
        if !ppvaudioptr1.is_null() {
            unsafe {
                ppvaudioptr1.write(self.backing.ptr_at(offset as usize, first as usize).cast());
            }
        }
        unsafe { put(pdwaudiobytes1, first) }.unwrap();
        if !ppvaudioptr2.is_null() {
            let ptr = if second > 0 {
                self.backing.ptr_at(0, second as usize).cast()
            } else {
                null_mut()
            };
            unsafe { ppvaudioptr2.write(ptr) };
        }
        unsafe { put(pdwaudiobytes2, second) }.unwrap();
        Ok(())
    }

    fn Play(&self, _dwreserved1: u32, _dwpriority: u32, _dwflags: u32) -> Result<()> {
        Ok(())
    }

    fn SetCurrentPosition(&self, dwnewposition: u32) -> Result<()> {
        self.cursor
            .store(dwnewposition % self.len(), Ordering::Relaxed);
        Ok(())
    }

    fn SetFormat(&self, _pcfxformat: *const WAVEFORMATEX) -> Result<()> {
        Ok(())
    }

    fn SetVolume(&self, lvolume: i32) -> Result<()> {
        self.volume
            .store(lvolume.cast_unsigned(), Ordering::Relaxed);
        Ok(())
    }

    fn SetPan(&self, _lpan: i32) -> Result<()> {
        Ok(())
    }

    fn SetFrequency(&self, _dwfrequency: u32) -> Result<()> {
        Ok(())
    }

    fn Stop(&self) -> Result<()> {
        Ok(())
    }

    fn Unlock(
        &self,
        _pvaudioptr1: *const c_void,
        _dwaudiobytes1: u32,
        _pvaudioptr2: *const c_void,
        _dwaudiobytes2: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn Restore(&self) -> Result<()> {
        Ok(())
    }
}

impl IDirectSoundBuffer8_Impl for FakeSoundBuffer_Impl {
    fn SetFX(
        &self,
        dweffectscount: u32,
        _pdsfxdesc: *const DSEFFECTDESC,
        pdwresultcodes: *mut u32,
    ) -> Result<()> {
        unsafe { get_zeroed_array(pdwresultcodes, dweffectscount as usize) }
    }

    fn AcquireResources(
        &self,
        _dwflags: u32,
        dweffectscount: u32,
        pdwresultcodes: *mut u32,
    ) -> Result<()> {
        unsafe { get_zeroed_array(pdwresultcodes, dweffectscount as usize) }
    }

    fn GetObjectInPath(
        &self,
        _rguidobject: *const GUID,
        _dwindex: u32,
        _rguidinterface: *const GUID,
        ppobject: *mut *mut c_void,
    ) -> Result<()> {
        unsafe { get_zeroed(ppobject)? };
        Err(Error::from_hresult(E_NOTIMPL))
    }
}
