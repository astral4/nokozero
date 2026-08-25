//! Implementation of `Direct3D9` for headless execution.

// `#[implement]` triggers these lints.
#![expect(trivial_casts, clippy::inline_always, clippy::ref_as_ptr)]

use super::backing::Backing;
use super::out::{get_zeroed, get_zeroed_array};
use crate::iat::{ImportRef, hook_import};
use std::cmp::max;
use std::ffi::c_void;
use std::iter::zip;
use std::process::abort;
use std::ptr::null_mut;
use std::sync::Arc;
use windows::Win32::Foundation::{E_NOTIMPL, HANDLE, HWND, POINT, RECT};
use windows::Win32::Graphics::Direct3D9::{
    D3DADAPTER_IDENTIFIER9, D3DBACKBUFFER_TYPE, D3DCAPS2_CANMANAGERESOURCE,
    D3DCAPS2_DYNAMICTEXTURES, D3DCAPS9, D3DCLIPSTATUS9, D3DDEVICE_CREATION_PARAMETERS, D3DDEVTYPE,
    D3DDEVTYPE_HAL, D3DDISPLAYMODE, D3DFMT_A1R5G5B5, D3DFMT_A2B10G10R10, D3DFMT_A2R10G10B10,
    D3DFMT_A4L4, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8L8, D3DFMT_A8R3G3B2,
    D3DFMT_A8R8G8B8, D3DFMT_D16, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_G16R16, D3DFMT_L8,
    D3DFMT_L16, D3DFMT_R3G3B2, D3DFMT_R5G6B5, D3DFMT_R8G8B8, D3DFMT_UNKNOWN, D3DFMT_VERTEXDATA,
    D3DFMT_X1R5G5B5, D3DFMT_X4R4G4B4, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8, D3DFORMAT, D3DGAMMARAMP,
    D3DINDEXBUFFER_DESC, D3DLIGHT9, D3DLOCKED_RECT, D3DMATERIAL9, D3DMULTISAMPLE_NONE,
    D3DMULTISAMPLE_TYPE, D3DPOOL, D3DPOOL_DEFAULT, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPRESENT_INTERVAL_ONE, D3DPRESENT_PARAMETERS, D3DPRIMITIVETYPE, D3DQUERYTYPE,
    D3DRASTER_STATUS, D3DRECT, D3DRECTPATCH_INFO, D3DRENDERSTATETYPE, D3DRESOURCETYPE,
    D3DRTYPE_INDEXBUFFER, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DRTYPE_VERTEXBUFFER,
    D3DSAMPLERSTATETYPE, D3DSTATEBLOCKTYPE, D3DSURFACE_DESC, D3DTEXF_LINEAR, D3DTEXTUREFILTERTYPE,
    D3DTEXTURESTAGESTATETYPE, D3DTRANSFORMSTATETYPE, D3DTRIPATCH_INFO, D3DVERTEXBUFFER_DESC,
    D3DVERTEXELEMENT9, D3DVIEWPORT9, IDirect3D9, IDirect3D9_Impl, IDirect3DBaseTexture9,
    IDirect3DBaseTexture9_Impl, IDirect3DCubeTexture9, IDirect3DDevice9, IDirect3DDevice9_Impl,
    IDirect3DIndexBuffer9, IDirect3DIndexBuffer9_Impl, IDirect3DPixelShader9, IDirect3DQuery9,
    IDirect3DResource9_Impl, IDirect3DStateBlock9, IDirect3DSurface9, IDirect3DSurface9_Impl,
    IDirect3DSwapChain9, IDirect3DTexture9, IDirect3DTexture9_Impl, IDirect3DVertexBuffer9,
    IDirect3DVertexBuffer9_Impl, IDirect3DVertexDeclaration9, IDirect3DVertexShader9,
    IDirect3DVolumeTexture9,
};
use windows::Win32::Graphics::Gdi::{HDC, HMONITOR, PALETTEENTRY, RGNDATA};
use windows::core::{
    BOOL, Error, GUID, HRESULT, IUnknownImpl as _, Interface as _, OutRef, Ref, Result, implement,
};
use windows_numerics::Matrix4x4;
use windows_sys::Win32::Foundation::HMODULE;

const D3DERR_INVALIDCALL: HRESULT = HRESULT(0x8876_086C_u32.cast_signed());

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(super) unsafe fn install(game: HMODULE) {
    unsafe {
        hook_import(
            game,
            ImportRef::Name("Direct3DCreate9"),
            hook_direct3dcreate9 as *mut (),
        );
    }
}

unsafe extern "system" fn hook_direct3dcreate9(_sdk_version: u32) -> *mut c_void {
    let obj: IDirect3D9 = FakeD3d9.into();
    obj.into_raw()
}

/// # Safety
///
/// `out` must be null or point to a writable `T`.
unsafe fn write_out<T>(out: *mut T, value: T) -> Result<()> {
    if out.is_null() {
        return Err(Error::from_hresult(D3DERR_INVALIDCALL));
    }
    unsafe { out.write(value) };
    Ok(())
}

/// `Lock` implementation for the vertex and index buffers. A `size` of 0 means "to the end".
///
/// # Safety
///
/// `ppbdata` must be null or a valid writable `*mut *mut c_void`.
unsafe fn lock_backing(
    backing: &Backing,
    offset: u32,
    size: u32,
    ppbdata: *mut *mut c_void,
) -> Result<()> {
    if ppbdata.is_null() {
        return Err(Error::from_hresult(D3DERR_INVALIDCALL));
    }
    let offset = offset as usize;
    let bytes = match size {
        0 => backing.len().saturating_sub(offset),
        _ => size as usize,
    };
    unsafe { ppbdata.write(backing.ptr_at(offset, bytes).cast()) };
    Ok(())
}

fn bytes_per_pixel(format: D3DFORMAT) -> u32 {
    match format {
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 | D3DFMT_A8B8G8R8 | D3DFMT_X8B8G8R8
        | D3DFMT_A2B10G10R10 | D3DFMT_A2R10G10B10 | D3DFMT_G16R16 | D3DFMT_D32 | D3DFMT_D24S8
        | D3DFMT_D24X8 => 4,
        D3DFMT_R8G8B8 => 3,
        D3DFMT_R5G6B5 | D3DFMT_A1R5G5B5 | D3DFMT_X1R5G5B5 | D3DFMT_A4R4G4B4 | D3DFMT_A8R3G3B2
        | D3DFMT_X4R4G4B4 | D3DFMT_A8L8 | D3DFMT_D16 | D3DFMT_L16 => 2,
        D3DFMT_A8 | D3DFMT_L8 | D3DFMT_R3G3B2 | D3DFMT_A4L4 => 1,
        other => {
            eprintln!("nokozero_hook: d3d9: unknown D3DFORMAT: {}", other.0);
            abort();
        }
    }
}

fn row_pitch(width: u32, format: D3DFORMAT) -> u32 {
    width
        .checked_mul(bytes_per_pixel(format))
        .and_then(|bytes| bytes.checked_next_multiple_of(4))
        .unwrap_or_else(|| {
            eprintln!("nokozero_hook: d3d9: row_pitch overflow: width {width}");
            abort();
        })
}

fn surface_bytes(pitch: u32, height: u32) -> usize {
    if let Some(bytes) = pitch.checked_mul(max(height, 1)) {
        bytes as usize
    } else {
        eprintln!("nokozero_hook: d3d9: surface size overflow: pitch {pitch}, height {height}");
        abort();
    }
}

/// Returns the `pBits` to be reported by `LockRect`.
///
/// # Safety
///
/// `prect` must be null or point to a readable `RECT`.
unsafe fn locked_bits(
    backing: &Backing,
    pitch: u32,
    format: D3DFORMAT,
    prect: *const RECT,
) -> *mut u8 {
    if prect.is_null() {
        return backing.ptr_at(0, backing.len());
    }

    let rect = unsafe { &*prect };
    let offset = {
        let top = max(rect.top, 0).cast_unsigned() as usize;
        let left = max(rect.left, 0).cast_unsigned() as usize;
        top.saturating_mul(pitch as usize)
            .saturating_add(left.saturating_mul(bytes_per_pixel(format) as usize))
    };
    let span = {
        let rows = max(rect.bottom.saturating_sub(rect.top), 0).cast_unsigned() as usize;
        let cols = max(rect.right.saturating_sub(rect.left), 0).cast_unsigned() as usize;
        match rows {
            0 => 0,
            _ => (rows - 1)
                .saturating_mul(pitch as usize)
                .saturating_add(cols.saturating_mul(bytes_per_pixel(format) as usize)),
        }
    };
    backing.ptr_at(offset, span)
}

/// Returns the back buffer size and format requested by the input `D3DPRESENT_PARAMETERS`, normalizing the struct in place.
///
/// # Safety
///
/// `pp` must be null or valid to read and write.
unsafe fn normalize_pp(pp: *mut D3DPRESENT_PARAMETERS) -> (u32, u32, D3DFORMAT) {
    let (mut width, mut height, mut format) = (640, 480, D3DFMT_X8R8G8B8);
    if !pp.is_null() {
        let pp = unsafe { &mut *pp };
        if pp.BackBufferWidth != 0 && pp.BackBufferHeight != 0 {
            width = pp.BackBufferWidth;
            height = pp.BackBufferHeight;
        }
        if pp.BackBufferFormat != D3DFMT_UNKNOWN {
            format = pp.BackBufferFormat;
        }
        (pp.BackBufferWidth, pp.BackBufferHeight) = (width, height);
        pp.BackBufferFormat = format;
    }
    (width, height, format)
}

fn display_mode() -> D3DDISPLAYMODE {
    D3DDISPLAYMODE {
        Width: 640,
        Height: 480,
        RefreshRate: 60,
        Format: D3DFMT_X8R8G8B8,
    }
}

fn device_caps() -> D3DCAPS9 {
    D3DCAPS9 {
        DeviceType: D3DDEVTYPE_HAL,
        Caps2: (D3DCAPS2_DYNAMICTEXTURES | D3DCAPS2_CANMANAGERESOURCE).cast_unsigned(),
        PresentationIntervals: (D3DPRESENT_INTERVAL_IMMEDIATE | D3DPRESENT_INTERVAL_ONE)
            .cast_unsigned(),
        MaxTextureWidth: 16384,
        MaxTextureHeight: 16384,
        MaxTextureRepeat: 8192,
        MaxAnisotropy: 16,
        MaxSimultaneousTextures: 8,
        MaxTextureBlendStages: 8,
        MaxStreams: 16,
        MaxStreamStride: 65536,
        MaxPrimitiveCount: 0x0055_5555,
        MaxVertexIndex: 0x00FF_FFFF,
        NumSimultaneousRTs: 4,
        VertexShaderVersion: 0xFFFE_0300, // vs_3_0
        PixelShaderVersion: 0xFFFF_0300,  // ps_3_0
        MaxVertexShaderConst: 256,
        PixelShader1xMaxValue: 8.,
        MaxPointSize: 256.,
        ..Default::default()
    }
}

#[implement(IDirect3D9)]
struct FakeD3d9;

impl IDirect3D9_Impl for FakeD3d9_Impl {
    fn RegisterSoftwareDevice(&self, _pinitializefunction: *mut c_void) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn GetAdapterCount(&self) -> u32 {
        1
    }

    fn GetAdapterIdentifier(
        &self,
        _adapter: u32,
        _flags: u32,
        pidentifier: *mut D3DADAPTER_IDENTIFIER9,
    ) -> Result<()> {
        if pidentifier.is_null() {
            return Err(Error::from_hresult(D3DERR_INVALIDCALL));
        }
        let mut id = D3DADAPTER_IDENTIFIER9::default();
        for (dst, &b) in zip(&mut id.Driver, b"nokozero") {
            *dst = b.cast_signed();
        }
        for (dst, &b) in zip(&mut id.Description, b"nokozero null device") {
            *dst = b.cast_signed();
        }
        unsafe { pidentifier.write(id) };
        Ok(())
    }

    fn GetAdapterModeCount(&self, _adapter: u32, _format: D3DFORMAT) -> u32 {
        1
    }

    fn EnumAdapterModes(
        &self,
        _adapter: u32,
        _format: D3DFORMAT,
        _mode: u32,
        pmode: *mut D3DDISPLAYMODE,
    ) -> Result<()> {
        unsafe { write_out(pmode, display_mode()) }
    }

    fn GetAdapterDisplayMode(&self, _adapter: u32, pmode: *mut D3DDISPLAYMODE) -> Result<()> {
        unsafe { write_out(pmode, display_mode()) }
    }

    fn CheckDeviceType(
        &self,
        _adapter: u32,
        _devtype: D3DDEVTYPE,
        _adapterformat: D3DFORMAT,
        _backbufferformat: D3DFORMAT,
        _bwindowed: BOOL,
    ) -> Result<()> {
        Ok(())
    }

    fn CheckDeviceFormat(
        &self,
        _adapter: u32,
        _devicetype: D3DDEVTYPE,
        _adapterformat: D3DFORMAT,
        _usage: u32,
        _rtype: D3DRESOURCETYPE,
        _checkformat: D3DFORMAT,
    ) -> Result<()> {
        Ok(())
    }

    fn CheckDeviceMultiSampleType(
        &self,
        _adapter: u32,
        _devicetype: D3DDEVTYPE,
        _surfaceformat: D3DFORMAT,
        _windowed: BOOL,
        _multisampletype: D3DMULTISAMPLE_TYPE,
        pqualitylevels: *mut u32,
    ) -> Result<()> {
        if !pqualitylevels.is_null() {
            unsafe { pqualitylevels.write(1) };
        }
        Ok(())
    }

    fn CheckDepthStencilMatch(
        &self,
        _adapter: u32,
        _devicetype: D3DDEVTYPE,
        _adapterformat: D3DFORMAT,
        _rendertargetformat: D3DFORMAT,
        _depthstencilformat: D3DFORMAT,
    ) -> Result<()> {
        Ok(())
    }

    fn CheckDeviceFormatConversion(
        &self,
        _adapter: u32,
        _devicetype: D3DDEVTYPE,
        _sourceformat: D3DFORMAT,
        _targetformat: D3DFORMAT,
    ) -> Result<()> {
        Ok(())
    }

    fn GetDeviceCaps(
        &self,
        _adapter: u32,
        _devtype: D3DDEVTYPE,
        pcaps: *mut D3DCAPS9,
    ) -> Result<()> {
        unsafe { write_out(pcaps, device_caps()) }
    }

    fn GetAdapterMonitor(&self, _adapter: u32) -> HMONITOR {
        HMONITOR(null_mut())
    }

    fn CreateDevice(
        &self,
        adapter: u32,
        _devicetype: D3DDEVTYPE,
        hfocuswindow: HWND,
        behaviorflags: u32,
        ppresentationparameters: *mut D3DPRESENT_PARAMETERS,
        ppreturneddeviceinterface: OutRef<'_, IDirect3DDevice9>,
    ) -> Result<()> {
        let (width, height, format) = unsafe { normalize_pp(ppresentationparameters) };

        let parent = self.to_interface();
        let back_buffer_backing = Arc::new(Backing::new(surface_bytes(
            row_pitch(width, format),
            height,
        )));
        let depth_backing = Arc::new(Backing::new(surface_bytes(
            row_pitch(width, D3DFMT_D24S8),
            height,
        )));
        let device = FakeDevice {
            parent,
            adapter,
            focus_window: hfocuswindow,
            behavior_flags: behaviorflags,
            back_buffer: (width, height, format),
            back_buffer_backing,
            depth_backing,
        };
        ppreturneddeviceinterface.write(Some(device.into()))
    }
}

#[implement(IDirect3DDevice9)]
struct FakeDevice {
    parent: IDirect3D9,
    adapter: u32,
    focus_window: HWND,
    behavior_flags: u32,
    /// `(width, height, format)` requested by the game in `CreateDevice`.
    back_buffer: (u32, u32, D3DFORMAT),
    /// Storage for `GetBackBuffer` and `GetRenderTarget`.
    back_buffer_backing: Arc<Backing>,
    /// Storage for `GetDepthStencilSurface`.
    depth_backing: Arc<Backing>,
}

impl IDirect3DDevice9_Impl for FakeDevice_Impl {
    fn TestCooperativeLevel(&self) -> Result<()> {
        Ok(())
    }

    fn GetAvailableTextureMem(&self) -> u32 {
        512 * 1024 * 1024
    }

    fn EvictManagedResources(&self) -> Result<()> {
        Ok(())
    }

    fn GetDirect3D(&self) -> Result<IDirect3D9> {
        Ok(self.parent.clone())
    }

    fn GetDeviceCaps(&self, pcaps: *mut D3DCAPS9) -> Result<()> {
        unsafe { write_out(pcaps, device_caps()) }
    }

    fn GetDisplayMode(&self, _iswapchain: u32, pmode: *mut D3DDISPLAYMODE) -> Result<()> {
        unsafe { write_out(pmode, display_mode()) }
    }

    fn GetCreationParameters(&self, pparameters: *mut D3DDEVICE_CREATION_PARAMETERS) -> Result<()> {
        let params = D3DDEVICE_CREATION_PARAMETERS {
            AdapterOrdinal: self.adapter,
            DeviceType: D3DDEVTYPE_HAL,
            hFocusWindow: self.focus_window,
            BehaviorFlags: self.behavior_flags,
        };
        unsafe { write_out(pparameters, params) }
    }

    fn SetCursorProperties(
        &self,
        _x: u32,
        _y: u32,
        _bmp: Ref<'_, IDirect3DSurface9>,
    ) -> Result<()> {
        Ok(())
    }

    fn SetCursorPosition(&self, _x: i32, _y: i32, _flags: u32) {}

    fn ShowCursor(&self, _bshow: BOOL) -> BOOL {
        BOOL(0)
    }

    fn CreateAdditionalSwapChain(
        &self,
        _pp: *mut D3DPRESENT_PARAMETERS,
        _psc: OutRef<'_, IDirect3DSwapChain9>,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn GetSwapChain(&self, _iswapchain: u32) -> Result<IDirect3DSwapChain9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn GetNumberOfSwapChains(&self) -> u32 {
        1
    }

    fn Reset(&self, ppresentationparameters: *mut D3DPRESENT_PARAMETERS) -> Result<()> {
        let requested = unsafe { normalize_pp(ppresentationparameters) };
        if requested != self.back_buffer {
            let (w, h, f) = self.back_buffer;
            let (rw, rh, rf) = requested;
            eprintln!(
                "nokozero_hook: d3d9: Reset with changed params: have {w}x{h} format {}, asked for {rw}x{rh} format {}",
                f.0, rf.0
            );
            abort();
        }
        Ok(())
    }

    fn Present(
        &self,
        _psourcerect: *const RECT,
        _pdestrect: *const RECT,
        _hdestwindowoverride: HWND,
        _pdirtyregion: *const RGNDATA,
    ) -> Result<()> {
        Ok(())
    }

    fn GetBackBuffer(
        &self,
        _iswapchain: u32,
        _ibackbuffer: u32,
        _type: D3DBACKBUFFER_TYPE,
    ) -> Result<IDirect3DSurface9> {
        let device = self.to_interface();
        let (w, h, fmt) = self.back_buffer;
        Ok(FakeSurface::from_shared(device, self.back_buffer_backing.clone(), w, h, fmt).into())
    }

    fn GetRasterStatus(&self, _iswapchain: u32, _prs: *mut D3DRASTER_STATUS) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetDialogBoxMode(&self, _b: BOOL) -> Result<()> {
        Ok(())
    }

    fn SetGammaRamp(&self, _iswapchain: u32, _flags: u32, _pramp: *const D3DGAMMARAMP) {}

    fn GetGammaRamp(&self, _iswapchain: u32, pramp: *mut D3DGAMMARAMP) {
        drop(unsafe { get_zeroed(pramp) });
    }

    fn CreateTexture(
        &self,
        width: u32,
        height: u32,
        levels: u32,
        usage: u32,
        format: D3DFORMAT,
        pool: D3DPOOL,
        pptexture: OutRef<'_, IDirect3DTexture9>,
        _psharedhandle: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        let tex = FakeTexture::new(device, width, height, levels, usage, format, pool);
        pptexture.write(Some(tex.into()))
    }

    fn CreateVolumeTexture(
        &self,
        _w: u32,
        _h: u32,
        _d: u32,
        _levels: u32,
        _usage: u32,
        _format: D3DFORMAT,
        _pool: D3DPOOL,
        _ppvt: OutRef<'_, IDirect3DVolumeTexture9>,
        _psh: *mut HANDLE,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn CreateCubeTexture(
        &self,
        _edge: u32,
        _levels: u32,
        _usage: u32,
        _format: D3DFORMAT,
        _pool: D3DPOOL,
        _ppct: OutRef<'_, IDirect3DCubeTexture9>,
        _psh: *mut HANDLE,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn CreateVertexBuffer(
        &self,
        length: u32,
        usage: u32,
        fvf: u32,
        pool: D3DPOOL,
        ppvertexbuffer: OutRef<'_, IDirect3DVertexBuffer9>,
        _psharedhandle: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        let vb = FakeVertexBuffer {
            device,
            backing: Backing::new(length as usize),
            length,
            usage,
            fvf,
            pool,
        };
        ppvertexbuffer.write(Some(vb.into()))
    }

    fn CreateIndexBuffer(
        &self,
        length: u32,
        usage: u32,
        format: D3DFORMAT,
        pool: D3DPOOL,
        ppindexbuffer: OutRef<'_, IDirect3DIndexBuffer9>,
        _psharedhandle: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        let ib = FakeIndexBuffer {
            device,
            backing: Backing::new(length as usize),
            length,
            usage,
            format,
            pool,
        };
        ppindexbuffer.write(Some(ib.into()))
    }

    fn CreateRenderTarget(
        &self,
        w: u32,
        h: u32,
        format: D3DFORMAT,
        _ms: D3DMULTISAMPLE_TYPE,
        _msq: u32,
        _lockable: BOOL,
        pps: OutRef<'_, IDirect3DSurface9>,
        _psh: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        pps.write(Some(FakeSurface::new(device, w, h, format).into()))
    }

    fn CreateDepthStencilSurface(
        &self,
        w: u32,
        h: u32,
        format: D3DFORMAT,
        _ms: D3DMULTISAMPLE_TYPE,
        _msq: u32,
        _discard: BOOL,
        pps: OutRef<'_, IDirect3DSurface9>,
        _psh: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        pps.write(Some(FakeSurface::new(device, w, h, format).into()))
    }

    fn UpdateSurface(
        &self,
        _src: Ref<'_, IDirect3DSurface9>,
        _srcrect: *const RECT,
        _dst: Ref<'_, IDirect3DSurface9>,
        _dstpoint: *const POINT,
    ) -> Result<()> {
        Ok(())
    }

    fn UpdateTexture(
        &self,
        _src: Ref<'_, IDirect3DBaseTexture9>,
        _dst: Ref<'_, IDirect3DBaseTexture9>,
    ) -> Result<()> {
        Ok(())
    }

    fn GetRenderTargetData(
        &self,
        _rt: Ref<'_, IDirect3DSurface9>,
        _dst: Ref<'_, IDirect3DSurface9>,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn GetFrontBufferData(&self, _iswapchain: u32, _dst: Ref<'_, IDirect3DSurface9>) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn StretchRect(
        &self,
        _src: Ref<'_, IDirect3DSurface9>,
        _srcrect: *const RECT,
        _dst: Ref<'_, IDirect3DSurface9>,
        _dstrect: *const RECT,
        _filter: D3DTEXTUREFILTERTYPE,
    ) -> Result<()> {
        Ok(())
    }

    fn ColorFill(
        &self,
        _ps: Ref<'_, IDirect3DSurface9>,
        _prect: *const RECT,
        _color: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn CreateOffscreenPlainSurface(
        &self,
        w: u32,
        h: u32,
        format: D3DFORMAT,
        _pool: D3DPOOL,
        pps: OutRef<'_, IDirect3DSurface9>,
        _psh: *mut HANDLE,
    ) -> Result<()> {
        let device = self.to_interface();
        pps.write(Some(FakeSurface::new(device, w, h, format).into()))
    }

    fn SetRenderTarget(&self, _i: u32, _prt: Ref<'_, IDirect3DSurface9>) -> Result<()> {
        Ok(())
    }

    fn GetRenderTarget(&self, _rtindex: u32) -> Result<IDirect3DSurface9> {
        let device = self.to_interface();
        let (w, h, fmt) = self.back_buffer;
        Ok(FakeSurface::from_shared(device, self.back_buffer_backing.clone(), w, h, fmt).into())
    }

    fn SetDepthStencilSurface(&self, _pz: Ref<'_, IDirect3DSurface9>) -> Result<()> {
        Ok(())
    }

    fn GetDepthStencilSurface(&self) -> Result<IDirect3DSurface9> {
        let device = self.to_interface();
        let (w, h, _) = self.back_buffer;
        Ok(FakeSurface::from_shared(device, self.depth_backing.clone(), w, h, D3DFMT_D24S8).into())
    }

    fn BeginScene(&self) -> Result<()> {
        Ok(())
    }

    fn EndScene(&self) -> Result<()> {
        Ok(())
    }

    fn Clear(
        &self,
        _count: u32,
        _prects: *const D3DRECT,
        _flags: u32,
        _color: u32,
        _z: f32,
        _stencil: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn SetTransform(&self, _state: D3DTRANSFORMSTATETYPE, _m: *const Matrix4x4) -> Result<()> {
        Ok(())
    }

    fn GetTransform(&self, _state: D3DTRANSFORMSTATETYPE, pmatrix: *mut Matrix4x4) -> Result<()> {
        #[rustfmt::skip]
        const IDENTITY: Matrix4x4 = Matrix4x4 {
            M11: 1., M12: 0., M13: 0., M14: 0.,
            M21: 0., M22: 1., M23: 0., M24: 0.,
            M31: 0., M32: 0., M33: 1., M34: 0.,
            M41: 0., M42: 0., M43: 0., M44: 1.,
        };

        if !pmatrix.is_null() {
            unsafe { pmatrix.write(IDENTITY) };
        }
        Ok(())
    }

    fn MultiplyTransform(&self, _s: D3DTRANSFORMSTATETYPE, _m: *const Matrix4x4) -> Result<()> {
        Ok(())
    }

    fn SetViewport(&self, _pviewport: *const D3DVIEWPORT9) -> Result<()> {
        Ok(())
    }

    fn GetViewport(&self, pviewport: *mut D3DVIEWPORT9) -> Result<()> {
        unsafe { get_zeroed(pviewport) }
    }

    fn SetMaterial(&self, _pmaterial: *const D3DMATERIAL9) -> Result<()> {
        Ok(())
    }

    fn GetMaterial(&self, pmaterial: *mut D3DMATERIAL9) -> Result<()> {
        unsafe { get_zeroed(pmaterial) }
    }

    fn SetLight(&self, _index: u32, _plight: *const D3DLIGHT9) -> Result<()> {
        Ok(())
    }

    fn GetLight(&self, _index: u32, plight: *mut D3DLIGHT9) -> Result<()> {
        unsafe { get_zeroed(plight) }
    }

    fn LightEnable(&self, _index: u32, _enable: BOOL) -> Result<()> {
        Ok(())
    }

    fn GetLightEnable(&self, _index: u32, penable: *mut BOOL) -> Result<()> {
        unsafe { get_zeroed(penable) }
    }

    fn SetClipPlane(&self, _index: u32, _pplane: *const f32) -> Result<()> {
        Ok(())
    }

    fn GetClipPlane(&self, _index: u32, pplane: *mut f32) -> Result<()> {
        unsafe { get_zeroed_array(pplane, 4) }
    }

    fn SetRenderState(&self, _state: D3DRENDERSTATETYPE, _value: u32) -> Result<()> {
        Ok(())
    }

    fn GetRenderState(&self, _state: D3DRENDERSTATETYPE, pvalue: *mut u32) -> Result<()> {
        unsafe { get_zeroed(pvalue) }
    }

    fn CreateStateBlock(&self, _t: D3DSTATEBLOCKTYPE) -> Result<IDirect3DStateBlock9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn BeginStateBlock(&self) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn EndStateBlock(&self) -> Result<IDirect3DStateBlock9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetClipStatus(&self, _pcs: *const D3DCLIPSTATUS9) -> Result<()> {
        Ok(())
    }

    fn GetClipStatus(&self, pcs: *mut D3DCLIPSTATUS9) -> Result<()> {
        unsafe { get_zeroed(pcs) }
    }

    fn GetTexture(&self, _stage: u32) -> Result<IDirect3DBaseTexture9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetTexture(&self, _stage: u32, _ptexture: Ref<'_, IDirect3DBaseTexture9>) -> Result<()> {
        Ok(())
    }

    fn GetTextureStageState(
        &self,
        _stage: u32,
        _t: D3DTEXTURESTAGESTATETYPE,
        pvalue: *mut u32,
    ) -> Result<()> {
        unsafe { get_zeroed(pvalue) }
    }

    fn SetTextureStageState(
        &self,
        _stage: u32,
        _t: D3DTEXTURESTAGESTATETYPE,
        _value: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn GetSamplerState(
        &self,
        _sampler: u32,
        _t: D3DSAMPLERSTATETYPE,
        pvalue: *mut u32,
    ) -> Result<()> {
        unsafe { get_zeroed(pvalue) }
    }

    fn SetSamplerState(&self, _sampler: u32, _t: D3DSAMPLERSTATETYPE, _value: u32) -> Result<()> {
        Ok(())
    }

    fn ValidateDevice(&self, pnumpasses: *mut u32) -> Result<()> {
        if !pnumpasses.is_null() {
            unsafe { pnumpasses.write(1) };
        }
        Ok(())
    }

    fn SetPaletteEntries(&self, _n: u32, _pe: *const PALETTEENTRY) -> Result<()> {
        Ok(())
    }

    fn GetPaletteEntries(&self, _n: u32, pe: *mut PALETTEENTRY) -> Result<()> {
        unsafe { get_zeroed_array(pe, 256) }
    }

    fn SetCurrentTexturePalette(&self, _n: u32) -> Result<()> {
        Ok(())
    }

    fn GetCurrentTexturePalette(&self, pn: *mut u32) -> Result<()> {
        unsafe { get_zeroed(pn) }
    }

    fn SetScissorRect(&self, _prect: *const RECT) -> Result<()> {
        Ok(())
    }

    fn GetScissorRect(&self, prect: *mut RECT) -> Result<()> {
        unsafe { get_zeroed(prect) }
    }

    fn SetSoftwareVertexProcessing(&self, _b: BOOL) -> Result<()> {
        Ok(())
    }

    fn GetSoftwareVertexProcessing(&self) -> BOOL {
        BOOL(0)
    }

    fn SetNPatchMode(&self, _n: f32) -> Result<()> {
        Ok(())
    }

    fn GetNPatchMode(&self) -> f32 {
        0.
    }

    fn DrawPrimitive(&self, _t: D3DPRIMITIVETYPE, _start: u32, _count: u32) -> Result<()> {
        Ok(())
    }

    fn DrawIndexedPrimitive(
        &self,
        _t: D3DPRIMITIVETYPE,
        _base: i32,
        _min: u32,
        _num: u32,
        _start: u32,
        _count: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn DrawPrimitiveUP(
        &self,
        _t: D3DPRIMITIVETYPE,
        _count: u32,
        _data: *const c_void,
        _stride: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn DrawIndexedPrimitiveUP(
        &self,
        _t: D3DPRIMITIVETYPE,
        _min: u32,
        _num: u32,
        _count: u32,
        _idx: *const c_void,
        _idxfmt: D3DFORMAT,
        _vtx: *const c_void,
        _stride: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn ProcessVertices(
        &self,
        _srcstart: u32,
        _destidx: u32,
        _count: u32,
        _dst: Ref<'_, IDirect3DVertexBuffer9>,
        _decl: Ref<'_, IDirect3DVertexDeclaration9>,
        _flags: u32,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn CreateVertexDeclaration(
        &self,
        _pve: *const D3DVERTEXELEMENT9,
    ) -> Result<IDirect3DVertexDeclaration9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetVertexDeclaration(&self, _pdecl: Ref<'_, IDirect3DVertexDeclaration9>) -> Result<()> {
        Ok(())
    }

    fn GetVertexDeclaration(&self) -> Result<IDirect3DVertexDeclaration9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetFVF(&self, _fvf: u32) -> Result<()> {
        Ok(())
    }

    fn GetFVF(&self, pfvf: *mut u32) -> Result<()> {
        unsafe { get_zeroed(pfvf) }
    }

    fn CreateVertexShader(&self, _pfunction: *const u32) -> Result<IDirect3DVertexShader9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetVertexShader(&self, _pshader: Ref<'_, IDirect3DVertexShader9>) -> Result<()> {
        Ok(())
    }

    fn GetVertexShader(&self) -> Result<IDirect3DVertexShader9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetVertexShaderConstantF(&self, _r: u32, _p: *const f32, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetVertexShaderConstantF(&self, _r: u32, p: *mut f32, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, (c as usize).checked_mul(4).unwrap_or(0)) }
    }

    fn SetVertexShaderConstantI(&self, _r: u32, _p: *const i32, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetVertexShaderConstantI(&self, _r: u32, p: *mut i32, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, (c as usize).checked_mul(4).unwrap_or(0)) }
    }

    fn SetVertexShaderConstantB(&self, _r: u32, _p: *const BOOL, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetVertexShaderConstantB(&self, _r: u32, p: *mut BOOL, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, c as usize) }
    }

    fn SetStreamSource(
        &self,
        _n: u32,
        _data: Ref<'_, IDirect3DVertexBuffer9>,
        _off: u32,
        _stride: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn GetStreamSource(
        &self,
        _n: u32,
        _pp: OutRef<'_, IDirect3DVertexBuffer9>,
        _poff: *mut u32,
        _pstride: *mut u32,
    ) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetStreamSourceFreq(&self, _n: u32, _setting: u32) -> Result<()> {
        Ok(())
    }

    fn GetStreamSourceFreq(&self, _n: u32, psetting: *mut u32) -> Result<()> {
        if !psetting.is_null() {
            unsafe { psetting.write(1) };
        }
        Ok(())
    }

    fn SetIndices(&self, _pindexdata: Ref<'_, IDirect3DIndexBuffer9>) -> Result<()> {
        Ok(())
    }

    fn GetIndices(&self) -> Result<IDirect3DIndexBuffer9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn CreatePixelShader(&self, _pfunction: *const u32) -> Result<IDirect3DPixelShader9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetPixelShader(&self, _pshader: Ref<'_, IDirect3DPixelShader9>) -> Result<()> {
        Ok(())
    }

    fn GetPixelShader(&self) -> Result<IDirect3DPixelShader9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn SetPixelShaderConstantF(&self, _r: u32, _p: *const f32, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetPixelShaderConstantF(&self, _r: u32, p: *mut f32, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, (c as usize).checked_mul(4).unwrap_or(0)) }
    }

    fn SetPixelShaderConstantI(&self, _r: u32, _p: *const i32, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetPixelShaderConstantI(&self, _r: u32, p: *mut i32, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, (c as usize).checked_mul(4).unwrap_or(0)) }
    }

    fn SetPixelShaderConstantB(&self, _r: u32, _p: *const BOOL, _c: u32) -> Result<()> {
        Ok(())
    }

    fn GetPixelShaderConstantB(&self, _r: u32, p: *mut BOOL, c: u32) -> Result<()> {
        unsafe { get_zeroed_array(p, c as usize) }
    }

    fn DrawRectPatch(&self, _h: u32, _pn: *const f32, _pi: *const D3DRECTPATCH_INFO) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn DrawTriPatch(&self, _h: u32, _pn: *const f32, _pi: *const D3DTRIPATCH_INFO) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn DeletePatch(&self, _handle: u32) -> Result<()> {
        Ok(())
    }

    fn CreateQuery(&self, _t: D3DQUERYTYPE) -> Result<IDirect3DQuery9> {
        Err(Error::from_hresult(E_NOTIMPL))
    }
}

macro_rules! resource_impl {
    ($rtype:expr) => {
        fn GetDevice(&self) -> Result<IDirect3DDevice9> {
            Ok(self.device.clone())
        }

        fn SetPrivateData(
            &self,
            _refguid: *const GUID,
            _pdata: *const c_void,
            _sizeofdata: u32,
            _flags: u32,
        ) -> Result<()> {
            Ok(())
        }

        fn GetPrivateData(
            &self,
            _refguid: *const GUID,
            _pdata: *mut c_void,
            _psizeofdata: *mut u32,
        ) -> Result<()> {
            Err(Error::from_hresult(D3DERR_INVALIDCALL))
        }

        fn FreePrivateData(&self, _refguid: *const GUID) -> Result<()> {
            Ok(())
        }

        fn SetPriority(&self, _prioritynew: u32) -> u32 {
            0
        }

        fn GetPriority(&self) -> u32 {
            0
        }

        fn PreLoad(&self) {}

        fn GetType(&self) -> D3DRESOURCETYPE {
            $rtype
        }
    };
}

#[implement(IDirect3DVertexBuffer9)]
struct FakeVertexBuffer {
    device: IDirect3DDevice9,
    backing: Backing,
    length: u32,
    usage: u32,
    fvf: u32,
    pool: D3DPOOL,
}

impl IDirect3DVertexBuffer9_Impl for FakeVertexBuffer_Impl {
    fn Lock(
        &self,
        offsettolock: u32,
        sizetolock: u32,
        ppbdata: *mut *mut c_void,
        _flags: u32,
    ) -> Result<()> {
        unsafe { lock_backing(&self.backing, offsettolock, sizetolock, ppbdata) }
    }

    fn Unlock(&self) -> Result<()> {
        Ok(())
    }

    fn GetDesc(&self, pdesc: *mut D3DVERTEXBUFFER_DESC) -> Result<()> {
        let desc = D3DVERTEXBUFFER_DESC {
            Format: D3DFMT_VERTEXDATA,
            Type: D3DRTYPE_VERTEXBUFFER,
            Usage: self.usage,
            Pool: self.pool,
            Size: self.length,
            FVF: self.fvf,
        };
        unsafe { write_out(pdesc, desc) }
    }
}

impl IDirect3DResource9_Impl for FakeVertexBuffer_Impl {
    resource_impl!(D3DRTYPE_VERTEXBUFFER);
}

#[implement(IDirect3DIndexBuffer9)]
struct FakeIndexBuffer {
    device: IDirect3DDevice9,
    backing: Backing,
    length: u32,
    usage: u32,
    format: D3DFORMAT,
    pool: D3DPOOL,
}

impl IDirect3DIndexBuffer9_Impl for FakeIndexBuffer_Impl {
    fn Lock(
        &self,
        offsettolock: u32,
        sizetolock: u32,
        ppbdata: *mut *mut c_void,
        _flags: u32,
    ) -> Result<()> {
        unsafe { lock_backing(&self.backing, offsettolock, sizetolock, ppbdata) }
    }

    fn Unlock(&self) -> Result<()> {
        Ok(())
    }

    fn GetDesc(&self, pdesc: *mut D3DINDEXBUFFER_DESC) -> Result<()> {
        let desc = D3DINDEXBUFFER_DESC {
            Format: self.format,
            Type: D3DRTYPE_INDEXBUFFER,
            Usage: self.usage,
            Pool: self.pool,
            Size: self.length,
        };
        unsafe { write_out(pdesc, desc) }
    }
}

impl IDirect3DResource9_Impl for FakeIndexBuffer_Impl {
    resource_impl!(D3DRTYPE_INDEXBUFFER);
}

#[implement(IDirect3DSurface9)]
struct FakeSurface {
    device: IDirect3DDevice9,
    backing: Arc<Backing>,
    width: u32,
    height: u32,
    format: D3DFORMAT,
}

impl FakeSurface {
    fn new(device: IDirect3DDevice9, width: u32, height: u32, format: D3DFORMAT) -> Self {
        let backing = Arc::new(Backing::new(surface_bytes(
            row_pitch(width, format),
            height,
        )));
        Self::from_shared(device, backing, width, height, format)
    }

    fn from_shared(
        device: IDirect3DDevice9,
        backing: Arc<Backing>,
        width: u32,
        height: u32,
        format: D3DFORMAT,
    ) -> Self {
        Self {
            device,
            backing,
            width,
            height,
            format,
        }
    }

    fn surface_desc(&self) -> D3DSURFACE_DESC {
        D3DSURFACE_DESC {
            Format: self.format,
            Type: D3DRTYPE_SURFACE,
            Usage: 0,
            Pool: D3DPOOL_DEFAULT,
            MultiSampleType: D3DMULTISAMPLE_NONE,
            MultiSampleQuality: 0,
            Width: self.width,
            Height: self.height,
        }
    }
}

impl IDirect3DSurface9_Impl for FakeSurface_Impl {
    fn GetContainer(&self, _riid: *const GUID, _pp: *mut *mut c_void) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn GetDesc(&self, pdesc: *mut D3DSURFACE_DESC) -> Result<()> {
        unsafe { write_out(pdesc, self.surface_desc()) }
    }

    fn LockRect(
        &self,
        plockedrect: *mut D3DLOCKED_RECT,
        prect: *const RECT,
        _flags: u32,
    ) -> Result<()> {
        let pitch = row_pitch(self.width, self.format);
        let bits = unsafe { locked_bits(&self.backing, pitch, self.format, prect) };
        let locked = D3DLOCKED_RECT {
            Pitch: pitch.cast_signed(),
            pBits: bits.cast(),
        };
        unsafe { write_out(plockedrect, locked) }
    }

    fn UnlockRect(&self) -> Result<()> {
        Ok(())
    }

    fn GetDC(&self, _phdc: *mut HDC) -> Result<()> {
        Err(Error::from_hresult(E_NOTIMPL))
    }

    fn ReleaseDC(&self, _hdc: HDC) -> Result<()> {
        Ok(())
    }
}

impl IDirect3DResource9_Impl for FakeSurface_Impl {
    resource_impl!(D3DRTYPE_SURFACE);
}

struct TexLevel {
    backing: Arc<Backing>,
    width: u32,
    height: u32,
}

#[implement(IDirect3DTexture9)]
struct FakeTexture {
    device: IDirect3DDevice9,
    levels: Vec<TexLevel>,
    usage: u32,
    format: D3DFORMAT,
    pool: D3DPOOL,
}

impl FakeTexture {
    fn new(
        device: IDirect3DDevice9,
        width: u32,
        height: u32,
        levels: u32,
        usage: u32,
        format: D3DFORMAT,
        pool: D3DPOOL,
    ) -> Self {
        let count = if levels == 0 {
            let max_dim = max(max(width, height), 1);
            32 - max_dim.leading_zeros()
        } else {
            levels
        };

        let mut mips = Vec::with_capacity(count as usize);
        let (mut w, mut h) = (max(width, 1), max(height, 1));
        for _ in 0..count {
            mips.push(TexLevel {
                backing: Arc::new(Backing::new(surface_bytes(row_pitch(w, format), h))),
                width: w,
                height: h,
            });
            w = max(w / 2, 1);
            h = max(h / 2, 1);
        }

        Self {
            device,
            levels: mips,
            usage,
            format,
            pool,
        }
    }

    fn level_desc(&self, level: &TexLevel) -> D3DSURFACE_DESC {
        D3DSURFACE_DESC {
            Format: self.format,
            Type: D3DRTYPE_TEXTURE,
            Usage: self.usage,
            Pool: self.pool,
            MultiSampleType: D3DMULTISAMPLE_NONE,
            MultiSampleQuality: 0,
            Width: level.width,
            Height: level.height,
        }
    }
}

impl IDirect3DTexture9_Impl for FakeTexture_Impl {
    fn GetLevelDesc(&self, level: u32, pdesc: *mut D3DSURFACE_DESC) -> Result<()> {
        if let Some(lvl) = self.levels.get(level as usize) {
            unsafe { write_out(pdesc, self.level_desc(lvl)) }
        } else {
            Err(Error::from_hresult(D3DERR_INVALIDCALL))
        }
    }

    fn GetSurfaceLevel(&self, level: u32) -> Result<IDirect3DSurface9> {
        self.levels
            .get(level as usize)
            .ok_or_else(|| Error::from_hresult(D3DERR_INVALIDCALL))
            .map(|lvl| {
                FakeSurface::from_shared(
                    self.device.clone(),
                    lvl.backing.clone(),
                    lvl.width,
                    lvl.height,
                    self.format,
                )
            })
            .map(Into::into)
    }

    fn LockRect(
        &self,
        level: u32,
        plockedrect: *mut D3DLOCKED_RECT,
        prect: *const RECT,
        _flags: u32,
    ) -> Result<()> {
        let Some(lvl) = self.levels.get(level as usize) else {
            return Err(Error::from_hresult(D3DERR_INVALIDCALL));
        };
        let pitch = row_pitch(lvl.width, self.format);
        let bits = unsafe { locked_bits(&lvl.backing, pitch, self.format, prect) };
        let locked = D3DLOCKED_RECT {
            Pitch: pitch.cast_signed(),
            pBits: bits.cast(),
        };
        unsafe { write_out(plockedrect, locked) }
    }

    fn UnlockRect(&self, _level: u32) -> Result<()> {
        Ok(())
    }

    fn AddDirtyRect(&self, _pdirtyrect: *const RECT) -> Result<()> {
        Ok(())
    }
}

impl IDirect3DBaseTexture9_Impl for FakeTexture_Impl {
    fn SetLOD(&self, _lodnew: u32) -> u32 {
        0
    }

    fn GetLOD(&self) -> u32 {
        0
    }

    fn GetLevelCount(&self) -> u32 {
        u32::try_from(self.levels.len()).unwrap_or(1)
    }

    fn SetAutoGenFilterType(&self, _filtertype: D3DTEXTUREFILTERTYPE) -> Result<()> {
        Ok(())
    }

    fn GetAutoGenFilterType(&self) -> D3DTEXTUREFILTERTYPE {
        D3DTEXF_LINEAR
    }

    fn GenerateMipSubLevels(&self) {}
}

impl IDirect3DResource9_Impl for FakeTexture_Impl {
    resource_impl!(D3DRTYPE_TEXTURE);
}
