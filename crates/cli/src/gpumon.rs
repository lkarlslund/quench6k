//! GPU temperature, SM clock, power and clock-event reasons sampled through NVML (loaded at run time)
//! while a benchmark measures, so that results taken on a hot or throttling GPU are recognizable.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const INTERVAL: Duration = Duration::from_millis(200);

/// Clock-event reason bits (nvml.h `nvmlClocksEventReason*`) and their names. Power cap is the normal
/// state of a power-limited GPU under load; the thermal and hardware reasons mean it slowed down.
const REASONS: &[(u64, &str, bool)] = &[
    (0x4, "power_cap", false),
    (0x8, "hw_slowdown", true),
    (0x20, "sw_thermal", true),
    (0x40, "hw_thermal", true),
    (0x80, "hw_power_brake", true),
];

type Device = *mut std::ffi::c_void;

struct Nvml {
    _lib: libloading::Library,
    device: Device,
    temperature: unsafe extern "C" fn(Device, u32, *mut u32) -> i32,
    clock: unsafe extern "C" fn(Device, u32, *mut u32) -> i32,
    power: unsafe extern "C" fn(Device, *mut u32) -> i32,
    reasons: unsafe extern "C" fn(Device, *mut u64) -> i32,
}

// The device handle is an opaque NVML pointer, valid on any thread.
unsafe impl Send for Nvml {}

impl Nvml {
    fn open() -> Result<Self, String> {
        let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
        unsafe {
            let lib = libloading::Library::new(name).map_err(|e| format!("{name}: {e}"))?;
            let sym = |n: &[u8]| -> Result<*const (), String> {
                lib.get::<*const ()>(n).map(|s| *s).map_err(|e| format!("{}: {e}", String::from_utf8_lossy(n)))
            };
            let init: unsafe extern "C" fn() -> i32 = std::mem::transmute(sym(b"nvmlInit_v2\0")?);
            let count: unsafe extern "C" fn(*mut u32) -> i32 = std::mem::transmute(sym(b"nvmlDeviceGetCount_v2\0")?);
            let by_index: unsafe extern "C" fn(u32, *mut Device) -> i32 = std::mem::transmute(sym(b"nvmlDeviceGetHandleByIndex_v2\0")?);
            let reasons = sym(b"nvmlDeviceGetCurrentClocksEventReasons\0").or_else(|_| sym(b"nvmlDeviceGetCurrentClocksThrottleReasons\0"))?;
            let rc = init();
            if rc != 0 {
                return Err(format!("nvmlInit failed ({rc})"));
            }
            let mut n = 0;
            count(&mut n);
            if n != 1 {
                return Err(format!("{n} NVIDIA GPUs; the monitor needs exactly one"));
            }
            let mut device = std::ptr::null_mut();
            let rc = by_index(0, &mut device);
            if rc != 0 {
                return Err(format!("nvmlDeviceGetHandleByIndex failed ({rc})"));
            }
            Ok(Self {
                device,
                temperature: std::mem::transmute(sym(b"nvmlDeviceGetTemperature\0")?),
                clock: std::mem::transmute(sym(b"nvmlDeviceGetClockInfo\0")?),
                power: std::mem::transmute(sym(b"nvmlDeviceGetPowerUsage\0")?),
                reasons: std::mem::transmute(reasons),
                _lib: lib,
            })
        }
    }

    fn sample(&self) -> Sample {
        let (mut t, mut c, mut p, mut r) = (0u32, 0u32, 0u32, 0u64);
        unsafe {
            (self.temperature)(self.device, 0, &mut t); // NVML_TEMPERATURE_GPU
            (self.clock)(self.device, 1, &mut c); // NVML_CLOCK_SM
            (self.power)(self.device, &mut p);
            (self.reasons)(self.device, &mut r);
        }
        Sample { temp: t, sm_mhz: c, power_mw: p, reasons: r }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    temp: u32,
    sm_mhz: u32,
    power_mw: u32,
    reasons: u64,
}

/// What the GPU did during one measurement window.
#[derive(Clone, Debug, Default)]
pub struct Window {
    pub samples: usize,
    pub temp_max: u32,
    pub sm_mhz_min: u32,
    pub sm_mhz_mean: f64,
    pub power_w_mean: f64,
    pub reasons: u64,
}

impl Window {
    /// The window saw a thermal or hardware slowdown.
    pub fn slowed(&self) -> bool {
        REASONS.iter().any(|&(bit, _, bad)| bad && self.reasons & bit != 0)
    }

    pub fn reason_names(&self) -> Vec<&'static str> {
        REASONS.iter().filter(|&&(bit, _, _)| self.reasons & bit != 0).map(|&(_, n, _)| n).collect()
    }

    pub fn summary(&self) -> String {
        format!("GPU max {} C, SM {}-{:.0} MHz (min-mean), {:.0} W", self.temp_max, self.sm_mhz_min, self.sm_mhz_mean, self.power_w_mean)
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"samples": self.samples, "temp_max_c": self.temp_max, "sm_mhz_min": self.sm_mhz_min, "sm_mhz_mean": self.sm_mhz_mean.round(),
                           "power_w_mean": self.power_w_mean.round(), "clock_events": self.reason_names(), "slowed": self.slowed()})
    }
}

#[derive(Default)]
struct Acc {
    n: usize,
    temp_max: u32,
    sm_min: u32,
    sm_sum: f64,
    power_sum: f64,
    reasons: u64,
}

/// A sampling thread; `begin` starts a window and `end` returns it.
pub struct Monitor {
    acc: Arc<Mutex<Acc>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Monitor {
    /// None (after saying why) when NVML is unavailable; benchmarks then run without GPU records.
    pub fn start() -> Option<Self> {
        let nvml = match Nvml::open() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("GPU monitor unavailable: {e}");
                return None;
            }
        };
        let acc = Arc::new(Mutex::new(Acc::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (a, s) = (acc.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let x = nvml.sample();
                let mut g = a.lock().unwrap();
                g.sm_min = if g.n == 0 { x.sm_mhz } else { g.sm_min.min(x.sm_mhz) };
                g.n += 1;
                g.temp_max = g.temp_max.max(x.temp);
                g.sm_sum += x.sm_mhz as f64;
                g.power_sum += x.power_mw as f64 / 1000.0;
                g.reasons |= x.reasons;
                drop(g);
                std::thread::sleep(INTERVAL);
            }
        });
        Some(Self { acc, stop, thread: Some(thread) })
    }

    pub fn begin(&self) {
        *self.acc.lock().unwrap() = Acc::default();
    }

    pub fn end(&self) -> Window {
        let g = self.acc.lock().unwrap();
        let n = g.n.max(1) as f64;
        Window { samples: g.n, temp_max: g.temp_max, sm_mhz_min: g.sm_min, sm_mhz_mean: g.sm_sum / n, power_w_mean: g.power_sum / n, reasons: g.reasons }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
