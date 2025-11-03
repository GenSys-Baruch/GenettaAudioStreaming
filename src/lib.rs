use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    ffi::c_void,
    net::UdpSocket,
    os::raw::{c_int, c_uchar, c_uint, c_ulonglong},
    ptr,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

type PcmCb = extern "C" fn(inst: c_ulonglong, pcm: *const i16, samples: usize, sample_rate: c_uint, channels: c_uchar);

#[derive(Clone)]
struct Instance {
    cb: Arc<Mutex<Option<PcmCb>>>,
    stop: Arc<Mutex<bool>>,
}

static INSTANCES: Lazy<Mutex<HashMap<u64, Instance>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(1));

#[no_mangle]
pub extern "C" fn gas_create_rtp_receiver(bind_port: u16) -> c_ulonglong {
    let id = {
        let mut n = NEXT_ID.lock().unwrap();
        let v = *n;
        *n += 1;
        v
    };

    let cb_holder: Arc<Mutex<Option<PcmCb>>> = Arc::new(Mutex::new(None));
    let stop_flag = Arc::new(Mutex::new(false));

    // Spawn a thread with a blocking UDP socket (simpler than async for MVP)
    let cb_clone = cb_holder.clone();
    let stop_clone = stop_flag.clone();

    thread::spawn(move || {
        let addr = format!("0.0.0.0:{}", bind_port);
        let sock = UdpSocket::bind(addr).expect("bind rtp");
        sock.set_read_timeout(Some(Duration::from_millis(500))).ok();

        let mut buf = [0u8; 2048];
        let mut pcm_out: Vec<i16> = Vec::with_capacity(160);

        loop {
            // stop check
            if *stop_clone.lock().unwrap() { break; }

            match sock.recv(&mut buf) {
                Ok(len) if len >= 12 => {
                    // Minimal RTP parse
                    let payload_type = buf[1] & 0x7F;
                    let header_len = 12 + (((buf[0] & 0x0F) as usize) * 4); // handle CSRC count
                    if header_len > len { continue; }
                    let payload = &buf[header_len..len];

                    // Only PT 0 (PCMU) or PT 8 (PCMA)
                    if payload_type == 0 || payload_type == 8 {
                        pcm_out.clear();
                        pcm_out.reserve(payload.len());
                        if payload_type == 0 {
                            for &b in payload { pcm_out.push(ulaw_to_i16(b)); }
                        } else {
                            for &b in payload { pcm_out.push(alaw_to_i16(b)); }
                        }
                        if let Some(cb) = *cb_clone.lock().unwrap() {
                            cb(id, pcm_out.as_ptr(), pcm_out.len(), 8000, 1);
                        }
                    }
                }
                _ => { /* timeout or short; loop */ }
            }
        }
    });

    let inst = Instance { cb: cb_holder, stop: stop_flag };
    INSTANCES.lock().unwrap().insert(id, inst);
    id as c_ulonglong
}

#[no_mangle]
pub extern "C" fn gas_on_pcm(inst: c_ulonglong, cb: PcmCb) {
    if let Some(i) = INSTANCES.lock().unwrap().get(&(inst as u64)) {
        *i.cb.lock().unwrap() = Some(cb);
    }
}

#[no_mangle]
pub extern "C" fn gas_stop(inst: c_ulonglong) {
    if let Some(i) = INSTANCES.lock().unwrap().get(&(inst as u64)) {
        *i.stop.lock().unwrap() = true;
    }
}

#[no_mangle]
pub extern "C" fn gas_destroy(inst: c_ulonglong) {
    gas_stop(inst);
    INSTANCES.lock().unwrap().remove(&(inst as u64));
}

// ---- G.711 decode ----
// μ-law and A-law tables derived from public-domain algorithms.
#[inline]
fn ulaw_to_i16(u: u8) -> i16 {
    // ITU G.711 µ-law decode
    const BIAS: i32 = 0x84;
    let u = !u as i32;
    let t = (((u & 0x0F) << 3) + BIAS) << ((u & 0x70) >> 4);
    let s = if (u & 0x80) != 0 { (BIAS - 1) - t } else { t - (BIAS - 1) };
    s.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[inline]
fn alaw_to_i16(a: u8) -> i16 {
    let inverted = a ^ 0x55;
    let mantissa = (inverted & 0x0f) as i32;
    let segment = ((inverted & 0x70) >> 4) as i32;
    
    // Use i64 to prevent overflow during calculation
    let value = if segment == 0 {
        ((mantissa << 4) + 8) as i64
    } else {
        (((mantissa + 16) << (segment + 3)) - 2048) as i64
    };
    
    // Fix sign logic: negate when sign bit is set
    let result = if (inverted & 0x80) != 0 {
        -value
    } else {
        value
    };
    
    result as i16
}