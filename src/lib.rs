use once_cell::sync::Lazy;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    net::UdpSocket,
    os::raw::{c_int, c_uchar, c_uint, c_ulonglong},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    thread,
    time::Duration,
};
#[cfg(all(feature = "realtime", any(target_os = "linux", target_os = "windows")))]
use thread_priority::*;
type PcmCb = extern "C" fn(
    inst: c_ulonglong,
    pcm: *const i16,
    samples: usize,
    sample_rate: c_uint,
    channels: c_uchar,
);

#[derive(Clone)]
struct Instance {
    cb: Arc<RwLock<Option<PcmCb>>>,
    stop: Arc<AtomicBool>,
}

static INSTANCES: Lazy<RwLock<HashMap<u64, Instance>>> = Lazy::new(|| RwLock::new(HashMap::new()));
static NEXT_ID: Lazy<RwLock<u64>> = Lazy::new(|| RwLock::new(1));

#[no_mangle]
pub extern "C" fn gas_create_rtp_receiver(bind_port: u16) -> c_ulonglong {
    let id = {
        let mut n = NEXT_ID.write().unwrap();
        let v = *n;
        *n += 1;
        v
    };

    let cb_holder: Arc<RwLock<Option<PcmCb>>> = Arc::new(RwLock::new(None));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let cb_clone = cb_holder.clone();
    let stop_clone = stop_flag.clone();

    thread::spawn(move || {
        // Optional RT priority
        #[cfg(all(feature = "realtime", any(target_os = "linux", target_os = "windows")))]
        {
            let _ = set_realtime_priority();
        }

        let addr = format!("0.0.0.0:{}", bind_port);
        let sock = UdpSocket::bind(addr).expect("bind rtp");
        let _ = sock.set_read_timeout(Some(Duration::from_millis(20))); // tight poll

        let mut buf = [0u8; 2048];

        // Jitter buffer config: target 40 ms depth. At 8 kHz and 20 ms/packet: ~2 packets.
        const TARGET_LATENCY_MS: u32 = 40;
        const SAMPLE_RATE_IN: u32 = 8000;
        const CHANNELS: u8 = 1;

        let mut jb = JitterBuffer::new(TARGET_LATENCY_MS, SAMPLE_RATE_IN);

        // DC-blocker state
        let mut dc = DcBlock::new(0.995f32);

        // Working buffers
        let mut decoded: Vec<i16> = Vec::with_capacity(320);
        let mut hpf_buf: Vec<i16> = Vec::with_capacity(320);
        let mut out_buf: Vec<i16> = Vec::with_capacity(640);

        loop {
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }

            match sock.recv(&mut buf) {
                Ok(len) if len >= 12 => {
                    if let Some(pkt) = parse_rtp(&buf[..len]) {
                        if pkt.payload_type == 0 || pkt.payload_type == 8 {
                            decoded.clear();
                            // Decode G.711
                            if pkt.payload_type == 0 {
                                // PCMU
                                decoded.extend(pkt.payload.iter().map(|&b| ULawTable[b as usize]));
                            } else {
                                // PCMA
                                decoded.extend(pkt.payload.iter().map(|&b| ALawTable[b as usize]));
                            }

                            jb.push(pkt.sequence_number, pkt.timestamp, decoded.as_slice());

                            // Pop frames in order according to target latency
                            while let Some(frame) = jb.pop_ready() {
                                // DC block
                                hpf_buf.clear();
                                hpf_buf.extend(frame.iter().map(|&s| dc.process_sample(s)));

                                // Optional upsample to 16k
                                #[cfg(feature = "resample16k")]
                                {
                                    out_buf.clear();
                                    upsample_linear_2x_i16(&hpf_buf, &mut out_buf);
                                    if let Some(cb) = *cb_clone.read().unwrap() {
                                        cb(
                                            id as c_ulonglong,
                                            out_buf.as_ptr(),
                                            out_buf.len(),
                                            16000,
                                            CHANNELS,
                                        );
                                    }
                                }

                                #[cfg(not(feature = "resample16k"))]
                                {
                                    if let Some(cb) = *cb_clone.read().unwrap() {
                                        cb(
                                            id as c_ulonglong,
                                            hpf_buf.as_ptr(),
                                            hpf_buf.len(),
                                            SAMPLE_RATE_IN,
                                            CHANNELS,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Timeout or short read. Also try to drain jitter buffer to keep latency bounded.
                    while let Some(frame) = jb.pop_ready() {
                        hpf_buf.clear();
                        hpf_buf.extend(frame.iter().map(|&s| dc.process_sample(s)));

                        #[cfg(feature = "resample16k")]
                        {
                            out_buf.clear();
                            upsample_linear_2x_i16(&hpf_buf, &mut out_buf);
                            if let Some(cb) = *cb_clone.read().unwrap() {
                                cb(id as c_ulonglong, out_buf.as_ptr(), out_buf.len(), 16000, 1);
                            }
                        }
                        #[cfg(not(feature = "resample16k"))]
                        {
                            if let Some(cb) = *cb_clone.read().unwrap() {
                                cb(id as c_ulonglong, hpf_buf.as_ptr(), hpf_buf.len(), 8000, 1);
                            }
                        }
                    }
                }
            }
        }
    });

    let inst = Instance {
        cb: cb_holder,
        stop: stop_flag,
    };
    INSTANCES.write().unwrap().insert(id, inst);
    id as c_ulonglong
}

#[no_mangle]
pub extern "C" fn gas_on_pcm(inst: c_ulonglong, cb: PcmCb) {
    if let Some(i) = INSTANCES.read().unwrap().get(&(inst as u64)) {
        *i.cb.write().unwrap() = Some(cb);
    }
}

#[no_mangle]
pub extern "C" fn gas_stop(inst: c_ulonglong) {
    if let Some(i) = INSTANCES.read().unwrap().get(&(inst as u64)) {
        i.stop.store(true, Ordering::Relaxed);
    }
}

#[no_mangle]
pub extern "C" fn gas_destroy(inst: c_ulonglong) {
    gas_stop(inst);
    INSTANCES.write().unwrap().remove(&(inst as u64));
}

/* ---------------- RTP, jitter buffer, filters ---------------- */

struct RtpPacket<'a> {
    payload_type: u8,
    sequence_number: u16,
    timestamp: u32,
    payload: &'a [u8],
}

fn parse_rtp(frame: &[u8]) -> Option<RtpPacket<'_>> {
    if frame.len() < 12 {
        return None;
    }
    let v = frame[0] >> 6;
    if v != 2 {
        return None;
    }
    let cc = (frame[0] & 0x0F) as usize;
    let x = (frame[0] & 0x10) != 0;
    let pt = frame[1] & 0x7F;
    let seq = u16::from_be_bytes([frame[2], frame[3]]);
    let ts = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]);
    let mut header_len = 12 + cc * 4;

    if header_len > frame.len() {
        return None;
    }
    if x {
        // extension header
        if header_len + 4 > frame.len() {
            return None;
        }
        let ext_len_words = u16::from_be_bytes([frame[header_len + 2], frame[header_len + 3]]) as usize;
        header_len += 4 + ext_len_words * 4;
        if header_len > frame.len() {
            return None;
        }
    }
    let payload = &frame[header_len..];
    Some(RtpPacket {
        payload_type: pt,
        sequence_number: seq,
        timestamp: ts,
        payload,
    })
}

struct JitterBuffer {
    target_latency_samples: u32,
    rate: u32,
    base_ts: Option<u32>,
    expected_seq: Option<u16>,
    // Map seq -> samples
    queue: BTreeMap<u16, Vec<i16>>,
    // Running availability, converts timestamp gap to number of frames to emit
    pending: VecDeque<Vec<i16>>,
}

impl JitterBuffer {
    fn new(target_ms: u32, rate: u32) -> Self {
        Self {
            target_latency_samples: (target_ms as u64 * rate as u64 / 1000) as u32,
            rate,
            base_ts: None,
            expected_seq: None,
            queue: BTreeMap::new(),
            pending: VecDeque::new(),
        }
    }

    fn push(&mut self, seq: u16, ts: u32, samples: &[i16]) {
        // Initialize base timestamp and expected seq
        if self.base_ts.is_none() {
            self.base_ts = Some(ts);
            self.expected_seq = Some(seq);
        }

        // Store decoded frame
        self.queue.insert(seq, samples.to_vec());

        // Try to reorder and build contiguous frames
        let mut cursor = self.expected_seq.unwrap_or(seq);
        while let Some(frame) = self.queue.remove(&cursor) {
            self.pending.push_back(frame);
            cursor = cursor.wrapping_add(1);
            self.expected_seq = Some(cursor);
        }

        // Enforce latency target: keep enough audio buffered
        // Convert queued samples into sample count
        let queued_samples: usize = self.pending.iter().map(|v| v.len()).sum();
        if queued_samples > self.target_latency_samples as usize {
            // ready to emit at least one frame
            // no-op here; pop_ready will provide frames
        }
    }

    fn pop_ready(&mut self) -> Option<Vec<i16>> {
        let queued_samples: usize = self.pending.iter().map(|v| v.len()).sum();
        if queued_samples >= self.target_latency_samples as usize {
            return self.pending.pop_front();
        }
        None
    }
}

struct DcBlock {
    a: f32,
    x1: f32,
    y1: f32,
}

impl DcBlock {
    fn new(a: f32) -> Self {
        Self { a, x1: 0.0, y1: 0.0 }
    }
    #[inline]
    fn process_sample(&mut self, s: i16) -> i16 {
        // y[n] = x[n] - x[n-1] + a * y[n-1]
        let x = s as f32;
        let y = x - self.x1 + self.a * self.y1;
        self.x1 = x;
        self.y1 = y;
        y.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}

// 2x linear upsampling: 8k -> 16k
fn upsample_linear_2x_i16(input: &[i16], out: &mut Vec<i16>) {
    out.reserve(input.len() * 2);
    for w in input.windows(2) {
        let a = w[0] as i32;
        let b = w[1] as i32;
        out.push(a as i16);
        let mid = ((a + b) / 2) as i16;
        out.push(mid);
    }
    if let Some(&last) = input.last() {
        out.push(last);
        out.push(last);
    }
}

/* ---------------- Precomputed G.711 tables ---------------- */

static ULawTable: Lazy<[i16; 256]> = Lazy::new(|| {
    let mut t = [0i16; 256];
    for i in 0..256 {
        t[i] = ulaw_to_i16(i as u8);
    }
    t
});

static ALawTable: Lazy<[i16; 256]> = Lazy::new(|| {
    let mut t = [0i16; 256];
    for i in 0..256 {
        t[i] = alaw_to_i16(i as u8);
    }
    t
});

// ---- G.711 decode ----
#[inline]
fn ulaw_to_i16(u: u8) -> i16 {
    const BIAS: i32 = 0x84;
    let u = !(u as i32);
    let t = (((u & 0x0F) << 3) + BIAS) << ((u & 0x70) >> 4);
    let s = if (u & 0x80) != 0 { (BIAS - 1) - t } else { t - (BIAS - 1) };
    s.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[inline]
fn alaw_to_i16(a: u8) -> i16 {
    let inverted = a ^ 0x55;
    let mantissa = (inverted & 0x0f) as i32;
    let segment = ((inverted & 0x70) >> 4) as i32;

    let value = if segment == 0 {
        ((mantissa << 4) + 8) as i32
    } else {
        (((mantissa + 16) << (segment + 3)) - 2048) as i32
    };

    let result = if (inverted & 0x80) != 0 { -value } else { value };
    result as i16
}

/* ---------------- Optional realtime priority ---------------- */

#[cfg(all(feature = "realtime", any(target_os = "linux", target_os = "windows")))]
fn set_realtime_priority() -> Result<(), ()> {
    use thread_priority::*;
    // Best-effort. Ignore errors to stay portable.
    let prio = ThreadPriority::Max;
    let policy = ThreadSchedulePolicy::Realtime(RealtimeThreadSchedulePolicy::Fifo);
    set_current_thread_priority(prio).ok();
    set_current_thread_schedule_policy(policy).ok();
    Ok(())
}

#[cfg(all(feature = "realtime", target_os = "macos"))]
fn set_realtime_priority() -> Result<(), ()> {
    // macOS requires Mach APIs; skip to avoid extra deps. No-op.
    Ok(())
}

#[cfg(not(feature = "realtime"))]
fn set_realtime_priority() -> Result<(), ()> { Ok(()) }

/* ---------------- C ABI stubs required by some linkers ---------------- */
#[no_mangle]
pub extern "C" fn gas_version_major() -> c_int { 1 }
#[no_mangle]
pub extern "C" fn gas_version_minor() -> c_int { 0 }