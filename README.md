# GenettaAudioStreaming

GenettaAudioStreaming is a small Rust library (`cdylib`) that receives G.711 (PCMU/PCMA) audio over RTP via UDP, performs jitter buffering and a DC‑blocking high‑pass filter, optionally upsamples from 8 kHz to 16 kHz, and delivers decoded PCM samples to a user‑provided callback through a C ABI.

This crate is designed to be embedded into applications written in C/C++/Rust (via FFI) that need to consume low‑latency RTP telephony audio.

## Stack
- Language: Rust (Edition 2021)
- Crate type: `cdylib` (C‑compatible dynamic library)
- Package manager/build: Cargo
- Dependencies (from `Cargo.toml`):
  - `once_cell`
  - `bytes`
  - `tokio` (present but not used directly in current `src/lib.rs`; kept for potential future async I/O) — TODO: confirm necessity
  - `thread-priority` (optional; enabled by `realtime` feature)
- Optional features:
  - `resample16k`: enable 2× linear upsampling (8 kHz -> 16 kHz) before invoking the PCM callback
  - `realtime`: attempt to raise thread priority (Linux/Windows; no‑op on macOS)

## Overview
- Binds a UDP socket to `0.0.0.0:<bind_port>`
- Parses RTP, accepting payload types 0 (PCMU/µ‑law) and 8 (PCMA/A‑law)
- Decodes to signed 16‑bit PCM (`i16`), mono
- Jitter buffer with target ~40 ms latency at 8 kHz
- DC‑block filter applied to reduce low‑frequency/DC bias
- Optional 2× linear upsampling to 16 kHz if `resample16k` feature is enabled
- Deliver samples to a user‑registered callback

## Public C ABI (entry points)
Exported by the library (see `src/lib.rs`):

```c
// Create an RTP receiver bound to the given UDP port; returns instance ID (>0) or 0 on error.
extern unsigned long long gas_create_rtp_receiver(unsigned short bind_port);

// Register a PCM callback; called from the receiver thread when decoded audio is ready.
// pcm: pointer to interleaved i16 samples (mono)
// samples: number of samples in the buffer
// sample_rate: 8000 or 16000 depending on features
// channels: 1
typedef void (*PcmCb)(unsigned long long inst,
                      const short* pcm,
                      size_t samples,
                      unsigned int sample_rate,
                      unsigned char channels);
extern void gas_on_pcm(unsigned long long inst, PcmCb cb);

// Request the receiver thread to stop.
extern void gas_stop(unsigned long long inst);

// Stop (if needed) and destroy the instance.
extern void gas_destroy(unsigned long long inst);

// Simple version helpers (not semantic versioning; for linkers/tools).
extern int gas_version_major(void); // returns 1
extern int gas_version_minor(void); // returns 0
```

Notes:
- The callback is invoked on the internal receiver thread; avoid heavy work in the callback. Copy or enqueue data and return quickly.
- The library decodes only payload type 0 (PCMU) and type 8 (PCMA).
- When `resample16k` is disabled, `sample_rate` is 8000; when enabled, `sample_rate` is 16000.
- Mono only (`channels == 1`).

## Requirements
- Rust toolchain with Cargo (stable recommended)
- Supported OS: Windows and Linux (tested paths in code); macOS builds but `realtime` feature is a no‑op
- Networking: ability to bind a UDP port and receive RTP packets
- C toolchain if linking from C/C++ (to link against the produced dynamic library)

TODOs:
- Specify minimum supported Rust version (MSRV)
- Confirm actual tested platforms/CI matrix

## Build
Build the dynamic library:

```bash
# Release build
cargo build --release

# With optional features
cargo build --release --features resample16k
cargo build --release --features realtime
cargo build --release --features "resample16k realtime"
```

Artifacts will be available under `target/release/` as a platform‑specific dynamic library (e.g., `.dll` on Windows, `.so` on Linux, `.dylib` on macOS) with the crate name `GenettaAudioStreaming`.

## Using from C/C++
- Include declarations for the exported functions (see the C signatures above).
- Load/link the produced library and call `gas_create_rtp_receiver(port)`, then `gas_on_pcm(inst, cb)` to receive samples.
- When finished, call `gas_stop(inst)` and `gas_destroy(inst)`.

Safety considerations:
- The callback pointer must remain valid for the lifetime of the instance.
- Callbacks are executed from a background thread; ensure thread safety in your code.
- The buffer pointer passed to the callback is only valid during the call; copy data if it must outlive the call.

## Running (Rust consumers)
Although this crate is a `cdylib`, you can still depend on it from Rust and call the exported `extern "C"` functions via FFI. Example crates/binaries are not provided in this repository.

TODO:
- Provide a small example program (Rust or C) demonstrating setup and packet feeding

## Environment variables
No environment variables are currently used by the library.

TODO:
- Document any future configuration knobs (e.g., adjustable jitter buffer depth) if added

## Tests
There are no tests in this repository.

TODO:
- Add unit tests for the G.711 tables and decoders
- Add integration tests for RTP parsing and jitter buffer behavior

## Project structure
```
.
├─ Cargo.toml          # Crate metadata, features, dependencies
├─ Cargo.lock          # Lockfile
├─ src
│  └─ lib.rs           # Library implementation, C ABI, RTP decode, jitter buffer, filters
└─ target/             # Build artifacts (generated)
```

## Scripts & common Cargo commands
- Build: `cargo build` / `cargo build --release`
- Lints (if you use Clippy): `cargo clippy` — TODO: set up if desired
- Docs: `cargo doc --open`
- Tests: `cargo test` (none present yet)

## Configuration & features
- `resample16k`: doubles sample rate by linear interpolation; callback receives 16 kHz PCM
- `realtime`: attempts to elevate thread priority on Linux/Windows using `thread-priority`

## License
No license file found.

TODO:
- Add a LICENSE file and state the license here (e.g., MIT/Apache‑2.0, proprietary, etc.)
