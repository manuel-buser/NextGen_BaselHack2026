# Camera install guide (XIAO ESP32S3 Sense, `esp32cam_ai_blur`)

How to build and flash the camera firmware on Linux with a minimal setup: a Python venv with the
PlatformIO CLI only (no IDE, no Docker). PlatformIO downloads the ESP-IDF toolchain by itself.

## Which firmware?

| | `esp32cam_ai_blur` (use this) | `esp32cam_snapshot` |
|---|---|---|
| Framework | ESP-IDF + ESP-DL | Arduino |
| Privacy | AI person detector blurs head & shoulders, colour finder blurs yellow/white licence plates | Pixelates anything that moves |
| Exposure auto-tuning | Yes (adapts to backlight / bright floors) | No |
| Live view page | Yes | No |
| Toolchain download | ~2 GB | ~400 MB |

## 1. Prerequisites

- Linux, about 3 GB free disk space
- A Python with `venv` support (>= 3.8). If `python3 -m venv` fails with "ensurepip is not
  available", either `sudo apt install python3-venv` or use another Python (e.g. Anaconda's).
- Serial port access without sudo: your user must be in the `dialout` group
  (`groups` to check; `sudo usermod -aG dialout $USER`, then log out and in).
- The board plugged in over USB-C. It shows up as `303a:1001 Espressif USB JTAG/serial debug unit`
  (`lsusb`) and as `/dev/ttyACM0`.

## 2. Install PlatformIO in a venv

From the `camera/` folder:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install platformio
.venv/bin/pio --version
```

`.venv/` is gitignored.

## 3. Configure the camera

```bash
cd esp32cam_ai_blur
cp src/config.example.h src/config.h
```

Edit `src/config.h` (gitignored, so passwords stay local):

- `WIFI_NETWORKS`: 2.4 GHz networks, tried in order: `{"ssid", "password"}`.
  Remove networks you don't use; each unreachable one delays the connect.
- `CAMERA_HOSTNAME`: the name shown in a phone hotspot's device list.

That's all: the camera doesn't need to know the server. The server fetches pictures from it.

## 4. Build

```bash
../.venv/bin/pio run -j 2
../.venv/bin/pio run -j 2
```

Run it **twice** the first time: the first run downloads the toolchain and the pedestrian model and
stops with `gen_model: model not downloaded yet - build again after this run`. The second run
succeeds (about 2.6 MB of the 7 MB app partition).

## 5. Flash

`platformio.ini` says `COM8` (Windows). On Linux, pass the port on the command line:

```bash
../.venv/bin/pio run -t upload --upload-port /dev/ttyACM0
```

Success ends with `Hash of data verified.` and `[SUCCESS]`.

**"Could not exclusively lock port /dev/ttyACM0"**: another program has the port open (a serial
monitor, the Arduino IDE, another terminal). Close it and flash again.

After changing `config.h` or the code, just run the upload command again; it rebuilds what changed.

## 6. Check it runs

Serial monitor (needs an interactive terminal; Ctrl+C to quit):

```bash
../.venv/bin/pio device monitor --port /dev/ttyACM0
```

Look for:

```
bikeslot: [1834 ms] Wi-Fi connected to <ssid>, IP 192.168.x.y   <- time since boot
bikeslot: Saved <ssid> (BSSID ..., channel 5) for fast connect   <- only when the AP changed
bikeslot: [<ssid>] live view http://192.168.x.y/  picture http://192.168.x.y/snapshot
bikeslot: person 0.41 at [...]          <- a person was detected and blurred
bikeslot: Auto-tune: median ...         <- exposure auto-tuning adjusted (when lighting changes)
```

### Wi-Fi connection

Wi-Fi starts first and connects in the background while the camera starts. After a successful
connect, the camera remembers that network's access point and channel (in NVS), and the next boot
connects to it directly without scanning, whatever the order in `config.h`. If that access point is
gone, the camera forgets it and scans all configured networks (2 attempts each until the first
connect, 4 after a later drop).

For range: a scan covers every channel and picks the **strongest** access point with that name
(memox has several). An access point weaker than -70 dBm (`WIFI_CACHE_MIN_RSSI`) is not
remembered, so the next boot scans for a better one. The link uses a 20 MHz channel, which holds
up better at long range than 40 MHz. Every 15 s the log shows the current signal:
`[memox.world] RSSI -64 dBm ch 5 | live view ...`. Around -70 dBm or better is fine; below
-80 dBm, move the camera or improve the antenna.

Problems show up in the log with the reason, e.g. `reason 201 (network not found)` or
`reason 15 (handshake timeout (wrong password?))`. If no network works, the camera lists the
networks it can see with channel and signal strength, then tries again.

## 7. Using the camera

Your computer must be on the same Wi-Fi as the camera. Replace the IP with the one from the log.

| URL | What it does |
|---|---|
| `http://<camera-ip>/` | Live view for aiming the camera (about 1 picture per second) |
| `http://<camera-ip>/snapshot` | One JPEG, people and plates already blurred |
| `http://<camera-ip>/status` | Camera image settings (JSON) |
| `http://<camera-ip>/health` | Health check: uptime, temperature, snapshot counters, heap/PSRAM, Wi-Fi signal (JSON) |
| `http://<camera-ip>/control` | Change settings at runtime |

```bash
curl -o img.jpg http://<camera-ip>/snapshot
```

### How pictures reach the server

The ESP32 only takes, auto-tunes and blurs the picture. Two ways to get it out:

- **Pull (same Wi-Fi):** the server in `code/` fetches `http://<camera-ip>/snapshot` on a timer
  (`PULL_CAMERAS` in `code/bikeslot/config.py`). Works when the laptop is on the camera's network.
- **Push via relay (any distance, e.g. phone hotspot):** set `RELAY_URL` and `UPLOAD_TOKEN` in
  `config.h`. The camera then posts its latest picture every `SNAPSHOT_INTERVAL_S` and its health
  every `HEALTH_INTERVAL_S` over HTTPS to the relay in the cloud (`relay/`, see
  `relay/README.md`). Anyone with the view key opens `https://<relay>/?key=...`. Log line per
  upload: `Snapshot -> relay: 46 KB, 1 people + 0 plates blurred (240 ms)`. `/health` shows
  `pushes_ok`, `pushes_failed` and `last_push_ms`.

For a phone hotspot: set the hotspot to **2.4 GHz** (the ESP32 can't see 5 GHz), turn off its
auto-off timeout, exempt it from battery saver, and put it first in `WIFI_NETWORKS`.

Note: `code/app/bikeslot/state.py` and `code/app/tools/fake_node.py` still use `/capture`; change
them to `/snapshot` if you use that app.

## Quick reference

```bash
cd camera/esp32cam_ai_blur
../.venv/bin/pio run -t upload --upload-port /dev/ttyACM0   # build + flash
../.venv/bin/pio device monitor --port /dev/ttyACM0         # serial log
```
