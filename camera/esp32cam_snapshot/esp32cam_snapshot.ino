// Find my Bike Slot - ESP32-CAM snapshot uploader
//
// Takes a still image every SNAPSHOT_INTERVAL_S seconds and POSTs it as JPEG to the
// server. No video stream, no SD card, nothing stored on the device.
//
// Boards (select in config.h), Arduino core esp32 >= 2.0.x:
//   Seeed XIAO ESP32S3 Sense (OV2640 or OV3660) - PlatformIO env seeed_xiao_esp32s3,
//     board_build.arduino.memory_type = qio_opi, -DBOARD_HAS_PSRAM
//   AI-Thinker ESP32-CAM - Tools > Board "AI Thinker ESP32-CAM", PSRAM enabled, Huge APP
//
// Setup: copy config.example.h -> config.h and edit it.

#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "config.h"

#ifndef FLIP_VERTICAL
#define FLIP_VERTICAL 0
#endif
#ifndef MIRROR_HORIZONTAL
#define MIRROR_HORIZONTAL 0
#endif

// Motion trigger defaults (override in config.h)
#ifndef MOTION_TRIGGER
#define MOTION_TRIGGER 1
#endif
#ifndef MOTION_CHECK_MS
#define MOTION_CHECK_MS 300
#endif
#ifndef MOTION_PIXEL_DIFF
#define MOTION_PIXEL_DIFF 25
#endif
#ifndef MOTION_MIN_FRACTION
#define MOTION_MIN_FRACTION 0.02f
#endif
#ifndef SETTLE_MS
#define SETTLE_MS 5000
#endif
#ifndef MAX_ACTIVE_MS
#define MAX_ACTIVE_MS 60000
#endif
#ifndef HEARTBEAT_S
#define HEARTBEAT_S 120
#endif
#ifndef MOTION_DEBUG
#define MOTION_DEBUG 0
#endif

// Privacy: pixelate whatever is moving at capture time, on the camera, before encoding
#ifndef MOTION_BLUR
#define MOTION_BLUR 1
#endif
#ifndef BLUR_BLOCK_PX
#define BLUR_BLOCK_PX 24
#endif
#ifndef BLUR_HOLD_MS
#define BLUR_HOLD_MS 1500
#endif
#ifndef BLUR_DILATE_CELLS
#define BLUR_DILATE_CELLS 2
#endif
#if MOTION_BLUR && !MOTION_TRIGGER
#error "MOTION_BLUR needs MOTION_TRIGGER 1 (it reuses the motion thumbnails)"
#endif

// Small web server on the camera: live view for aiming + /capture for pull mode
#ifndef LIVE_VIEW
#define LIVE_VIEW 1
#endif

// Editing pixels on the camera (masking / blurring) needs raw RGB565 frames instead of JPEG
#define RAW_CAPTURE (ON_DEVICE_MASK || MOTION_BLUR)

#if defined(CAMERA_MODEL_XIAO_ESP32S3)
// ---- Seeed XIAO ESP32S3 Sense pin map ----
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13
#else
// ---- AI-Thinker ESP32-CAM pin map ----
#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_GPIO     4
#endif

// (Defined up here: the .ino converter puts function prototypes above the first function.)
// A picture as it may leave the camera: pixelated / masked on the device, JPEG-encoded.
// Used for uploads and for the live view, so both always show exactly the same thing.
struct Shot {
  camera_fb_t *fb = nullptr;
  uint8_t *jpg = nullptr;
  size_t len = 0;
  int blurred = 0;
  bool ownJpg = false;
};

static const int MASKED_JPEG_QUALITY = 80;  // frame2jpg uses a 1-100 scale (higher = better)

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

#if RAW_CAPTURE
  // Raw pixels so we can black out / pixelate them before encoding.
  config.pixel_format = PIXFORMAT_RGB565;
#if CONFIG_IDF_TARGET_ESP32S3
  config.frame_size   = FRAMESIZE_VGA;      // 640x480 raw = 600 KB, fine with the S3's 8 MB PSRAM
#else
  config.frame_size   = FRAMESIZE_QVGA;     // 320x240; larger raw frames strain the classic ESP32
#endif
#else
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_SVGA;     // 800x600: enough for occupancy, too coarse to read plates at distance
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count     = 2;                  // two small JPEG buffers so every grab is a current frame
  config.grab_mode    = CAMERA_GRAB_LATEST;
#endif

  if (!psramFound()) {
    Serial.println("PSRAM not found - enable it in Tools > PSRAM");
    return false;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  bool ov3660 = s->id.PID == OV3660_PID;
  Serial.printf("Sensor PID 0x%x%s\n", s->id.PID, ov3660 ? " (OV3660)" : "");
  if (ov3660) {
    // OV3660 images come out upside down and over-saturated by default
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  s->set_vflip(s, ov3660 ? !FLIP_VERTICAL : FLIP_VERTICAL);
  s->set_hmirror(s, MIRROR_HORIZONTAL);
  return true;
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(WiFi.status() == WL_CONNECTED ? " ok" : " failed");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("IP: %s  RSSI: %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  }
}

#if ON_DEVICE_MASK
// Black out every pixel that is not inside one of the KEEP_RECTS.
// RGB565 black is 0x0000, so byte order does not matter.
void applyMask(camera_fb_t *fb) {
  const int w = fb->width, h = fb->height;
  const int nRects = sizeof(KEEP_RECTS) / sizeof(KEEP_RECTS[0]);
  uint16_t *px = (uint16_t *)fb->buf;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      bool keep = false;
      for (int r = 0; r < nRects && !keep; r++) {
        const int rx = KEEP_RECTS[r][0] * w, ry = KEEP_RECTS[r][1] * h;
        const int rw = KEEP_RECTS[r][2] * w, rh = KEEP_RECTS[r][3] * h;
        keep = x >= rx && x < rx + rw && y >= ry && y < ry + rh;
      }
      if (!keep) px[y * w + x] = 0;
    }
  }
}
#endif

// Grab a fresh frame. With fb_count = 1 the buffer may hold a frame captured right after
// the previous upload, so throw one away first.
camera_fb_t *grabFresh() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
  return esp_camera_fb_get();
}

#if MOTION_BLUR
bool makeThumb(camera_fb_t *fb);
float motionFraction();
int blurRecentMotion(camera_fb_t *fb);
#endif

bool takeShot(Shot &s) {
  s.fb = grabFresh();
  if (!s.fb) {
    Serial.println("Capture failed");
    return false;
  }
  s.jpg = s.fb->buf;
  s.len = s.fb->len;
#if MOTION_BLUR
  // compare this exact frame with the last check, so whatever moves *now* is marked too
  if (makeThumb(s.fb)) motionFraction();
  s.blurred = blurRecentMotion(s.fb);
#endif
#if ON_DEVICE_MASK
  applyMask(s.fb);
#endif
#if RAW_CAPTURE
  if (!frame2jpg(s.fb, MASKED_JPEG_QUALITY, &s.jpg, &s.len)) {
    Serial.println("JPEG encode failed");
    esp_camera_fb_return(s.fb);
    return false;
  }
  s.ownJpg = true;
#endif
  return true;
}

void releaseShot(Shot &s) {
  if (s.ownJpg) free(s.jpg);
  if (s.fb) esp_camera_fb_return(s.fb);
  s = Shot();
}

// trigger: why the picture was taken ("boot", "motion", "motion-timeout", "heartbeat", "timer")
void uploadSnapshot(const char *trigger) {
  Shot s;
  if (!takeShot(s)) return;
  uint8_t *jpg = s.jpg;
  const size_t jpgLen = s.len;
  const int blurred = s.blurred;

  HTTPClient http;
  http.begin(SERVER_URL);
  http.setTimeout(10000);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Camera-Id", CAMERA_ID);
  http.addHeader("X-Upload-Token", UPLOAD_TOKEN);
  http.addHeader("X-Masked", ON_DEVICE_MASK ? "1" : "0");
  http.addHeader("X-Trigger", trigger);
  http.addHeader("X-Blurred-Blocks", String(blurred));

  int code = http.POST(jpg, jpgLen);
  Serial.printf("Upload [%s] %u bytes, %d blocks pixelated on camera -> HTTP %d\n",
                trigger, (unsigned)jpgLen, blurred, code);
  if (code > 0) Serial.println(http.getString());
  http.end();
  releaseShot(s);
}

// ---------------------------------------------------------------- live view / pull mode
// GET /         page that refreshes the processed picture about once a second (for aiming)
// GET /capture  one processed JPEG - the laptop can fetch pictures this way (poll_camera.py)
//               when the camera cannot reach the laptop (firewall, hotspot)
// Both only ever serve what takeShot() produces, i.e. already pixelated / masked on the camera.
#if LIVE_VIEW
#include <WebServer.h>
WebServer web(80);

#include "live_page.h"   // HTML in a header: the .ino converter would mistake its JavaScript for C++

void handleCapture() {
  Shot s;
  if (!takeShot(s)) {
    web.send(503, "text/plain", "capture failed");
    return;
  }
  web.sendHeader("Cache-Control", "no-store");
  web.sendHeader("X-Blurred-Blocks", String(s.blurred));
  web.setContentLength(s.len);
  web.send(200, "image/jpeg", "");
  web.client().write(s.jpg, s.len);
  releaseShot(s);
}

void startLiveView() {
  web.on("/", []() { web.send_P(200, "text/html", LIVE_PAGE); });
  web.on("/capture", handleCapture);
  web.begin();
  Serial.printf("Live view: http://%s/\n", WiFi.localIP().toString().c_str());
}
#endif

void sendSnapshot(const char *trigger) {
  connectWiFi();
  if (WiFi.status() == WL_CONNECTED) uploadSnapshot(trigger);
}

// ---------------------------------------------------------------- motion trigger
// Every MOTION_CHECK_MS a frame is shrunk to a 1/8-scale grey thumbnail (800x600 -> 100x75)
// and compared with the previous one - only inside KEEP_RECTS, so people passing outside the
// racks do not trigger. On motion we wait until the scene has been still for SETTLE_MS:
// the picture is taken *after* someone has parked and walked away, not while they block the slot.
#if MOTION_TRIGGER
static uint8_t *thumbPrev = nullptr, *thumbCur = nullptr, *regionMask = nullptr, *rgbBuf = nullptr;
static int tw = 0, th = 0, regionPixels = 0;
static bool havePrev = false;
static uint32_t *lastMoved = nullptr;   // per thumbnail cell: millis() of its last change (0 = never)

static inline uint8_t grey565(uint8_t hi, uint8_t lo) {
  // esp32-camera stores RGB565 big-endian (high byte first)
  const uint16_t p = (hi << 8) | lo;
  const uint8_t r = (p >> 11) << 3, g = ((p >> 5) & 0x3f) << 2, b = (p & 0x1f) << 3;
  return (r * 77 + g * 150 + b * 29) >> 8;
}

bool allocThumbs(int w, int h) {
  tw = w;
  th = h;
  thumbPrev  = (uint8_t *)ps_malloc(w * h);
  thumbCur   = (uint8_t *)ps_malloc(w * h);
  regionMask = (uint8_t *)ps_malloc(w * h);
  rgbBuf     = (uint8_t *)ps_malloc((w + 1) * (h + 1) * 2);  // margin for odd frame sizes
  lastMoved  = (uint32_t *)ps_calloc(w * h, sizeof(uint32_t));
  if (!thumbPrev || !thumbCur || !regionMask || !rgbBuf || !lastMoved) {
    Serial.println("motion: out of memory");
    return false;
  }
  const int nRects = sizeof(KEEP_RECTS) / sizeof(KEEP_RECTS[0]);
  regionPixels = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const float fx = (x + 0.5f) / w, fy = (y + 0.5f) / h;
      bool in = false;
      for (int r = 0; r < nRects && !in; r++) {
        in = fx >= KEEP_RECTS[r][0] && fx < KEEP_RECTS[r][0] + KEEP_RECTS[r][2] &&
             fy >= KEEP_RECTS[r][1] && fy < KEEP_RECTS[r][1] + KEEP_RECTS[r][3];
      }
      regionMask[y * w + x] = in;
      regionPixels += in;
    }
  }
  Serial.printf("motion: %dx%d thumbnails, %d px in rack area\n", w, h, regionPixels);
  return true;
}

// Fill thumbCur from a camera frame.
bool makeThumb(camera_fb_t *fb) {
  const int w = fb->width / 8, h = fb->height / 8;
  if (!thumbCur && !allocThumbs(w, h)) return false;
#if RAW_CAPTURE
  // raw RGB565: average 16 pixels (every 2nd row/column) of each 8x8 cell. A single sample
  // per cell is far too noisy - sensor noise alone would look like motion everywhere.
  for (int y = 0; y < th; y++) {
    for (int x = 0; x < tw; x++) {
      int sum = 0;
      for (int dy = 0; dy < 8; dy += 2) {
        const size_t row = (size_t)(y * 8 + dy) * fb->width;
        for (int dx = 0; dx < 8; dx += 2) {
          const size_t i = (row + x * 8 + dx) * 2;
          sum += grey565(fb->buf[i], fb->buf[i + 1]);
        }
      }
      thumbCur[y * tw + x] = sum / 16;
    }
  }
#else
  // the JPEG decoder can output 1/8 scale directly, which is cheap
  if (!jpg2rgb565(fb->buf, fb->len, rgbBuf, JPG_SCALE_8X)) return false;
  for (int i = 0; i < tw * th; i++) thumbCur[i] = grey565(rgbBuf[2 * i], rgbBuf[2 * i + 1]);
#endif
  return true;
}

// Share of rack-area pixels that changed since the previous thumbnail (0..1).
float motionFraction() {
  if (!havePrev) {
    memcpy(thumbPrev, thumbCur, tw * th);
    havePrev = true;
    return 0;
  }
  // Remove global brightness shifts (auto-exposure, a cloud): compare each cell relative to
  // its picture's mean, so only local changes count as motion.
  const int n = tw * th;
  long sumCur = 0, sumPrev = 0;
  for (int i = 0; i < n; i++) {
    sumCur += thumbCur[i];
    sumPrev += thumbPrev[i];
  }
  const int offset = (int)((sumCur - sumPrev) / n);

  int changed = 0;
  const uint32_t now = millis() | 1;    // never 0, which means "never moved"
  for (int i = 0; i < n; i++) {
    if (abs((int)thumbCur[i] - (int)thumbPrev[i] - offset) > MOTION_PIXEL_DIFF) {
      lastMoved[i] = now;               // whole frame: for blurring, people anywhere count
      if (regionMask[i]) changed++;     // rack area only: for triggering
    }
  }
  uint8_t *t = thumbPrev;  // current becomes the reference for the next check
  thumbPrev = thumbCur;
  thumbCur = t;
  return regionPixels ? (float)changed / regionPixels : 0;
}

#if MOTION_BLUR
// Pixelate every BLUR_BLOCK_PX square that contains (or is near) a cell which moved within
// the last BLUR_HOLD_MS. Runs on the raw RGB565 frame before it is JPEG-encoded, so a person
// walking through the picture never leaves the camera unblurred. Returns the number of blocks.
// Deliberately short hold: pictures are taken ~5 s after motion stops and the bike that was
// just parked must stay sharp for detection; a person standing still is left to the server's
// person detector (second layer).
int blurRecentMotion(camera_fb_t *fb) {
  if (!lastMoved) return 0;
  const int W = fb->width, H = fb->height, B = BLUR_BLOCK_PX, D = BLUR_DILATE_CELLS;
  const int cellsPerBlock = B / 8;
  const uint32_t now = millis();
  uint8_t *px = fb->buf;
  int blocks = 0;

  for (int by = 0; by < H; by += B) {
    for (int bx = 0; bx < W; bx += B) {
      // any recently moving cell in this block or within D cells around it?
      const int cx0 = max(0, bx / 8 - D), cx1 = min(tw - 1, bx / 8 + cellsPerBlock - 1 + D);
      const int cy0 = max(0, by / 8 - D), cy1 = min(th - 1, by / 8 + cellsPerBlock - 1 + D);
      bool hot = false;
      for (int cy = cy0; cy <= cy1 && !hot; cy++) {
        for (int cx = cx0; cx <= cx1 && !hot; cx++) {
          const uint32_t t = lastMoved[cy * tw + cx];
          hot = t && now - t < BLUR_HOLD_MS;
        }
      }
      if (!hot) continue;

      // average the block (RGB565 is stored big-endian) and fill it with that colour
      const int x1 = min(W, bx + B), y1 = min(H, by + B);
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int y = by; y < y1; y++) {
        for (int x = bx; x < x1; x++) {
          const size_t i = ((size_t)y * W + x) * 2;
          const uint16_t p = (px[i] << 8) | px[i + 1];
          r += p >> 11;
          g += (p >> 5) & 0x3f;
          b += p & 0x1f;
          n++;
        }
      }
      const uint16_t avg = ((r / n) << 11) | ((g / n) << 5) | (b / n);
      for (int y = by; y < y1; y++) {
        for (int x = bx; x < x1; x++) {
          const size_t i = ((size_t)y * W + x) * 2;
          px[i] = avg >> 8;
          px[i + 1] = avg & 0xff;
        }
      }
      blocks++;
    }
  }
  return blocks;
}
#endif

void motionLoop() {
  enum State { IDLE, ACTIVE };
  static State state = IDLE;
  static unsigned long lastCheck = 0, lastMotion = 0, activeSince = 0, lastUpload = 0;
  static bool booted = false;
  const unsigned long now = millis();

  if (!booted) {                       // one picture right after start: the server gets a state at once
    sendSnapshot("boot");
    booted = true;
    lastUpload = millis();
    havePrev = false;
    return;
  }

  if (now - lastCheck >= MOTION_CHECK_MS) {
    lastCheck = now;
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      const bool ok = makeThumb(fb);
      esp_camera_fb_return(fb);
      if (ok) {
        const float f = motionFraction();
#if MOTION_DEBUG
        if (f > 0.005f) Serial.printf("motion: %.1f%%\n", f * 100);
#endif
        if (f >= MOTION_MIN_FRACTION) {
          if (state == IDLE) {
            state = ACTIVE;
            activeSince = now;
            Serial.printf("Motion (%.1f%% of rack area) - waiting until it is still for %d s\n",
                          f * 100, SETTLE_MS / 1000);
          }
          lastMotion = now;
        }
      }
    }
  }

  if (state == ACTIVE) {
    const bool settled = now - lastMotion >= SETTLE_MS;
    if (settled || now - activeSince >= MAX_ACTIVE_MS) {   // someone lingering: send anyway
      sendSnapshot(settled ? "motion" : "motion-timeout");
      state = IDLE;
      lastUpload = millis();
      havePrev = false;                // re-baseline: uploading took a moment
    }
  } else if (now - lastUpload >= HEARTBEAT_S * 1000UL) {
    sendSnapshot("heartbeat");                 // proves the camera is alive, catches slow changes
    lastUpload = millis();
    havePrev = false;
  }
}
#endif

void setup() {
  Serial.begin(115200);
#ifdef FLASH_LED_GPIO
  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);   // keep the AI-Thinker's bright flash LED off
#endif

  if (!initCamera()) {
    delay(5000);
    ESP.restart();
  }
  // Let auto-exposure / white balance settle.
  for (int i = 0; i < 5; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(100);
  }
  connectWiFi();
#if LIVE_VIEW
  startLiveView();
#endif
}

void loop() {
#if LIVE_VIEW
  web.handleClient();
  static unsigned long lastAddr = 0;
  if (millis() - lastAddr >= 15000) {   // USB serial often misses the boot messages
    lastAddr = millis();
    Serial.printf("[%s] live view http://%s/  -> uploads to %s\n", WIFI_SSID,
                  WiFi.localIP().toString().c_str(), SERVER_URL);
  }
#endif
#if MOTION_TRIGGER
  motionLoop();
  delay(20);
#else
  static unsigned long lastShot = 0;
  if (lastShot == 0 || millis() - lastShot >= SNAPSHOT_INTERVAL_S * 1000UL) {
    lastShot = millis();
    sendSnapshot("timer");
  }
  delay(100);
#endif
}
