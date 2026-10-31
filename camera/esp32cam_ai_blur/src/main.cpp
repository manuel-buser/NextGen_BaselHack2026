// Find my Bike Slot - camera firmware with on-device AI person blurring
//
// Every picture is processed on the ESP32-S3 before it leaves the camera:
//   raw RGB565 frame -> ESP-DL pedestrian detector -> people pixelated -> JPEG
// and then fetched by the laptop via GET /snapshot (pull mode). GET / is a live view for aiming the camera.
// Unblurred frames exist only in camera RAM for the ~1 s of processing.
//
// Board: Seeed XIAO ESP32S3 Sense (OV2640 / OV3660), ESP-IDF >= 5.3, ESP-DL pedestrian_detect.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#include "esp_camera.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "img_converters.h"
#include "nvs_flash.h"
#include "pedestrian_detect.hpp"

#include "config.h"
#include "live_page.h"

// Defaults for settings added later (override in config.h)
#ifndef CAMERA_VFLIP
#define CAMERA_VFLIP 0          // the IDF camera driver already orients the OV3660 correctly
#endif
#ifndef CAMERA_HMIRROR
#define CAMERA_HMIRROR 0
#endif
#ifndef BLUR_HEAD_ONLY
#define BLUR_HEAD_ONLY 1        // 1 = head & shoulders (keeps bikes visible), 0 = whole person
#endif
#ifndef HEAD_FRACTION
#define HEAD_FRACTION 0.4f      // top part of the person box that counts as head & shoulders
#endif
#ifndef BLUR_HOLD_MS
#define BLUR_HOLD_MS 2000       // keep blurring a person's last position this long (missed frames)
#endif
#ifndef PERSON_HUGE_BOX_FRAC
#define PERSON_HUGE_BOX_FRAC 0.6f   // a "person" box covering more of the picture than this...
#endif
#ifndef PERSON_HUGE_BOX_SCORE
#define PERSON_HUGE_BOX_SCORE 0.7f  // ...only counts with at least this confidence (else false alarm)
#endif

// Licence plates: colour-based finder for yellow (S-pedelec, moped) and white plates.
// Same algorithm and thresholds as server/plate_colour.py, tuned on the sponsor photos.
#ifndef PLATE_BLUR
#define PLATE_BLUR 1
#endif
#ifndef PLATE_BOX_PAD
#define PLATE_BOX_PAD 0.3f      // grow each plate box by 30 % on every side
#endif

static const char *TAG = "bikeslot";

// ---- Seeed XIAO ESP32S3 Sense camera pins ----
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// JPEG quality on the camera-web-server scale (4 = best ... 63 = smallest), settable in the panel.
// The camera delivers raw pixels, so this drives our own encoder (frame2jpg, scale 1-100).
static int s_jpeg_q = 20;   // ~quality 76 on the encoder scale: half the bytes of 10, still sharp
static int jpeg_quality_100() { return std::max(10, std::min(95, 100 - (s_jpeg_q - 4) * 90 / 59)); }

static EventGroupHandle_t s_wifi_events;
static const EventBits_t WIFI_CONNECTED = BIT0;
static esp_ip4_addr_t s_ip = {};

static PedestrianDetect *s_detect = nullptr;
static SemaphoreHandle_t s_cam_lock;  // camera + detector are shared by the loop and the web server

// ---------------------------------------------------------------- Wi-Fi
// Several known networks (WIFI_NETWORKS in config.h), tried in order: after WIFI_SWITCH_AFTER
// failed attempts the camera moves on to the next one. So the same firmware works indoors on
// the venue Wi-Fi and outdoors on a phone hotspot, without reflashing.

struct WifiNet {
    const char *ssid;
    const char *password;
};
static const WifiNet s_nets[] = WIFI_NETWORKS;
static const int s_net_count = sizeof(s_nets) / sizeof(s_nets[0]);
static int s_net = 0, s_net_fails = 0;
#ifndef WIFI_SWITCH_AFTER
#define WIFI_SWITCH_AFTER 4
#endif
#ifndef CAMERA_HOSTNAME
#define CAMERA_HOSTNAME "bikeslot-cam"
#endif

static void wifi_apply(int idx)
{
    wifi_config_t wc = {};
    // the struct fields are fixed-size and need no terminating zero when full
    memcpy(wc.sta.ssid, s_nets[idx].ssid, strnlen(s_nets[idx].ssid, sizeof(wc.sta.ssid)));
    memcpy(wc.sta.password, s_nets[idx].password, strnlen(s_nets[idx].password, sizeof(wc.sta.password)));
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;  // minimum; WPA3 networks work too
    wc.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
    esp_wifi_set_config(WIFI_IF_STA, &wc);
    ESP_LOGI(TAG, "Connecting to %s ...", s_nets[idx].ssid);
}

static void wifi_event(void *, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_events, WIFI_CONNECTED);
        if (s_net_count > 1 && ++s_net_fails >= WIFI_SWITCH_AFTER) {
            s_net_fails = 0;
            s_net = (s_net + 1) % s_net_count;
            ESP_LOGW(TAG, "Wi-Fi not reachable - switching to %s", s_nets[s_net].ssid);
            wifi_apply(s_net);
        } else {
            ESP_LOGW(TAG, "Wi-Fi disconnected - reconnecting to %s", s_nets[s_net].ssid);
        }
        vTaskDelay(pdMS_TO_TICKS(500));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_ip = static_cast<ip_event_got_ip_t *>(data)->ip_info.ip;
        s_net_fails = 0;
        ESP_LOGI(TAG, "Wi-Fi connected to %s, IP " IPSTR, s_nets[s_net].ssid, IP2STR(&s_ip));
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED);
    }
}

static void wifi_start()
{
    s_wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *netif = esp_netif_create_default_wifi_sta();
    esp_netif_set_hostname(netif, CAMERA_HOSTNAME);  // shown in the phone's list of hotspot devices

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event, nullptr, nullptr));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    wifi_apply(s_net);
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);  // lower latency for the live view

}

// ---------------------------------------------------------------- camera

static bool camera_start()
{
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer = LEDC_TIMER_0;
    c.pin_d0 = Y2_GPIO_NUM;
    c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;
    c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;
    c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;
    c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk = XCLK_GPIO_NUM;
    c.pin_pclk = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href = HREF_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_RGB565;  // raw pixels: the detector reads them, the blur edits them
    c.frame_size = FRAMESIZE_VGA;       // 640x480
    c.fb_count = 2;                     // two buffers + GRAB_LATEST: every grab is a current frame
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&c);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return false;
    }
    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {  // OV3660: over-saturated by default
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    // Orientation matters: the person detector only recognises upright people.
    s->set_vflip(s, CAMERA_VFLIP);
    s->set_hmirror(s, CAMERA_HMIRROR);
    ESP_LOGI(TAG, "Camera ready, sensor PID 0x%x", s->id.PID);
    return true;
}

// ---------------------------------------------------------------- exposure auto-tuning
// The sensor's own auto-exposure keeps the frame *average* near a target, so a bright window
// or a white floor dominates: backlit bikes come out too dark, sunny floors clip to white.
// This slow outer loop measures the *median* brightness and the share of blown-out pixels in
// the region of interest and scales the sensor's AEC target; the AEC then sets exposure and
// gain itself. Reference and explanation: server/autotune_ref.py (keep the constants in sync).
#ifndef AUTOTUNE
#define AUTOTUNE 1
#endif
#ifndef AUTOTUNE_PERIOD_MS
#define AUTOTUNE_PERIOD_MS 1500
#endif
#ifndef AUTOTUNE_ROI
#define AUTOTUNE_ROI {0.0f, 0.0f, 1.0f, 1.0f}   // x1, y1, x2, y2 normalised: where the bike slots are
#endif

namespace autotune {
constexpr int DESIRED_MEDIAN = 115;
constexpr float DEADBAND = 0.12f;
constexpr int CLIP_LEVEL = 245;
constexpr float CLIP_MAX = 0.04f, CLIP_MAX_BACKLIT = 0.20f;
constexpr float GAIN = 1.5f, STEP_MIN = 0.67f, STEP_MAX = 1.5f;
constexpr float TARGET_MIN = 12, TARGET_MAX = 180;  // driver's ae_level -5..+5 spans 5..105
constexpr float ROI[4] = AUTOTUNE_ROI;

static volatile bool enabled = AUTOTUNE;
static float target = 55;  // AEC target, 55 = ae_level 0
static volatile int last_median = -1, last_clip_pm = 0, last_target = 55;
static const char *volatile last_reason = "starting";

// OV3660 / OV5640: write the AEC target window directly (same formula as the driver's
// set_ae_level, but continuous). Other sensors: nearest ae_level step.
static void apply(sensor_t *s, float t)
{
    const int tl = static_cast<int>(t + 0.5f);
    if (s->id.PID == OV3660_PID || s->id.PID == OV5640_PID) {
        const int hi = tl * 27 / 25, lo = tl * 23 / 25;
        s->set_reg(s, 0x3a0f, 0xff, hi);
        s->set_reg(s, 0x3a10, 0xff, lo);
        s->set_reg(s, 0x3a1b, 0xff, hi);
        s->set_reg(s, 0x3a1e, 0xff, lo);
        s->set_reg(s, 0x3a11, 0xff, std::min(255, hi * 2));
        s->set_reg(s, 0x3a1f, 0xff, lo / 2);
    } else {
        s->set_ae_level(s, std::max(-2, std::min(2, tl / 10 - 5)));
    }
    last_target = tl;
}

// Median brightness and blown-out share on an 8-px grid inside the ROI (RGB565 big-endian).
static void measure(const camera_fb_t *fb, int &median, float &clipped)
{
    uint32_t hist[256] = {};
    const int W = fb->width, H = fb->height;
    const int x1 = ROI[0] * W, y1 = ROI[1] * H, x2 = ROI[2] * W, y2 = ROI[3] * H;
    uint32_t n = 0;
    for (int y = y1; y < y2; y += 8) {
        for (int x = x1; x < x2; x += 8) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 2;
            const uint16_t p = (fb->buf[i] << 8) | fb->buf[i + 1];
            const int r = (p >> 11) << 3, g = ((p >> 5) & 0x3f) << 2, b = (p & 0x1f) << 3;
            hist[(77 * r + 150 * g + 29 * b) >> 8]++;
            n++;
        }
    }
    uint32_t acc = 0, clip = 0;
    median = 0;
    for (int v = 0; v < 256; v++) {
        if (acc < n / 2) {
            acc += hist[v];
            median = v;
        }
        if (v >= CLIP_LEVEL) clip += hist[v];
    }
    clipped = n ? static_cast<float>(clip) / n : 0;
}

// One control step: the factor for the AEC target (1 = leave it), as in autotune_ref.decide().
static float decide(int median, float clipped, const char *&reason)
{
    const float lo = DESIRED_MEDIAN * (1 - DEADBAND), hi = DESIRED_MEDIAN * (1 + DEADBAND);
    if (median < lo) {
        if (clipped >= CLIP_MAX_BACKLIT) {
            reason = "dark, window blown out: hold";
            return 1;
        }
        reason = "too dark: brighter";
        return std::min(powf(static_cast<float>(DESIRED_MEDIAN) / std::max(median, 1), GAIN), STEP_MAX);
    }
    if (median > hi || clipped > CLIP_MAX) {
        float f = powf(static_cast<float>(DESIRED_MEDIAN) / median, GAIN);
        if (clipped > CLIP_MAX) f = std::min(f, 0.85f);
        reason = (clipped > CLIP_MAX && median <= hi) ? "blown-out areas: darker" : "too bright: darker";
        return std::max(f, STEP_MIN);
    }
    reason = "ok";
    return 1;
}

static void step(sensor_t *s, const camera_fb_t *fb)
{
    int median;
    float clipped;
    measure(fb, median, clipped);
    const char *reason;
    const float f = decide(median, clipped, reason);
    last_median = median;
    last_clip_pm = static_cast<int>(clipped * 1000);
    last_reason = reason;
    if (f == 1) return;
    const float t = std::max(TARGET_MIN, std::min(TARGET_MAX, target * f));
    if (static_cast<int>(t + 0.5f) == static_cast<int>(target + 0.5f)) return;  // at the limit
    ESP_LOGI(TAG, "Auto-tune: median %d, %.1f%% blown out -> %s, AEC target %.0f -> %.0f", median,
             clipped * 100, reason, target, t);
    target = t;
    apply(s, t);
}

// Hand control back to the sensor's automatics and re-apply our target (e.g. after manual tweaks).
static void enable(sensor_t *s, bool on)
{
    enabled = on;
    if (on) {
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
        apply(s, target);
    }
    ESP_LOGI(TAG, "Auto-tune %s", on ? "on" : "off");
}
}  // namespace autotune

static void autotune_task(void *)
{
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(AUTOTUNE_PERIOD_MS));
        if (!autotune::enabled) continue;
        xSemaphoreTake(s_cam_lock, portMAX_DELAY);
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            autotune::step(esp_camera_sensor_get(), fb);  // only reads the frame; it never leaves the chip
            esp_camera_fb_return(fb);
        }
        xSemaphoreGive(s_cam_lock);
    }
}

// ---------------------------------------------------------------- AI blur

// Replace every BLUR_BLOCK_PX square inside the box by its average colour.
// esp32-camera RGB565 is big-endian (high byte first).
static void pixelate(camera_fb_t *fb, int x1, int y1, int x2, int y2)
{
    const int W = fb->width, B = BLUR_BLOCK_PX;
    uint8_t *px = fb->buf;
    for (int by = y1; by < y2; by += B) {
        for (int bx = x1; bx < x2; bx += B) {
            const int ex = std::min(x2, bx + B), ey = std::min(y2, by + B);
            uint32_t r = 0, g = 0, b = 0, n = 0;
            for (int y = by; y < ey; y++) {
                for (int x = bx; x < ex; x++) {
                    const size_t i = (static_cast<size_t>(y) * W + x) * 2;
                    const uint16_t p = (px[i] << 8) | px[i + 1];
                    r += p >> 11;
                    g += (p >> 5) & 0x3f;
                    b += p & 0x1f;
                    n++;
                }
            }
            if (!n) continue;
            const uint16_t avg = ((r / n) << 11) | ((g / n) << 5) | (b / n);
            for (int y = by; y < ey; y++) {
                for (int x = bx; x < ex; x++) {
                    const size_t i = (static_cast<size_t>(y) * W + x) * 2;
                    px[i] = avg >> 8;
                    px[i + 1] = avg & 0xff;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- licence plates
// Reference: server/plate_colour.py (keep thresholds in sync). Works at half resolution:
//   classify pixels yellow / white / dark -> connected clusters -> keep plate-like ones.
// Errs on the side of blurring too much: a wrongly blurred patch costs nothing.
#if PLATE_BLUR
namespace plates {
// OpenCV HSV scale: H 0-180, S/V 0-255
constexpr int YELLOW_H_MIN = 15, YELLOW_H_MAX = 38, YELLOW_S_MIN = 90, YELLOW_V_MIN = 100;
constexpr int WHITE_S_MAX = 50, WHITE_V_MIN = 160, DARK_V_MAX = 90;
constexpr int MIN_AREA_YELLOW = 20, MIN_AREA_WHITE = 60;   // pixels at half resolution
constexpr float MAX_AREA_FRAC = 0.01f, ASPECT_MIN = 0.6f, ASPECT_MAX = 5.0f, FILL_MIN = 0.55f;
constexpr float TEXT_WHITE_MIN = 0.08f, TEXT_WHITE_MAX = 0.55f, COLOUR_MIN_WHITE = 0.45f;
constexpr float TEXT_YELLOW_MAX = 0.6f;
enum : uint8_t { YELLOW = 1, WHITE = 2, DARK = 4 };

static uint8_t *cls = nullptr;     // per half-res pixel: YELLOW | WHITE | DARK bits
static uint8_t *seen = nullptr;    // flood-fill bookkeeping
static int32_t *stack = nullptr;
static int hw = 0, hh = 0;

static void rgb_to_hsv(int r, int g, int b, int &h, int &s, int &v)
{
    const int mx = std::max(r, std::max(g, b)), mn = std::min(r, std::min(g, b)), d = mx - mn;
    v = mx;
    s = mx ? 255 * d / mx : 0;
    if (!d) {
        h = 0;
        return;
    }
    int h360;
    if (mx == r) h360 = 60 * (g - b) / d;
    else if (mx == g) h360 = 120 + 60 * (b - r) / d;
    else h360 = 240 + 60 * (r - g) / d;
    if (h360 < 0) h360 += 360;
    h = h360 / 2;
}

struct Box {
    int x1, y1, x2, y2;  // full-resolution pixels
};

// Find plate boxes in a raw RGB565 (big-endian) frame.
static std::vector<Box> find(const camera_fb_t *fb)
{
    std::vector<Box> out;
    const int W = fb->width, H = fb->height;
    if (cls && (hw != W / 2 || hh != H / 2)) {   // resolution changed in the settings panel
        heap_caps_free(cls);
        heap_caps_free(seen);
        heap_caps_free(stack);
        cls = seen = nullptr;
        stack = nullptr;
    }
    if (!cls) {
        hw = W / 2;
        hh = H / 2;
        cls = static_cast<uint8_t *>(heap_caps_malloc(hw * hh, MALLOC_CAP_SPIRAM));
        seen = static_cast<uint8_t *>(heap_caps_malloc(hw * hh, MALLOC_CAP_SPIRAM));
        stack = static_cast<int32_t *>(heap_caps_malloc(hw * hh * sizeof(int32_t), MALLOC_CAP_SPIRAM));
        if (!cls || !seen || !stack) {
            ESP_LOGE(TAG, "plates: out of memory");
            return out;
        }
    }

    // 1+2. half resolution (2x2 average) and pixel classes
    const uint8_t *px = fb->buf;
    for (int y = 0; y < hh; y++) {
        for (int x = 0; x < hw; x++) {
            int r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const size_t i = (static_cast<size_t>(y * 2 + dy) * W + x * 2 + dx) * 2;
                    const uint16_t p = (px[i] << 8) | px[i + 1];
                    const int r5 = p >> 11, g6 = (p >> 5) & 0x3f, b5 = p & 0x1f;
                    r += (r5 << 3) | (r5 >> 2);
                    g += (g6 << 2) | (g6 >> 4);
                    b += (b5 << 3) | (b5 >> 2);
                }
            }
            int h, s, v;
            rgb_to_hsv(r / 4, g / 4, b / 4, h, s, v);
            uint8_t c = 0;
            if (h >= YELLOW_H_MIN && h <= YELLOW_H_MAX && s >= YELLOW_S_MIN && v >= YELLOW_V_MIN) c |= YELLOW;
            if (s <= WHITE_S_MAX && v >= WHITE_V_MIN) c |= WHITE;
            if (v <= DARK_V_MAX) c |= DARK;
            cls[y * hw + x] = c;
        }
    }

    // 3+4. connected clusters per colour (8-neighbourhood), keep plate-like ones
    const int max_area = static_cast<int>(MAX_AREA_FRAC * hw * hh);
    for (const uint8_t colour : {YELLOW, WHITE}) {
        memset(seen, 0, hw * hh);
        for (int start = 0; start < hw * hh; start++) {
            if (!(cls[start] & colour) || seen[start]) continue;
            int sp = 0, area = 0, x1 = hw, y1 = hh, x2 = -1, y2 = -1;
            stack[sp++] = start;
            seen[start] = 1;
            while (sp) {
                const int i = stack[--sp], x = i % hw, y = i / hw;
                area++;
                x1 = std::min(x1, x); x2 = std::max(x2, x);
                y1 = std::min(y1, y); y2 = std::max(y2, y);
                for (int ny = std::max(0, y - 1); ny <= std::min(hh - 1, y + 1); ny++) {
                    for (int nx = std::max(0, x - 1); nx <= std::min(hw - 1, x + 1); nx++) {
                        const int j = ny * hw + nx;
                        if (!seen[j] && (cls[j] & colour)) {
                            seen[j] = 1;
                            stack[sp++] = j;
                        }
                    }
                }
            }
            const int bw = x2 - x1 + 1, bh = y2 - y1 + 1;
            const int min_area = colour == WHITE ? MIN_AREA_WHITE : MIN_AREA_YELLOW;
            if (area < min_area || area > max_area || bw < 3 || bh < 3) continue;
            const float aspect = static_cast<float>(bw) / bh;
            if (aspect < ASPECT_MIN || aspect > ASPECT_MAX) continue;
            int n_colour = 0, n_dark = 0, n_fill = 0;
            for (int y = y1; y <= y2; y++) {
                for (int x = x1; x <= x2; x++) {
                    const uint8_t c = cls[y * hw + x];
                    n_colour += (c & colour) != 0;
                    n_dark += (c & DARK) != 0;
                    n_fill += (c & (colour | DARK)) != 0;
                }
            }
            const float box_area = static_cast<float>(bw * bh);
            const float fill = n_fill / box_area, text = n_dark / box_area, col = n_colour / box_area;
            if (fill < FILL_MIN) continue;
            if (colour == WHITE && (text < TEXT_WHITE_MIN || text > TEXT_WHITE_MAX || col < COLOUR_MIN_WHITE)) continue;
            if (colour == YELLOW && text > TEXT_YELLOW_MAX) continue;
            out.push_back({x1 * 2, y1 * 2, (x2 + 1) * 2, (y2 + 1) * 2});
        }
    }
    return out;
}
}  // namespace plates
#endif

// Person boxes seen recently (padded, pixel coordinates). Blurring keeps covering them for
// BLUR_HOLD_MS, so a person the detector misses for a frame or two does not flash up unblurred.
struct PersonBox {
    int x1, y1, x2, y2;
    int64_t seen_us;
};
static std::vector<PersonBox> s_recent;

// A picture as it may leave the camera. jpg must be freed with free().
struct Shot {
    uint8_t *jpg = nullptr;
    size_t len = 0;
    int people = 0;           // detected in this frame
    int plates = 0;           // licence-plate-like areas blurred
    float person_area = 0;    // share of the picture covered by people (0..1)
    int64_t detect_ms = 0;
    std::string boxes;        // "x1,y1,x2,y2;..." normalised 0..1, for the server
};

// Run the person detector, then pixelate every recent person (head & shoulders or whole box).
static void blur_people(camera_fb_t *fb, Shot &s)
{
    dl::image::img_t img = {};
    img.data = fb->buf;
    img.width = static_cast<uint16_t>(fb->width);
    img.height = static_cast<uint16_t>(fb->height);
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;

    const int64_t t0 = esp_timer_get_time();
    auto &results = s_detect->run(img);
#if PLATE_BLUR
    // on the unmodified frame: pixelated heads must not look like plates (or hide them)
    const std::vector<plates::Box> plate_boxes = plates::find(fb);
#endif
    const int64_t now = esp_timer_get_time();
    s.detect_ms = (now - t0) / 1000;

    const int W = fb->width, H = fb->height;
    float area = 0;
    char buf[48];
    for (const auto &r : results) {
        const int w = r.box[2] - r.box[0], h = r.box[3] - r.box[1];
        // The model sometimes reports a low-confidence "person" spanning the whole frame (seen on
        // empty corridors). A mounted camera never sees a person that large, so require high
        // confidence for huge boxes - otherwise the whole picture would be pixelated.
        if (static_cast<float>(w) * h > PERSON_HUGE_BOX_FRAC * fb->width * fb->height && r.score < PERSON_HUGE_BOX_SCORE) {
            ESP_LOGI(TAG, "ignored frame-filling person box (score %.2f)", r.score);
            continue;
        }
        const int px = static_cast<int>(w * PERSON_BOX_PAD), py = static_cast<int>(h * PERSON_BOX_PAD);
        PersonBox b = {std::max(0, r.box[0] - px), std::max(0, r.box[1] - py),
                       std::min(W, r.box[2] + px), std::min(H, r.box[3] + py), now};
        s_recent.push_back(b);
        area += static_cast<float>(std::max(0, r.box[2] - r.box[0]) * std::max(0, r.box[3] - r.box[1])) / (W * H);
        snprintf(buf, sizeof(buf), "%s%.3f,%.3f,%.3f,%.3f", s.boxes.empty() ? "" : ";",
                 r.box[0] / (float)W, r.box[1] / (float)H, r.box[2] / (float)W, r.box[3] / (float)H);
        s.boxes += buf;
        ESP_LOGI(TAG, "person %.2f at [%d,%d,%d,%d]", r.score, r.box[0], r.box[1], r.box[2], r.box[3]);
        s.people++;
    }
    s.person_area = std::min(1.0f, area);

    // forget old boxes, blur the rest
    s_recent.erase(std::remove_if(s_recent.begin(), s_recent.end(),
                                  [now](const PersonBox &b) { return now - b.seen_us > BLUR_HOLD_MS * 1000LL; }),
                   s_recent.end());
    for (const auto &b : s_recent) {
        int y2 = b.y2;
#if BLUR_HEAD_ONLY
        y2 = b.y1 + static_cast<int>((b.y2 - b.y1) * HEAD_FRACTION);
#endif
        pixelate(fb, b.x1, b.y1, b.x2, y2);
    }
#if PLATE_BLUR
    for (const auto &p : plate_boxes) {
        const int pw = static_cast<int>((p.x2 - p.x1) * PLATE_BOX_PAD) + 2;
        const int ph = static_cast<int>((p.y2 - p.y1) * PLATE_BOX_PAD) + 2;
        pixelate(fb, std::max(0, p.x1 - pw), std::max(0, p.y1 - ph), std::min(W, p.x2 + pw), std::min(H, p.y2 + ph));
    }
    s.plates = static_cast<int>(plate_boxes.size());
#endif
}

static bool take_shot(Shot &s)
{
    xSemaphoreTake(s_cam_lock, portMAX_DELAY);
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        xSemaphoreGive(s_cam_lock);
        ESP_LOGE(TAG, "Capture failed");
        return false;
    }
    blur_people(fb, s);
    const bool ok = frame2jpg(fb, jpeg_quality_100(), &s.jpg, &s.len);
    esp_camera_fb_return(fb);  // the unblurred raw frame is gone from here on
    xSemaphoreGive(s_cam_lock);
    if (!ok) ESP_LOGE(TAG, "JPEG encode failed");
    return ok;
}

// ---------------------------------------------------------------- shared latest picture
// One background task processes ~1 picture per LIVE_INTERVAL_MS while someone watches the live
// view; every viewer and the server's poller get that same finished picture. Before this, each browser
// request ran the whole pipeline itself and several open tabs made the camera unusably slow.
#ifndef LIVE_INTERVAL_MS
#define LIVE_INTERVAL_MS 1000
#endif
#ifndef LIVE_IDLE_AFTER_S
#define LIVE_IDLE_AFTER_S 10   // nobody fetched a picture for this long: stop processing for the live view
#endif

using ShotPtr = std::shared_ptr<Shot>;
static SemaphoreHandle_t s_latest_lock;
static ShotPtr s_latest;
static int64_t s_latest_us = 0;
static volatile int64_t s_last_view_us = 0;

static ShotPtr new_shot()
{
    ShotPtr sh(new Shot(), [](Shot *p) {
        free(p->jpg);
        delete p;
    });
    if (!take_shot(*sh)) return nullptr;
    xSemaphoreTake(s_latest_lock, portMAX_DELAY);
    s_latest = sh;
    s_latest_us = esp_timer_get_time();
    xSemaphoreGive(s_latest_lock);
    return sh;
}

// The latest processed picture if it is younger than max_age_ms, else a freshly processed one.
static ShotPtr get_shot(int max_age_ms)
{
    xSemaphoreTake(s_latest_lock, portMAX_DELAY);
    ShotPtr sh = s_latest;
    const int64_t age = esp_timer_get_time() - s_latest_us;
    xSemaphoreGive(s_latest_lock);
    if (sh && age <= max_age_ms * 1000LL) return sh;
    return new_shot();
}

static void live_task(void *)
{
    while (true) {
        if (esp_timer_get_time() - s_last_view_us < LIVE_IDLE_AFTER_S * 1000000LL) {
            new_shot();
            vTaskDelay(pdMS_TO_TICKS(LIVE_INTERVAL_MS));
        } else {
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }
}

// ---------------------------------------------------------------- live view / pull mode

static esp_err_t page_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, LIVE_PAGE, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req)
{
    s_last_view_us = esp_timer_get_time();   // keeps the live task running
    ShotPtr sh = get_shot(3000);
    if (!sh) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    const Shot &s = *sh;
    char people[8], plates_n[8], ms[12], area[12];
    snprintf(people, sizeof(people), "%d", s.people);
    snprintf(plates_n, sizeof(plates_n), "%d", s.plates);
    snprintf(ms, sizeof(ms), "%lld", s.detect_ms);
    snprintf(area, sizeof(area), "%.3f", s.person_area);
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_hdr(req, "X-People-Blurred", people);
    httpd_resp_set_hdr(req, "X-Plates-Blurred", plates_n);
    httpd_resp_set_hdr(req, "X-Person-Area", area);
    httpd_resp_set_hdr(req, "X-Detect-Ms", ms);
    if (!s.boxes.empty()) httpd_resp_set_hdr(req, "X-Person-Boxes", s.boxes.c_str());
    char tune[96];
    if (autotune::enabled) {
        snprintf(tune, sizeof(tune), "median %d, %.1f%% blown out, target %d (%s)", autotune::last_median,
                 autotune::last_clip_pm / 10.0f, autotune::last_target, autotune::last_reason);
    } else {
        snprintf(tune, sizeof(tune), "off");
    }
    httpd_resp_set_hdr(req, "X-Autotune", tune);
    return httpd_resp_send(req, reinterpret_cast<const char *>(s.jpg), s.len);
}

// ---- camera settings: same interface as Espressif's CameraWebServer (/status, /control) ----
// Only camera image settings. The privacy settings (person / plate blurring) are deliberately
// NOT exposed here - anyone on the network could switch them off. They live in config.h.

static esp_err_t status_handler(httpd_req_t *req)
{
    sensor_t *s = esp_camera_sensor_get();
    const camera_status_t &st = s->status;
    char json[768];
    snprintf(json, sizeof(json),
             "{\"autotune\":%d,\"at_median\":%d,\"at_clip_pct\":%.1f,\"at_target\":%d,\"xclk\":%d,\"framesize\":%u,\"quality\":%d,\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
             "\"sharpness\":%d,\"denoise\":%u,\"ae_level\":%d,\"gainceiling\":%u,\"special_effect\":%u,"
             "\"awb\":%u,\"awb_gain\":%u,\"wb_mode\":%u,\"aec\":%u,\"aec2\":%u,\"aec_value\":%u,\"agc\":%u,"
             "\"agc_gain\":%u,\"raw_gma\":%u,\"lenc\":%u,\"hmirror\":%u,\"vflip\":%u,\"bpc\":%u,\"wpc\":%u,"
             "\"dcw\":%u,\"colorbar\":%u,\"pid\":%u}",
             autotune::enabled ? 1 : 0, autotune::last_median, autotune::last_clip_pm / 10.0f,
             autotune::last_target, s->xclk_freq_hz / 1000000, st.framesize, s_jpeg_q, st.brightness, st.contrast, st.saturation,
             st.sharpness, st.denoise, st.ae_level, st.gainceiling, st.special_effect, st.awb, st.awb_gain,
             st.wb_mode, st.aec, st.aec2, st.aec_value, st.agc, st.agc_gain, st.raw_gma, st.lenc, st.hmirror,
             st.vflip, st.bpc, st.wpc, st.dcw, st.colorbar, s->id.PID);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t control_handler(httpd_req_t *req)
{
    char query[96], var[32], val[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
        httpd_query_key_value(query, "val", val, sizeof(val)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "need ?var=...&val=...");
        return ESP_FAIL;
    }
    const int v = atoi(val);
    sensor_t *s = esp_camera_sensor_get();
    int res = 0;

    xSemaphoreTake(s_cam_lock, portMAX_DELAY);  // never change settings in the middle of a capture
    // Manual exposure settings and auto-tuning would fight each other: touching one turns it off.
    if (autotune::enabled && (!strcmp(var, "ae_level") || !strcmp(var, "aec") || !strcmp(var, "aec_value") ||
                              !strcmp(var, "agc") || !strcmp(var, "agc_gain"))) {
        autotune::enable(s, false);
    }
    if (!strcmp(var, "autotune")) {
        autotune::enable(s, v != 0);
    } else if (!strcmp(var, "framesize")) {
        // raw RGB565 frames + on-camera AI: up to SVGA (800x600)
        if (v < FRAMESIZE_QVGA || v > FRAMESIZE_SVGA) res = -1;
        else res = s->set_framesize(s, static_cast<framesize_t>(v));
    } else if (!strcmp(var, "quality")) {
        s_jpeg_q = std::max(4, std::min(63, v));
    } else if (!strcmp(var, "xclk")) {
        res = s->set_xclk(s, LEDC_TIMER_0, std::max(8, std::min(24, v)));
    }
    else if (!strcmp(var, "brightness")) res = s->set_brightness(s, v);
    else if (!strcmp(var, "contrast")) res = s->set_contrast(s, v);
    else if (!strcmp(var, "saturation")) res = s->set_saturation(s, v);
    else if (!strcmp(var, "sharpness")) res = s->set_sharpness(s, v);
    else if (!strcmp(var, "denoise")) res = s->set_denoise(s, v);
    else if (!strcmp(var, "ae_level")) res = s->set_ae_level(s, v);
    else if (!strcmp(var, "gainceiling")) res = s->set_gainceiling(s, static_cast<gainceiling_t>(v));
    else if (!strcmp(var, "special_effect")) res = s->set_special_effect(s, v);
    else if (!strcmp(var, "awb")) res = s->set_whitebal(s, v);
    else if (!strcmp(var, "awb_gain")) res = s->set_awb_gain(s, v);
    else if (!strcmp(var, "wb_mode")) res = s->set_wb_mode(s, v);
    else if (!strcmp(var, "aec")) res = s->set_exposure_ctrl(s, v);
    else if (!strcmp(var, "aec2")) res = s->set_aec2(s, v);
    else if (!strcmp(var, "aec_value")) res = s->set_aec_value(s, v);
    else if (!strcmp(var, "agc")) res = s->set_gain_ctrl(s, v);
    else if (!strcmp(var, "agc_gain")) res = s->set_agc_gain(s, v);
    else if (!strcmp(var, "raw_gma")) res = s->set_raw_gma(s, v);
    else if (!strcmp(var, "lenc")) res = s->set_lenc(s, v);
    else if (!strcmp(var, "hmirror")) res = s->set_hmirror(s, v);
    else if (!strcmp(var, "vflip")) res = s->set_vflip(s, v);
    else if (!strcmp(var, "bpc")) res = s->set_bpc(s, v);
    else if (!strcmp(var, "wpc")) res = s->set_wpc(s, v);
    else if (!strcmp(var, "dcw")) res = s->set_dcw(s, v);
    else if (!strcmp(var, "colorbar")) res = s->set_colorbar(s, v);
    else res = -2;
    xSemaphoreGive(s_cam_lock);

    if (res == -2) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "unknown or not allowed setting");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "setting %s = %d -> %s", var, v, res == 0 ? "ok" : "not supported by this sensor");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    if (res) return httpd_resp_send_500(req);
    return httpd_resp_send(req, "ok", 2);
}

static void web_start()
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 16384;  // the detector can run inside the capture handler (no fresh picture yet)
    cfg.lru_purge_enable = true;  // drop idle connections of forgotten browser tabs
    httpd_handle_t server = nullptr;
    ESP_ERROR_CHECK(httpd_start(&server, &cfg));
    const httpd_uri_t page = {.uri = "/", .method = HTTP_GET, .handler = page_handler, .user_ctx = nullptr};
    const httpd_uri_t cap = {.uri = "/snapshot", .method = HTTP_GET, .handler = capture_handler, .user_ctx = nullptr};
    httpd_register_uri_handler(server, &page);
    httpd_register_uri_handler(server, &cap);
    const httpd_uri_t status = {.uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = nullptr};
    const httpd_uri_t control = {.uri = "/control", .method = HTTP_GET, .handler = control_handler, .user_ctx = nullptr};
    httpd_register_uri_handler(server, &status);
    httpd_register_uri_handler(server, &control);
}

// ---------------------------------------------------------------- main

extern "C" void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    s_cam_lock = xSemaphoreCreateMutex();
    s_latest_lock = xSemaphoreCreateMutex();
    if (!camera_start()) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }

    autotune::enable(esp_camera_sensor_get(), AUTOTUNE);
    xTaskCreatePinnedToCore(autotune_task, "autotune", 6144, nullptr, 3, nullptr, 0);

    s_detect = new PedestrianDetect();
    s_detect->set_score_thr(PERSON_SCORE_THR);
    ESP_LOGI(TAG, "Person detector ready (score threshold %.2f)", PERSON_SCORE_THR);

    wifi_start();
    xEventGroupWaitBits(s_wifi_events, WIFI_CONNECTED, pdFALSE, pdTRUE, portMAX_DELAY);
    web_start();
    xTaskCreatePinnedToCore(live_task, "live", 16384, nullptr, 4, nullptr, 1);

    // Pull mode only: the server fetches GET /snapshot. Repeat the address now and then,
    // because USB serial often misses the boot messages.
    while (true) {
        ESP_LOGI(TAG, "[%s] live view http://" IPSTR "/  picture http://" IPSTR "/snapshot",
                 s_nets[s_net].ssid, IP2STR(&s_ip), IP2STR(&s_ip));
        vTaskDelay(pdMS_TO_TICKS(15000));
    }
}
