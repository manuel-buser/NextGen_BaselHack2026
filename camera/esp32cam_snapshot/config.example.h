// Copy this file to config.h and fill in your values.
// config.h is per-camera: flash each ESP32-CAM with its own CAMERA_ID.
#pragma once

// ---- Board ----
// Seeed XIAO ESP32S3 Sense (the boards handed out at BaselHack). Comment out for AI-Thinker ESP32-CAM.
#define CAMERA_MODEL_XIAO_ESP32S3

// ---- Wi-Fi ----
#define WIFI_SSID      "your-ssid"
#define WIFI_PASSWORD  "your-password"

// ---- Server (the laptop running server/app.py) ----
// Use the laptop's LAN IP, not "localhost".
#define SERVER_URL     "http://192.168.1.50:8000/api/upload"
#define CAMERA_ID      "shelter-a"        // must match a camera id in server/config/sites.yaml
#define UPLOAD_TOKEN   "change-me"        // must match UPLOAD_TOKEN on the server

// ---- When to take a picture ----
// 1 = on motion in the rack area (after the scene is still again) + a heartbeat picture
// 0 = plain timer every SNAPSHOT_INTERVAL_S
#define MOTION_TRIGGER       1
#define MOTION_CHECK_MS      300          // how often to compare thumbnails
#define MOTION_PIXEL_DIFF    25           // grey change (0-255) for a pixel to count as "changed"
#define MOTION_MIN_FRACTION  0.02f        // share of rack-area pixels that must change (2 %)
#define SETTLE_MS            5000         // picture once nothing moved for 5 s
#define MAX_ACTIVE_MS        60000        // ...or after 60 s of continuous motion
#define HEARTBEAT_S          120          // picture anyway every 2 min without motion
#define MOTION_DEBUG         0            // 1 = print every small change (for tuning)
#define SNAPSHOT_INTERVAL_S  30           // only used when MOTION_TRIGGER is 0

// ---- Capture ----
#define JPEG_QUALITY         12           // 0-63, lower = better quality / bigger file

// ---- Privacy: blur on the camera ----
// 1 = pixelate everything that moves at capture time *on the ESP32*, before the JPEG is made:
//     people walking through never leave the camera unblurred. (Server blurs people again.)
//     Needs raw frames, so uploads are VGA 640x480 (S3) / QVGA 320x240 (classic ESP32).
#define MOTION_BLUR          1
#define BLUR_BLOCK_PX        24           // pixelation block size (multiple of 8)
#define BLUR_HOLD_MS         1500         // blur cells that moved within this time
#define BLUR_DILATE_CELLS    2            // extra margin around moving areas (1 cell = 8 px)

// ---- Privacy: mask on the device ----
// 1 = capture raw RGB565, black out everything outside the KEEP_RECTS below,
//     then JPEG-encode on the ESP32. Pixels outside the slots never leave the camera.
//     Uses VGA (640x480) on the ESP32-S3, QVGA (320x240) on the classic ESP32.
// 0 = send the full JPEG (SVGA 800x600); the server masks it in RAM on arrival.
#define ON_DEVICE_MASK       0

// Rack area as fractions of the image (x, y, w, h), 0.0-1.0. Used for:
//  - the motion trigger: only changes inside these rectangles count
//  - ON_DEVICE_MASK: everything outside is blacked out before sending
// Cover the rack area generously; the server applies the exact slot polygons afterwards.
static const float KEEP_RECTS[][4] = {
  {0.00f, 0.35f, 1.00f, 0.65f},   // e.g. the lower 65% of the frame where the racks are
};
