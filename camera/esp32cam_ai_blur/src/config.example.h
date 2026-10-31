// Copy this file to config.h and fill in your values (one config.h per camera).
#pragma once

// ---- Wi-Fi (2.4 GHz only) ----
// Known networks, tried in this order. If one can't be reached (e.g. outdoors, out of range of
// the venue Wi-Fi), the camera switches to the next one after a few attempts.
// Third value: the laptop's upload URL on that network (laptop IP + :8000/api/upload).
#define WIFI_NETWORKS { \
    {"phone-hotspot", "hotspot-password", "http://10.0.0.2:8000/api/upload"}, \
    {"venue-wifi",    "venue-password",   "http://192.168.1.50:8000/api/upload"}, \
}
#define CAMERA_HOSTNAME "bikeslot-cam"     // name in the phone's hotspot device list

// ---- Server ----
#define CAMERA_ID      "test-cam"         // must match a camera id in server/config/sites.yaml
#define UPLOAD_TOKEN   "change-me"        // must match UPLOAD_TOKEN on the server

// ---- Capture ----
#define SNAPSHOT_INTERVAL_S  30           // push a picture this often (pull via /capture any time)

// ---- Privacy: AI person blurring on the camera ----
#define PERSON_SCORE_THR     0.4f         // lower = blur more eagerly (model default is 0.7)
#define PERSON_BOX_PAD       0.15f        // grow each person box by 15 % on every side
#define BLUR_BLOCK_PX        16           // pixelation block size
