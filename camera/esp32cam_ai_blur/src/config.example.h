// Copy this file to config.h and fill in your values (one config.h per camera).
#pragma once

// ---- Wi-Fi (2.4 GHz only) ----
// Known networks, tried in this order. If one can't be reached (e.g. outdoors, out of range of
// the venue Wi-Fi), the camera switches to the next one after a few attempts.
#define WIFI_NETWORKS { \
    {"phone-hotspot", "hotspot-password"}, \
    {"venue-wifi",    "venue-password"}, \
}
#define CAMERA_HOSTNAME "bikeslot-cam"     // name in the phone's hotspot device list

// ---- Privacy: AI person blurring on the camera ----
#define PERSON_SCORE_THR     0.4f         // lower = blur more eagerly (model default is 0.7)
#define PERSON_BOX_PAD       0.15f        // grow each person box by 15 % on every side
#define BLUR_BLOCK_PX        16           // pixelation block size

// ---- Relay: push pictures + health over the internet (e.g. via a phone hotspot) ----
// Leave RELAY_URL empty ("") to switch pushing off (pull mode only: GET /snapshot, /health).
// Values come from the relay's deployment (Render: Environment tab). No trailing slash.
#define RELAY_URL            "https://bikeslot-relay.onrender.com"
#define UPLOAD_TOKEN         "paste-UPLOAD_TOKEN-from-the-relay"
#define CAMERA_ID            "cam01"      // 1-32 characters: A-Z a-z 0-9 _ -
#define SNAPSHOT_INTERVAL_S  10           // ~16 MB mobile data per hour at 10 s; 2 s = fresher, ~90 MB/h
#define HEALTH_INTERVAL_S    10
