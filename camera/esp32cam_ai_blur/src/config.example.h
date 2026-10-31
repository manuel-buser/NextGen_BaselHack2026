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
