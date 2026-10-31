// Live-view page served by the camera at GET /
#pragma once
#include <pgmspace.h>

static const char LIVE_PAGE[] PROGMEM = R"HTML(<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Find my Bike Slot - camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px system-ui;text-align:center}
img{max-width:100%;margin-top:8px}p{margin:8px}</style></head>
<body><p><b>Find my Bike Slot</b> - camera live view (processed on the camera: moving areas pixelated)</p>
<img id="v" alt="loading..."><p id="s"></p>
<script>
const v=document.getElementById('v'),s=document.getElementById('s');
function next(){v.src='/capture?t='+Date.now();}
v.onload=()=>{s.textContent='updated '+new Date().toLocaleTimeString();setTimeout(next,800);};
v.onerror=()=>{s.textContent='no picture - retrying';setTimeout(next,2000);};
next();
</script></body></html>)HTML";
