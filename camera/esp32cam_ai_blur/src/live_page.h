// Page served by the camera at GET /: camera settings (like Espressif's CameraWebServer) on the
// left, the processed live picture on the right. Talks to /status, /control and /snapshot.
#pragma once

static const char LIVE_PAGE[] = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Find my Bike Slot - camera</title>
<style>
body{margin:0;background:#181818;color:#eee;font:15px system-ui,Arial}
header{padding:10px 16px;background:#222;border-bottom:1px solid #333}
header b{color:#ff3034}
.wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
#panel{background:#363636;border-radius:4px;padding:12px 14px;width:330px;box-sizing:border-box}
#panel h3{margin:4px 0 10px;font-size:15px}
.row{display:flex;align-items:center;justify-content:space-between;margin:9px 0;min-height:24px}
.row label{flex:0 0 130px}
.row .ctl{flex:1;display:flex;align-items:center;gap:6px}
.row .ctl span{font-size:12px;color:#bbb;min-width:22px;text-align:center}
input[type=range]{flex:1;accent-color:#ff3034}
select{flex:1;background:#fff;border:0;border-radius:3px;padding:3px}
.sw{position:relative;width:46px;height:24px;flex:0 0 46px}
.sw input{display:none}
.sw i{position:absolute;inset:0;background:grey;border-radius:12px;cursor:pointer;transition:.2s}
.sw i:before{content:"";position:absolute;width:18px;height:18px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s}
.sw input:checked+i{background:#ff3034}
.sw input:checked+i:before{transform:translateX(22px)}
.na{opacity:.35}
button{background:#ff3034;color:#fff;border:0;border-radius:4px;padding:6px 14px;cursor:pointer}
#view{flex:1;min-width:320px;text-align:center}
#view img{max-width:100%;border:1px solid #333}
#info{color:#bbb;margin-top:6px;font-size:13px}
.note{font-size:12px;color:#aaa;margin-top:12px;line-height:1.4}
</style></head><body>
<header><b>Find my Bike Slot</b> - camera (people and licence plates are pixelated <u>on the camera</u> before the picture leaves it)</header>
<div class="wrap">
<div id="panel"><h3>Camera settings</h3><div id="ctl"></div>
<div class="row"><button id="pause">Pause live view</button><button id="still">Save still</button></div>
<div class="note">Privacy settings (person and plate blurring) are fixed in the firmware and cannot be switched off here.</div>
</div>
<div id="view"><img id="v" alt="loading..."><div id="info"></div></div>
</div>
<script>
const C=[
 ['autotune','Auto-Tune exposure','sw'],
 ['xclk','XCLK MHz','num',8,24],
 ['framesize','Resolution','sel',[[6,'QVGA (320x240)'],[8,'CIF (400x296)'],[9,'HVGA (480x320)'],[10,'VGA (640x480)'],[11,'SVGA (800x600)']]],
 ['quality','Quality','rng',4,63],
 ['brightness','Brightness','rng',-2,2],
 ['contrast','Contrast','rng',-2,2],
 ['saturation','Saturation','rng',-2,2],
 ['sharpness','Sharpness','rng',-3,3],
 ['denoise','De-Noise','rng',0,8],
 ['ae_level','Exposure Level','rng',-2,2],
 ['gainceiling','Gainceiling','rng',0,6],
 ['special_effect','Special Effect','sel',[[0,'No Effect'],[1,'Negative'],[2,'Grayscale'],[3,'Red Tint'],[4,'Green Tint'],[5,'Blue Tint'],[6,'Sepia']]],
 ['awb','AWB Enable','sw'],
 ['awb_gain','Advanced AWB','sw'],
 ['wb_mode','WB Mode','sel',[[0,'Auto'],[1,'Sunny'],[2,'Cloudy'],[3,'Office'],[4,'Home']]],
 ['aec','AEC Enable','sw'],
 ['aec2','Night Mode','sw'],
 ['aec_value','Exposure','rng',0,1200],
 ['agc','AGC','sw'],
 ['agc_gain','Gain','rng',0,30],
 ['raw_gma','GMA Enable','sw'],
 ['lenc','Lens Correction','sw'],
 ['hmirror','H-Mirror','sw'],
 ['vflip','V-Flip','sw'],
 ['bpc','BPC','sw'],
 ['wpc','WPC','sw'],
 ['dcw','DCW (downsize)','sw'],
 ['colorbar','Color Bar','sw']];
const ctl=document.getElementById('ctl'),el={};
const MANUAL=['ae_level','aec','aec_value','agc','agc_gain'];
function send(k,v,row){if(MANUAL.includes(k))el.autotune.checked=false;
 fetch('/control?var='+k+'&val='+v).then(r=>{row.classList.toggle('na',!r.ok);if(k=='framesize'||k=='autotune')setTimeout(load,300);});}
for(const [k,label,type,a,b] of C){
 const row=document.createElement('div');row.className='row';
 row.innerHTML='<label>'+label+'</label><div class="ctl"></div>';const d=row.lastChild;let inp;
 if(type=='rng'){d.innerHTML='<span>'+a+'</span><input type="range" min="'+a+'" max="'+b+'"><span>'+b+'</span>';inp=d.querySelector('input');inp.onchange=()=>send(k,inp.value,row);}
 else if(type=='num'){d.innerHTML='<input type="number" min="'+a+'" max="'+b+'" style="width:60px"><button>Set</button>';inp=d.querySelector('input');d.querySelector('button').onclick=()=>send(k,inp.value,row);}
 else if(type=='sel'){inp=document.createElement('select');for(const [v,t] of a)inp.add(new Option(t,v));d.appendChild(inp);inp.onchange=()=>send(k,inp.value,row);}
 else{d.innerHTML='<label class="sw"><input type="checkbox"><i></i></label>';inp=d.querySelector('input');inp.onchange=()=>send(k,inp.checked?1:0,row);}
 el[k]=inp;ctl.appendChild(row);}
function range(k,a,b){const i=el[k];i.min=a;i.max=b;const sp=i.parentNode.querySelectorAll('span');sp[0].textContent=a;sp[1].textContent=b;}
function load(){fetch('/status').then(r=>r.json()).then(s=>{
 if(s.pid==0x3660||s.pid==0x5640){range('brightness',-3,3);range('contrast',-3,3);range('saturation',-4,4);range('sharpness',-3,3);range('ae_level',-5,5);range('gainceiling',0,511);range('agc_gain',0,64);}
 for(const k in el){if(!(k in s))continue;const i=el[k];
 if(i.type=='checkbox')i.checked=!!s[k];else i.value=s[k];}});}
load();
const v=document.getElementById('v'),info=document.getElementById('info');let paused=false;
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!paused)next();});
async function next(){if(paused||document.hidden)return;try{const r=await fetch('/snapshot?t='+Date.now());
 if(!r.ok)throw 0;const b=await r.blob();URL.revokeObjectURL(v.src);v.src=URL.createObjectURL(b);
 info.textContent='updated '+new Date().toLocaleTimeString()+'  |  people blurred: '+r.headers.get('X-People-Blurred')+
 '  |  plate areas blurred: '+r.headers.get('X-Plates-Blurred')+'  |  processing '+r.headers.get('X-Detect-Ms')+' ms'+
 '  |  auto-tune: '+r.headers.get('X-Autotune');
 setTimeout(next,1000);}catch(e){info.textContent='no picture - retrying';setTimeout(next,2000);}}
next();
document.getElementById('pause').onclick=e=>{paused=!paused;e.target.textContent=paused?'Resume live view':'Pause live view';if(!paused)next();};
document.getElementById('still').onclick=()=>{const a=document.createElement('a');a.href=v.src;a.download='bikeslot_'+Date.now()+'.jpg';a.click();};
</script></body></html>)HTML";
