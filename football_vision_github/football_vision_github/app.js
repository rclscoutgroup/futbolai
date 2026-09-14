/* Football Vision - browser-only prototype
 * Detection: YOLOv8n ONNX (COCO), loaded via Hugging Face at runtime.
 * Tracking: lightweight IoU tracker.
 * Event inference: ball/player geometry + possession transitions.
 * IMPORTANT: event labels/xG/xA are heuristic unless replaced with custom football-trained models.
 */

const MODEL_URL = 'https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx';
const MODEL_INPUT = 640;
const COCO_PERSON = 0;
const COCO_SPORTS_BALL = 32;
const PITCH_W = 105;
const PITCH_H = 68;

const $ = id => document.getElementById(id);
const state = { file:null, url:null, videoMeta:null, session:null, tracks:new Map(), nextTrack:1, samples:[], actions:[], playerSummary:new Map(), teamColorModels:{}, transform:null, json:null };

const setStatus=(text,ok=true)=>{ $('statusText').textContent=text; $('statusDot').style.background=ok?'#62d99b':'#ff6b6b'; };
const setProgress=(pct,label)=>{ $('progressWrap').classList.remove('hidden'); $('progressBar').style.width=`${pct}%`; $('progressPct').textContent=`${Math.round(pct)}%`; $('progressLabel').textContent=label; };
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const sigmoid=x=>1/(1+Math.exp(-x));

$('chooseBtn').onclick=()=>$('videoInput').click();
$('videoInput').onchange=e=>selectFile(e.target.files[0]);
$('conf').oninput=e=>$('confValue').textContent=Number(e.target.value).toFixed(2);
['dragover','dragenter'].forEach(ev=>$('dropzone').addEventListener(ev,e=>{e.preventDefault();$('dropzone').classList.add('drag')}));
['dragleave','drop'].forEach(ev=>$('dropzone').addEventListener(ev,e=>{e.preventDefault();$('dropzone').classList.remove('drag')}));
$('dropzone').addEventListener('drop',e=>selectFile(e.dataTransfer.files[0]));

function selectFile(file){
  if(!file || !file.type.startsWith('video/')) return;
  state.file=file; state.url=URL.createObjectURL(file); $('fileName').textContent=file.name; $('runBtn').disabled=false;
  $('video').src=state.url; $('video').load(); $('previewSection').classList.remove('hidden');
  $('warningBox').textContent='El navegador procesa el vídeo localmente. En un partido completo, 2–3 fps puede tardar bastante; no cierres la pestaña durante el análisis.';
}

async function loadModel(){
  if(state.session) return state.session;
  setStatus('Cargando modelo…');
  state.session = await ort.InferenceSession.create(MODEL_URL,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  return state.session;
}

function letterbox(canvas, w=MODEL_INPUT, h=MODEL_INPUT){
  const scale=Math.min(w/canvas.width,h/canvas.height);
  const nw=Math.round(canvas.width*scale), nh=Math.round(canvas.height*scale);
  const ox=Math.floor((w-nw)/2), oy=Math.floor((h-nh)/2);
  const c=document.createElement('canvas'); c.width=w;c.height=h; const ctx=c.getContext('2d'); ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,w,h);ctx.drawImage(canvas,0,0,canvas.width,canvas.height,ox,oy,nw,nh);
  const im=ctx.getImageData(0,0,w,h).data; const data=new Float32Array(1*3*w*h); const area=w*h;
  for(let i=0;i<area;i++){data[i]=im[i*4]/255;data[area+i]=im[i*4+1]/255;data[2*area+i]=im[i*4+2]/255;}
  return {tensor:new ort.Tensor('float32',data,[1,3,h,w]),scale,ox,oy};
}

function parseYolo(output, originalW, originalH, meta, confTh){
  const dims=output.dims; const data=output.data;
  let C=dims[1], N=dims[2], transposed=false;
  if(C>N){[C,N]=[N,C];transposed=true;}
  const rows=[];
  for(let i=0;i<N;i++){
    let cx,cy,w,h; if(!transposed){cx=data[i+4*N];cy=data[i+5*N];w=data[i+6*N];h=data[i+7*N];} else {cx=data[i* C];cy=data[i*C+1];w=data[i*C+2];h=data[i*C+3];}
    let best=-1,bestP=0;
    for(let c=4;c<C;c++){const p=transposed?data[i*C+c]:data[c*N+i];if(p>bestP){bestP=p;best=c-4;}}
    if(bestP<confTh) continue;
    const x1=(cx-w/2-meta.ox)/meta.scale, y1=(cy-h/2-meta.oy)/meta.scale, x2=(cx+w/2-meta.ox)/meta.scale, y2=(cy+h/2-meta.oy)/meta.scale;
    const xx1=clamp(x1,0,originalW), yy1=clamp(y1,0,originalH), xx2=clamp(x2,0,originalW), yy2=clamp(y2,0,originalH);
    if(xx2-xx1<4||yy2-yy1<4) continue;
    rows.push({classId:best,score:bestP,x1:xx1,y1:yy1,x2:xx2,y2:yy2,cx:(xx1+xx2)/2,cy:(yy1+yy2)/2,w:xx2-xx1,h:yy2-yy1});
  }
  return nms(rows,0.45);
}
function iou(a,b){const x1=Math.max(a.x1,b.x1),y1=Math.max(a.y1,b.y1),x2=Math.min(a.x2,b.x2),y2=Math.min(a.y2,b.y2);const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);return inter/(a.w*a.h+b.w*b.h-inter+1e-6)}
function nms(rows,th){rows.sort((a,b)=>b.score-a.score);const out=[];while(rows.length){const x=rows.shift();out.push(x);for(let i=rows.length-1;i>=0;i--)if(rows[i].classId===x.classId&&iou(x,rows[i])>th)rows.splice(i,1)}return out}

function dominantHue(video, det){
  const c=document.createElement('canvas');c.width=40;c.height=40;const ctx=c.getContext('2d');
  const y=det.y1+det.h*0.18,h=det.h*0.45,x=det.x1+det.w*0.2,w=det.w*0.6;
  ctx.drawImage(video,x,y,w,h,0,0,40,40); const d=ctx.getImageData(0,0,40,40).data;
  let sum=0,n=0;for(let i=0;i<d.length;i+=4){const r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;const mx=Math.max(r,g,b),mn=Math.min(r,g,b),diff=mx-mn;if(diff<.08||mx<.18)continue;let hue=0;if(mx===r)hue=((g-b)/diff)%6;else if(mx===g)hue=(b-r)/diff+2;else hue=(r-g)/diff+4;sum+=(hue*60+360)%360;n++;}return n?sum/n:null;
}
function nearestTrack(det, candidates){let best=null,bestScore=-1;for(const t of candidates){const ov=iou(det,t.det);const cd=dist(det,t.det);const s=ov*2-Math.min(cd/250,1);if(s>bestScore){bestScore=s;best=t}}return {best,bestScore}}
function assignTracks(playerDets,timestamp,video){
  const active=[...state.tracks.values()].filter(t=>timestamp-t.lastTs<1.6);
  const used=new Set();
  for(const d of playerDets){const {best,bestScore}=nearestTrack(d,active.filter(t=>!used.has(t.id)));let t=null;if(best&&bestScore>.18){t=best;used.add(t.id)}else{t={id:state.nextTrack++,team:null,history:[],hueSamples:[],lastTs:-99,det:null}};t.det=d;t.lastTs=timestamp;t.history.push({ts:timestamp,x:d.cx,y:d.cy,pixX:d.cx,pixY:d.cy});if(t.history.length>80)t.history.shift();const hue=dominantHue(video,d);if(hue!==null)t.hueSamples.push(hue);if(t.hueSamples.length>20)t.hueSamples.shift();state.tracks.set(t.id,t)}
  for(const t of state.tracks.values()) if(timestamp-t.lastTs>1.6 && t.history.length>2){t.expired=true}
}
function finalizeTeams(){
  const tracks=[...state.tracks.values()].filter(t=>t.hueSamples.length>4);
  const vals=tracks.map(t=>median(t.hueSamples)); if(vals.length<2)return;
  let c1=vals[0],c2=vals[vals.length-1]; for(let it=0;it<8;it++){const a=[],b=[];for(const v of vals)(circDist(v,c1)<circDist(v,c2)?a:b).push(v);if(a.length)c1=circularMean(a);if(b.length)c2=circularMean(b)}
  for(const t of tracks){const v=median(t.hueSamples);t.team=circDist(v,c1)<=circDist(v,c2)?'TEAM_A':'TEAM_B'}
}
const circDist=(a,b)=>Math.min(Math.abs(a-b),360-Math.abs(a-b));
const circularMean=xs=>{const s=xs.reduce((q,v)=>q+Math.sin(v*Math.PI/180),0),c=xs.reduce((q,v)=>q+Math.cos(v*Math.PI/180),0);return (Math.atan2(s,c)*180/Math.PI+360)%360};
const median=xs=>{const a=[...xs].sort((a,b)=>a-b);return a[Math.floor(a.length/2)]};

function buildTransform(frameW,frameH){
  // Automatic first-pass transform. It intentionally exposes confidence in the CSV.
  // If the full pitch is visible, normalized image coordinates map to 105x68 m.
  return {type:'normalized_frame_fallback',confidence:.28,apply:(x,y)=>({x:clamp(x/frameW,0,1)*PITCH_W,y:clamp(y/frameH,0,1)*PITCH_H})};
}

function actionInference(samples){
  const actions=[]; let actionId=1; const players=[...state.tracks.values()].filter(t=>t.history.length>3&&!t.expired);
  for(const s of samples){
    const ball=s.ball; if(!ball)continue;
    let nearest=null,best=1e9;for(const t of players){const p=interpTrack(t.history,s.ts);if(!p)continue;const d=Math.hypot(p.x-ball.cx,p.y-ball.cy);if(d<best){best=d;nearest={t,p,d}}}
    s.possessor=nearest&&best<Math.max(28,ball.w*2.2)?nearest.t.id:null;
  }
  // possession/carry/pass candidates
  let prev=null,seq=[];
  for(const s of samples){if(s.possessor!==prev){if(prev!==null){const tr=state.tracks.get(prev);const end=state.tracks.get(s.possessor||-1);if(end&&tr){const a=lastPositionAt(tr,s.ts),b=lastPositionAt(end,s.ts);if(a&&b){const d=dist(a,b);if(d>15){actions.push(makeAction(actionId++,'PASS_CANDIDATE',tr,end,s.ts,d,a,b,s.ball,.42));}else{actions.push(makeAction(actionId++,'POSSESSION_CHANGE',tr,end,s.ts,d,a,b,s.ball,.30));}}}}
      seq=[]; prev=s.possessor;
    }
    if(s.possessor!==null)seq.push(s);
    if(seq.length>=4){const tr=state.tracks.get(s.possessor);const first=seq[0],last=seq[seq.length-1];if(tr){const p1=lastPositionAt(tr,first.ts),p2=lastPositionAt(tr,last.ts);if(p1&&p2&&dist(p1,p2)>6){const dur=Math.max(.1,last.ts-first.ts);actions.push(makeAction(actionId++,'CARRY_CANDIDATE',tr,null,last.ts,dur,p1,p2,s.ball,Math.min(.75,.35+dist(p1,p2)/35)));seq=[]}}}
  }
  // shot candidates: sudden ball movement towards either end while near a player.
  for(let i=2;i<samples.length;i++){const a=samples[i-2],b=samples[i-1],c=samples[i];if(!a.ball||!b.ball||!c.ball)continue;const v1={x:b.ball.cx-a.ball.cx,y:b.ball.cy-a.ball.cy},v2={x:c.ball.cx-b.ball.cx,y:c.ball.cy-b.ball.cy};const speed=Math.hypot(v2.x,v2.y);const change=Math.abs(Math.atan2(v2.y,v2.x)-Math.atan2(v1.y,v1.x));if(speed>35&&speed>Math.hypot(v1.x,v1.y)*1.5){const owner=c.possessor?state.tracks.get(c.possessor):null;if(owner){const from=lastPositionAt(owner,c.ts)||{x:c.ball.cx,y:c.ball.cy},end={x:c.ball.cx,y:c.ball.cy};const pm=project(from.x,from.y);const xg=heuristicXG(pm.x,pm.y);actions.push(makeAction(actionId++,'SHOT_CANDIDATE',owner,null,c.ts,speed,from,end,c.ball,.35+Math.min(.4,speed/150),{xg}));}}}
  return dedupeActions(actions);
}
function dedupeActions(xs){xs.sort((a,b)=>a.timestamp-b.timestamp);const out=[];for(const a of xs){const last=out[out.length-1];if(last&&last.player_id===a.player_id&&last.action_type===a.action_type&&Math.abs(last.timestamp-a.timestamp)<1.0)continue;out.push(a)}return out}
function makeAction(id,type,from,to,ts,metric,start,end,ball,confidence,extra={}){const p1=project(start.x,start.y),p2=project(end.x,end.y);return {action_id:id,timestamp:+ts.toFixed(3),frame:Math.round(ts*(state.videoMeta?.fps||25)),action_type:type,team:from?.team||'',player_id:from?`P${from.id}`:'',player_track_id:from?.id||'',receiver_player_id:to?`P${to.id}`:'',receiver_track_id:to?.id||'',start_x_m:+p1.x.toFixed(2),start_y_m:+p1.y.toFixed(2),end_x_m:+p2.x.toFixed(2),end_y_m:+p2.y.toFixed(2),ball_x_m:ball?+project(ball.cx,ball.cy).x.toFixed(2):'',ball_y_m:ball?+project(ball.cx,ball.cy).y.toFixed(2):'',distance_or_metric:+Number(metric||0).toFixed(2),outcome:'CANDIDATE',xg:extra.xg??'',xa:'',confidence:+clamp(confidence,0,1).toFixed(3),coordinate_confidence:+state.transform.confidence.toFixed(3),inference_source:'browser_yolo+geometry'} }
function project(x,y){return state.transform.apply(x,y)}
function heuristicXG(x,y){const goalX=PITCH_W;const gx=goalX-x;const gy=Math.abs(PITCH_H/2-y);const distM=Math.hypot(gx,gy);const ang=Math.atan2(7.32/2,Math.max(gx,1));return +clamp(sigmoid(2.3-0.095*distM+1.15*ang),.01,.8).toFixed(3)}
function interpTrack(h,ts){if(!h.length)return null;let best=h[0],bd=Math.abs(h[0].ts-ts);for(const p of h){const d=Math.abs(p.ts-ts);if(d<bd){bd=d;best=p}}return {x:best.x,y:best.y}}
function lastPositionAt(t,ts){const p=interpTrack(t.history,ts);return p?{x:p.x,y:p.y}:null}

function drawPitch(canvas){const ctx=canvas.getContext('2d'),w=canvas.width=canvas.clientWidth*2,h=canvas.height=canvas.clientWidth*2*(PITCH_H/PITCH_W);ctx.clearRect(0,0,w,h);ctx.fillStyle='#10351a';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(235,255,240,.8)';ctx.lineWidth=3;const sx=w/PITCH_W,sy=h/PITCH_H;ctx.strokeRect(2,2,w-4,h-4);ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();ctx.beginPath();ctx.arc(w/2,h/2,h*9.15/PITCH_H,0,Math.PI*2);ctx.stroke();ctx.strokeRect(0,h*13.84/PITCH_H,w*16.5/PITCH_W,h-2*h*13.84/PITCH_H);ctx.strokeRect(w-w*16.5/PITCH_W,h*13.84/PITCH_H,w*16.5/PITCH_W,h-2*h*13.84/PITCH_H);return {ctx,w,h,sx,sy}}
function plotPoint(ctx,x,y,sx,sy,r,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.arc(x*sx,y*sy,r,0,Math.PI*2);ctx.fill()}
function renderHeatmap(){const c=$('heatmapCanvas');const {ctx,w,h,sx,sy}=drawPitch(c);const bins=35,xb=Array.from({length:bins},()=>Array(bins).fill(0));for(const t of state.tracks.values())for(const p of t.history){const pp=project(p.x,p.y),ix=clamp(Math.floor(pp.x/PITCH_W*bins),0,bins-1),iy=clamp(Math.floor(pp.y/PITCH_H*bins),0,bins-1);xb[ix][iy]++}let max=1;xb.forEach(r=>r.forEach(v=>max=Math.max(max,v)));for(let ix=0;ix<bins;ix++)for(let iy=0;iy<bins;iy++){const v=xb[ix][iy]/max;if(v){ctx.fillStyle=`rgba(255,120,60,${Math.min(.65,v*.65)})`;ctx.fillRect(ix*w/bins,iy*h/bins,w/bins+1,h/bins+1)}}}
function renderPassMap(){const c=$('passMapCanvas');const {ctx,w,h,sx,sy}=drawPitch(c);for(const a of state.actions.filter(x=>x.action_type.includes('PASS'))){ctx.strokeStyle='rgba(98,217,155,.72)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.start_x_m*sx,a.start_y_m*sy);ctx.lineTo(a.end_x_m*sx,a.end_y_m*sy);ctx.stroke();plotPoint(ctx,a.end_x_m,a.end_y_m,sx,sy,4,'#62d99b')}}
function renderShotMap(){const c=$('shotMapCanvas');const {ctx,w,h,sx,sy}=drawPitch(c);for(const a of state.actions.filter(x=>x.action_type.includes('SHOT'))){const x=a.start_x_m,y=a.start_y_m;plotPoint(ctx,x,y,sx,sy,8,'#ff6b6b');ctx.fillStyle='#fff';ctx.font='10px system-ui';ctx.fillText(a.xg=== ''?'':`xG ${a.xg}`,x*sx+9,y*sy-8)}}
function renderVoronoi(){const c=$('voronoiCanvas');const {ctx,w,h,sx,sy}=drawPitch(c);const latest=[...state.tracks.values()].map(t=>({t,p:t.history.at(-1)})).filter(o=>o.p);for(let gx=0;gx<53;gx++)for(let gy=0;gy<34;gy++){const x=(gx+.5)*PITCH_W/53,y=(gy+.5)*PITCH_H/34;let best=null,bd=1e9;for(const o of latest){const pp=project(o.p.x,o.p.y),d=Math.hypot(pp.x-x,pp.y-y);if(d<bd){bd=d;best=o.t}}if(best){ctx.fillStyle=best.team==='TEAM_B'?'rgba(99,166,255,.22)':'rgba(98,217,155,.22)';ctx.fillRect(gx*w/53,gy*h/34,w/53+1,h/34+1)}}for(const o of latest){const pp=project(o.p.x,o.p.y);plotPoint(ctx,pp.x,pp.y,sx,sy,5,o.t.team==='TEAM_B'?'#63a6ff':'#62d99b')}}

function summarize(){const all=state.actions;const shots=all.filter(a=>a.action_type.includes('SHOT')),passes=all.filter(a=>a.action_type.includes('PASS')),carries=all.filter(a=>a.action_type.includes('CARRY'));const xg=shots.reduce((s,a)=>s+Number(a.xg||0),0);const players=[...state.tracks.values()].filter(t=>!t.expired);$('summaryCards').innerHTML=[['Acciones',all.length],['Pases candidatos',passes.length],['Tiros candidatos',shots.length],['xG heurístico',xg.toFixed(2)]].map(([l,v])=>`<div class="metric"><div class="label">${l.toUpperCase()}</div><div class="value">${v}</div></div>`).join('');
  const rows=all.slice(0,500).map(a=>`<tr><td>${a.timestamp.toFixed(1)}</td><td>${a.action_type.replace('_CANDIDATE','')}</td><td>${a.team}</td><td>${a.player_id}</td><td>${a.receiver_player_id}</td><td>${a.start_x_m}, ${a.start_y_m}</td><td>${a.end_x_m}, ${a.end_y_m}</td><td>${a.xg}</td><td>${a.confidence}</td></tr>`).join('');$('actionsTable').innerHTML=`<thead><tr><th>s</th><th>acción</th><th>equipo</th><th>jugador</th><th>receptor</th><th>inicio (m)</th><th>fin (m)</th><th>xG</th><th>conf.</th></tr></thead><tbody>${rows}</tbody>`;$('actionCount').textContent=`${all.length} filas`;
  const prows=players.map(t=>{
    const h=t.history;
    let meters=0;
    for(let i=1;i<h.length;i++){
      const p1=project(h[i-1].x,h[i-1].y);
      const p2=project(h[i].x,h[i].y);
      meters+=Math.hypot(p2.x-p1.x,p2.y-p1.y);
    }
    let maxV=0;
    if(h.length>1){
      for(let i=1;i<h.length;i++){
        const a=project(h[i-1].x,h[i-1].y);
        const b=project(h[i].x,h[i].y);
        const dt=Math.max(.1,h[i].ts-h[i-1].ts);
        maxV=Math.max(maxV,Math.hypot(b.x-a.x,b.y-a.y)/dt*3.6);
      }
    }
    return `<tr><td>P${t.id}</td><td>${t.team||'?'}</td><td>${meters.toFixed(1)}</td><td>${maxV.toFixed(1)}</td><td>${h.length}</td></tr>`;
  }).join('');
  const schema=['action_id','timestamp','frame','action_type','team','player_id','player_track_id','receiver_player_id','receiver_track_id','start_x_m','start_y_m','end_x_m','end_y_m','ball_x_m','ball_y_m','distance_or_metric','outcome','xg','xa','confidence','coordinate_confidence','inference_source'];$('schema').innerHTML=schema.map(x=>`<span>${x}</span>`).join('');
}

function download(name,text,mime){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:mime}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
const csvEscape=v=>{const s=String(v??'');return /[",\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s};
function actionsCsv(){if(!state.actions.length)return '';const keys=Object.keys(state.actions[0]);return [keys.join(','),...state.actions.map(r=>keys.map(k=>csvEscape(r[k])).join(','))].join('\n')}
$('downloadCsv').onclick=()=>download('football_vision_actions.csv',actionsCsv(),'text/csv;charset=utf-8');
$('downloadTracks').onclick=()=>{const rows=[];for(const t of state.tracks.values())for(const p of t.history){const q=project(p.x,p.y);rows.push({timestamp:p.ts.toFixed(3),player_id:`P${t.id}`,track_id:t.id,team:t.team||'',pixel_x:p.x.toFixed(2),pixel_y:p.y.toFixed(2),pitch_x_m:q.x.toFixed(2),pitch_y_m:q.y.toFixed(2)})}const keys=Object.keys(rows[0]||{timestamp:'',player_id:'',track_id:'',team:'',pixel_x:'',pixel_y:'',pitch_x_m:'',pitch_y_m:''});download('football_vision_tracking.csv',[keys.join(','),...rows.map(r=>keys.map(k=>csvEscape(r[k])).join(','))].join('\n'),'text/csv;charset=utf-8')};
$('downloadJson').onclick=()=>download('football_vision_results.json',JSON.stringify(state.json,null,2),'application/json;charset=utf-8');

async function analyze(){
  const video=$('video'); await new Promise(r=>video.readyState>=1?r():video.addEventListener('loadedmetadata',r,{once:true}));
  state.videoMeta={fps:25,duration:video.duration,width:video.videoWidth,height:video.videoHeight}; $('videoMeta').textContent=`${video.videoWidth}×${video.videoHeight} · ${video.duration.toFixed(1)} s`;
  state.tracks.clear();state.nextTrack=1;state.actions=[];state.samples=[];state.transform=buildTransform(video.videoWidth,video.videoHeight);
  const session=await loadModel(); const sampleFps=Number($('sampleFps').value), conf=Number($('conf').value), imgSize=Number($('imgSize').value);
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const total=Math.max(1,Math.ceil(video.duration*sampleFps));
  const step=1/sampleFps;
  for(let i=0;i<total;i++){
    const ts=Math.min(video.duration-.001,i*step);video.currentTime=ts;await new Promise(r=>video.addEventListener('seeked',r,{once:true}));ctx.drawImage(video,0,0,canvas.width,canvas.height);
    const small=document.createElement('canvas');small.width=imgSize;small.height=Math.round(imgSize*video.videoHeight/video.videoWidth);small.getContext('2d').drawImage(video,0,0,small.width,small.height);
    const meta=letterbox(small,MODEL_INPUT,MODEL_INPUT);const out=await session.run({images:meta.tensor});const output=out[Object.keys(out)[0]];const dets=parseYolo(output,small.width,small.height,meta,conf);
    const players=dets.filter(d=>d.classId===COCO_PERSON);const balls=dets.filter(d=>d.classId===COCO_SPORTS_BALL);assignTracks(players,ts,video);
    const ball=balls.sort((a,b)=>b.score-a.score)[0]||null;state.samples.push({ts,ball,players:players.length});
    if(i%Math.max(1,Math.floor(sampleFps/2))===0){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(video,0,0);drawDetections(ctx,dets);$('annotatedCanvas').width=canvas.width;$('annotatedCanvas').height=canvas.height;$('annotatedCanvas').getContext('2d').drawImage(canvas,0,0);$('frameMeta').textContent=`${i+1}/${total}`;}
    setProgress(5+(i/total)*80,`Procesando vídeo · frame ${i+1}/${total}`);
  }
  finalizeTeams();setProgress(87,'Infiriendo acciones…');state.actions=actionInference(state.samples);setProgress(94,'Generando mapas…');renderHeatmap();renderPassMap();renderShotMap();renderVoronoi();summarize();state.json={created_at:new Date().toISOString(),video:{name:state.file.name,duration:video.duration,width:video.videoWidth,height:video.videoHeight,sample_fps:sampleFps},coordinate_system:{width_m:PITCH_W,height_m:PITCH_H,transform:state.transform.type,confidence:state.transform.confidence},actions:state.actions,players:[...state.tracks.values()].map(t=>({player_id:`P${t.id}`,team:t.team,track_id:t.id,samples:t.history.length}))};$('dashboard').classList.remove('hidden');setProgress(100,'Análisis terminado');setStatus('Análisis terminado');
}
function drawDetections(ctx,dets){for(const d of dets){ctx.strokeStyle=d.classId===COCO_SPORTS_BALL?'#ffd23f':'#62d99b';ctx.lineWidth=2;ctx.strokeRect(d.x1,d.y1,d.w,d.h);ctx.fillStyle='#fff';ctx.font='12px system-ui';ctx.fillText(d.classId===COCO_SPORTS_BALL?'BALL':`P ${Math.round(d.score*100)}%`,d.x1,d.y1-3)}}
$('runBtn').onclick=async()=>{ $('runBtn').disabled=true; try{await analyze()}catch(e){console.error(e);setStatus('Error',false);$('warningBox').textContent=`Error: ${e.message}. Para el modelo YOLO, revisa la conexión a internet y el tamaño del vídeo.`; }finally{$('runBtn').disabled=false}};
