import { useState, useRef, useCallback, useEffect } from "react";
import * as tf from "tensorflow";

const MODEL_URLS = {
  unet: "https://sonoscanai.github.io/lesion-detection-models/models/unet/model.json",
  classifier: "https://sonoscanai.github.io/lesion-detection-models/models/classifier/model.json",
};

const IMG_SIZE = 128;
const MASK_THRESHOLD = 0.5;
const MIN_REGION_RATIO = 0.003;
const VESSEL_RATIO = 0.019;

const DEFAULT_CONFIG = {
  radiologistEmail: "radiologist@hospital.org",
  radiologistName: "Dr. Smith",
  senderEmail: "clinic-app@hospital.org",
  zoomMeetingUrl: "https://zoom.us/j/1234567890",
  zoomMeetingId: "123 456 7890",
  zoomMeetingPass: "",
};

/* ─── Design tokens ─── */
const T = {
  bg: "#060b18", surface: "#0c1225", panel: "#111a33",
  border: "#1a2744", borderHi: "#243660",
  accent: "#00e5c3", accentDim: "#009e87",
  call: "#3584f7", callHi: "#1d6ce0",
  email: "#f0a020", emailHi: "#c98410",
  danger: "#f04444", success: "#34d399",
  text: "#dfe6f0", muted: "#687896", faint: "#2e3f5e",
  mono: "'JetBrains Mono','Fira Code','SF Mono',monospace",
  sans: "'Manrope','Outfit',system-ui,sans-serif",
};

/* ═══════════════════════════════════════════════
   Image processing (mirrors Python pipeline)
   ═══════════════════════════════════════════════ */
function imageToCanvas(img, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  return c;
}
function getImageData(canvas) {
  return canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
}
function sharpenImageData(imgData) {
  const { width: w, height: h, data } = imgData;
  const out = new Uint8ClampedArray(data.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        const idx = (y * w + x) * 4 + c;
        let val = 32 * data[idx];
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            if (dy !== 0 || dx !== 0)
              val -= 2 * data[((y + dy) * w + (x + dx)) * 4 + c];
        out[idx] = Math.min(255, Math.max(0, Math.round(val / 16)));
      }
      out[(y * w + x) * 4 + 3] = 255;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let c = 0; c < 4; c++) {
      out[x * 4 + c] = data[x * 4 + c];
      out[((h - 1) * w + x) * 4 + c] = data[((h - 1) * w + x) * 4 + c];
    }
  }
  for (let y = 0; y < h; y++) {
    for (let c = 0; c < 4; c++) {
      out[(y * w) * 4 + c] = data[(y * w) * 4 + c];
      out[(y * w + w - 1) * 4 + c] = data[(y * w + w - 1) * 4 + c];
    }
  }
  return new ImageData(out, w, h);
}
function enhanceContrast(imgData, factor) {
  const { data } = imgData;
  let sum = 0, count = data.length / 4 * 3;
  for (let i = 0; i < data.length; i += 4)
    sum += data[i] + data[i + 1] + data[i + 2];
  const mean = sum / count;
  for (let i = 0; i < data.length; i += 4)
    for (let c = 0; c < 3; c++)
      data[i + c] = Math.min(255, Math.max(0, Math.round(mean + factor * (data[i + c] - mean))));
  return imgData;
}
function imageDataToTensor(imgData) {
  const { width: w, height: h, data } = imgData;
  const f = new Float32Array(w * h * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    f[j] = data[i] / 255; f[j + 1] = data[i + 1] / 255; f[j + 2] = data[i + 2] / 255;
  }
  return tf.tensor4d(f, [1, h, w, 3]);
}

/* Connected components (8-connected BFS) */
function labelConnectedComponents(mask, w, h) {
  const labels = new Int32Array(w * h);
  let cur = 0;
  const regions = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (mask[idx] === 0 || labels[idx] !== 0) continue;
      cur++;
      let minX = x, maxX = x, minY = y, maxY = y, area = 0;
      const q = [[x, y]]; labels[idx] = cur;
      while (q.length) {
        const [cx, cy] = q.shift(); area++;
        if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]) {
          const nx = cx + dx, ny = cy + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            const ni = ny * w + nx;
            if (mask[ni] === 1 && labels[ni] === 0) { labels[ni] = cur; q.push([nx, ny]); }
          }
        }
      }
      regions.push({ label: cur, minX, minY, maxX: maxX + 1, maxY: maxY + 1, area });
    }
  }
  return regions;
}

/* ═══════════════════════════════════════════════
   Full inference pipeline
   ═══════════════════════════════════════════════ */
async function runInference(unet, classifier, imageEl, origW, origH) {
  const small = imageToCanvas(imageEl, IMG_SIZE, IMG_SIZE);
  let imgData = getImageData(small);
  imgData = sharpenImageData(imgData);
  imgData = enhanceContrast(imgData, 1.5);
  const inputTensor = imageDataToTensor(imgData);

  const predTensor = unet.predict(inputTensor);
  const predData = await predTensor.data();
  tf.dispose([inputTensor, predTensor]);

  const smallMask = new Uint8Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < predData.length; i++)
    smallMask[i] = predData[i] > MASK_THRESHOLD ? 1 : 0;

  const mask = new Uint8Array(origW * origH);
  const sx = IMG_SIZE / origW, sy = IMG_SIZE / origH;
  for (let y = 0; y < origH; y++)
    for (let x = 0; x < origW; x++)
      mask[y * origW + x] = smallMask[Math.floor(y * sy) * IMG_SIZE + Math.floor(x * sx)];

  const regions = labelConnectedComponents(mask, origW, origH);
  const imageArea = origW * origH;
  const results = [];

  for (const r of regions) {
    if (r.area < MIN_REGION_RATIO * imageArea) continue;
    if (r.area < VESSEL_RATIO * imageArea) {
      results.push({ ...r, classification: "Blood Vessel", confidence: 0 });
      continue;
    }
    const cc = document.createElement("canvas");
    cc.width = IMG_SIZE; cc.height = IMG_SIZE;
    cc.getContext("2d").drawImage(imageEl, r.minX, r.minY, r.maxX - r.minX, r.maxY - r.minY, 0, 0, IMG_SIZE, IMG_SIZE);
    const ct = imageDataToTensor(getImageData(cc));
    const cp = classifier.predict(ct);
    const cd = await cp.data();
    tf.dispose([ct, cp]);

    let classIdx, confidence;
    if (Math.abs(cd[0] - 0.5) < 1e-4 && Math.abs(cd[1] - 0.5) < 1e-4) {
      classIdx = 0; confidence = 50;
    } else {
      classIdx = cd[1] > cd[0] ? 1 : 0;
      confidence = Math.round(cd[classIdx] * 100);
    }
    results.push({ ...r, classification: classIdx === 0 ? "Benign" : "Malignant", confidence });
  }
  return { mask, regions: results };
}

/* ═══════════════════════════════════════════════
   UI Components
   ═══════════════════════════════════════════════ */
function Dot({ color, pulse }) {
  return <span style={{
    display: "inline-block", width: 7, height: 7, borderRadius: "50%",
    background: color, boxShadow: `0 0 8px ${color}80`, marginRight: 8,
    animation: pulse ? "dotPulse 1.8s ease-in-out infinite" : "none",
  }} />;
}
function Badge({ children, color }) {
  return <span style={{
    fontSize: 10, fontWeight: 700, fontFamily: T.mono, letterSpacing: ".06em",
    color, border: `1px solid ${color}40`, borderRadius: 4, padding: "2px 7px", background: `${color}10`,
  }}>{children}</span>;
}
function SectionLabel({ children }) {
  return <div style={{
    fontSize: 9.5, fontWeight: 700, letterSpacing: ".12em",
    textTransform: "uppercase", color: T.muted, padding: "0 0 6px", fontFamily: T.mono,
  }}>{children}</div>;
}
function Divider() { return <div style={{ height: 1, background: T.border, margin: "10px 0" }} />; }
function ScanOverlay({ active }) {
  if (!active) return null;
  return <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", borderRadius: 6, zIndex: 10 }}>
    <div style={{
      position: "absolute", left: 0, right: 0, height: 2,
      background: `linear-gradient(90deg,transparent,${T.accent},transparent)`,
      boxShadow: `0 0 30px ${T.accent}90`, animation: "scanY 2.2s ease-in-out infinite",
    }} />
  </div>;
}

function ModelIndicator({ label, state }) {
  const colors = { idle: T.muted, loading: T.email, ready: T.success, error: T.danger };
  const labels = { idle: "NOT LOADED", loading: "LOADING…", ready: "READY", error: "ERROR" };
  const c = colors[state];
  return <div style={{
    display: "flex", alignItems: "center", gap: 6, padding: "5px 8px",
    borderRadius: 4, background: T.surface, border: `1px solid ${T.border}`,
    fontSize: 10, fontFamily: T.mono, flex: 1,
  }}>
    <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}60`, flexShrink: 0,
      animation: state === "loading" ? "dotPulse 1s ease-in-out infinite" : "none" }} />
    <span style={{ color: T.text, fontSize: 10 }}>{label}</span>
    <span style={{ marginLeft: "auto", color: c, fontSize: 8, fontWeight: 700 }}>{labels[state]}</span>
  </div>;
}

/* ═══════════════════════════════════════════════
   Main App
   ═══════════════════════════════════════════════ */
export default function LesionAnalysisApp() {
  const [unet, setUnet] = useState(null);
  const [classifier, setClassifier] = useState(null);
  const [unetState, setUnetState] = useState("idle");     // idle | loading | ready | error
  const [clsState, setClsState] = useState("idle");
  const [image, setImage] = useState(null);
  const [imgEl, setImgEl] = useState(null);
  const [fileName, setFileName] = useState("");
  const [status, setStatus] = useState("Initializing…");
  const [statusColor, setStatusColor] = useState(T.muted);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [showSettings, setShowSettings] = useState(false);
  const [loadMode, setLoadMode] = useState("auto");  // "auto" (from URL) or "manual" (file picker)
  const canvasRef = useRef(null);
  const imgFileRef = useRef(null);
  const unetFileRef = useRef(null);
  const clsFileRef = useRef(null);

  /* ── Auto-load models from GitHub on mount ── */
  useEffect(() => {
    if (loadMode !== "auto") return;
    const isPlaceholder = MODEL_URLS.unet.includes("<username>");
    if (isPlaceholder) {
      setStatus("Configure model URLs or load models manually");
      setStatusColor(T.email);
      setLoadMode("manual");
      return;
    }

    let cancelled = false;
    async function load() {
      // Load U-Net
      setUnetState("loading");
      setStatus("Downloading U-Net from GitHub…");
      setStatusColor(T.email);
      try {
        const u = await tf.loadLayersModel(MODEL_URLS.unet);
        if (cancelled) return;
        setUnet(u);
        setUnetState("ready");
      } catch (err) {
        if (cancelled) return;
        console.error("U-Net load failed:", err);
        setUnetState("error");
        setStatus(`U-Net download failed: ${err.message}`);
        setStatusColor(T.danger);
        return;
      }

      // Load Classifier
      setClsState("loading");
      setStatus("Downloading Classifier from GitHub…");
      try {
        const c = await tf.loadLayersModel(MODEL_URLS.classifier);
        if (cancelled) return;
        setClassifier(c);
        setClsState("ready");
        setStatus("Both models loaded — upload an image");
        setStatusColor(T.success);
      } catch (err) {
        if (cancelled) return;
        console.error("Classifier load failed:", err);
        setClsState("error");
        setStatus(`Classifier download failed: ${err.message}`);
        setStatusColor(T.danger);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [loadMode]);

  /* ── Manual file-based model loading ── */
  const loadModelFiles = useCallback(async (files, setter, stateSetter, name) => {
    if (!files?.length) return;
    const arr = Array.from(files);
    const json = arr.find(f => f.name.endsWith(".json"));
    if (!json) { alert(`No model.json found. Select all files from the ${name} folder.`); return; }
    const bins = arr.filter(f => f.name.endsWith(".bin")).sort((a, b) => a.name.localeCompare(b.name));
    stateSetter("loading");
    setStatus(`Loading ${name} from files…`);
    setStatusColor(T.email);
    try {
      const model = await tf.loadLayersModel(tf.io.browserFiles([json, ...bins]));
      setter(model);
      stateSetter("ready");
      setStatus(`${name} loaded successfully`);
      setStatusColor(T.success);
    } catch (err) {
      console.error(err);
      stateSetter("error");
      setStatus(`Failed to load ${name}: ${err.message}`);
      setStatusColor(T.danger);
    }
  }, []);

  /* ── Upload image ── */
  const onUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setResults(null); setElapsed(null);
    const url = URL.createObjectURL(file);
    setImage(url);
    const img = new window.Image();
    img.onload = () => { setImgEl(img); setStatus("Image loaded — ready to analyze"); setStatusColor(T.accent); };
    img.src = url;
  }, []);

  /* ── Draw canvas ── */
  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs || !imgEl) return;
    const ctx = cvs.getContext("2d");
    const box = cvs.parentElement;
    const cw = box.clientWidth, ch = box.clientHeight;
    const scale = Math.min(cw / imgEl.width, ch / imgEl.height);
    const dw = Math.floor(imgEl.width * scale), dh = Math.floor(imgEl.height * scale);
    cvs.width = cw; cvs.height = ch;
    ctx.clearRect(0, 0, cw, ch);
    const ox = Math.floor((cw - dw) / 2), oy = Math.floor((ch - dh) / 2);

    if (results?.mask) {
      ctx.drawImage(imgEl, ox, oy, dw, dh);
      const id = ctx.getImageData(ox, oy, dw, dh);
      const d = id.data;
      for (let py = 0; py < dh; py++)
        for (let px = 0; px < dw; px++) {
          const mx = Math.floor(px / scale), my = Math.floor(py / scale);
          if (mx < imgEl.width && my < imgEl.height && results.mask[my * imgEl.width + mx] === 1) {
            const i = (py * dw + px) * 4;
            d[i] = 0; d[i + 1] = 255; d[i + 2] = 0;
          }
        }
      ctx.putImageData(id, ox, oy);
    } else {
      ctx.drawImage(imgEl, ox, oy, dw, dh);
    }

    if (results?.regions) {
      results.regions.forEach((r) => {
        const rx = ox + r.minX * scale, ry = oy + r.minY * scale;
        const rw = (r.maxX - r.minX) * scale, rh = (r.maxY - r.minY) * scale;
        const mal = r.classification === "Malignant";
        const bv = r.classification === "Blood Vessel";
        const color = bv ? "#ffffffcc" : mal ? T.danger : T.success;

        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        const cl = Math.min(14, rw / 3, rh / 3);
        ctx.lineWidth = 3; ctx.strokeStyle = color;
        [[rx, ry, 1, 1], [rx + rw, ry, -1, 1], [rx, ry + rh, 1, -1], [rx + rw, ry + rh, -1, -1]]
          .forEach(([cx, cy, dx, dy]) => {
            ctx.beginPath(); ctx.moveTo(cx, cy + dy * cl); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx * cl, cy); ctx.stroke();
          });

        const label = bv ? "Blood Vessel — no lesion" : `${r.classification} · ${r.confidence}%`;
        ctx.font = `600 12px ${T.sans}`;
        const tw = ctx.measureText(label).width;
        const lx = rx, ly = ry - 26 > 0 ? ry - 26 : ry + 4;
        ctx.fillStyle = "#000000dd";
        ctx.beginPath(); ctx.roundRect(lx, ly, tw + 16, 22, 3); ctx.fill();
        ctx.fillStyle = color; ctx.fillText(label, lx + 8, ly + 15);
      });

      if (elapsed !== null) {
        const hud = `${elapsed.toFixed(2)}s · ${results.regions.length} region${results.regions.length !== 1 ? "s" : ""}`;
        ctx.font = `500 11px ${T.mono}`;
        const hw = ctx.measureText(hud).width;
        ctx.fillStyle = "#000000bb";
        ctx.beginPath(); ctx.roundRect(ox + 10, oy + 10, hw + 18, 22, 3); ctx.fill();
        ctx.fillStyle = T.email; ctx.fillText(hud, ox + 19, oy + 25);
      }
    }
  }, [imgEl, results, elapsed]);

  /* ── Analyze ── */
  const onAnalyze = useCallback(async () => {
    if (!imgEl || !unet || !classifier || running) return;
    setRunning(true); setResults(null);
    setStatus("Running U-Net segmentation + classification…");
    setStatusColor(T.email);
    const t0 = performance.now();
    try {
      await new Promise(r => setTimeout(r, 50));
      const res = await runInference(unet, classifier, imgEl, imgEl.width, imgEl.height);
      const dt = (performance.now() - t0) / 1000;
      setElapsed(dt); setResults(res);
      setStatus(res.regions.length === 0 ? "Complete — no lesions detected" : `Complete — ${res.regions.length} region(s) identified`);
      setStatusColor(T.success);
    } catch (err) {
      console.error(err);
      setStatus(`Inference error: ${err.message}`); setStatusColor(T.danger);
    } finally { setRunning(false); }
  }, [imgEl, unet, classifier, running]);

  const onSave = useCallback(() => {
    const cvs = canvasRef.current; if (!cvs || !results) return;
    const a = document.createElement("a");
    a.download = `lesion_analysis_${Date.now()}.png`; a.href = cvs.toDataURL("image/png"); a.click();
  }, [results]);

  const onClear = useCallback(() => {
    setImage(null); setImgEl(null); setFileName(""); setResults(null); setElapsed(null);
    setStatus(unet && classifier ? "Models loaded — upload an image" : "Load models to begin");
    setStatusColor(T.muted);
    if (imgFileRef.current) imgFileRef.current.value = "";
    const cvs = canvasRef.current; if (cvs) cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height);
  }, [unet, classifier]);

  const onZoom = useCallback(() => {
    const p = config.zoomMeetingPass ? `\n  Passcode: ${config.zoomMeetingPass}` : "";
    if (!window.confirm(`Open Zoom with:\n\n  ${config.radiologistName}\n  ID: ${config.zoomMeetingId}${p}\n\nContinue?`)) return;
    let url = config.zoomMeetingUrl;
    if (config.zoomMeetingPass) url += `${url.includes("?") ? "&" : "?"}pwd=${config.zoomMeetingPass}`;
    window.open(url, "_blank");
    setStatus(`Zoom opened → ${config.radiologistName}`); setStatusColor(T.call);
  }, [config]);

  const onEmail = useCallback(() => {
    if (!image) { alert("Upload an image first."); return; }
    if (!window.confirm(`Email image to ${config.radiologistName} <${config.radiologistEmail}>?`)) return;
    const s = encodeURIComponent("Image Interpretation Request — Ultrasound Lesion Analysis");
    const b = encodeURIComponent(`Dear ${config.radiologistName},\n\nPlease interpret the attached ultrasound image.\n\nFile: ${fileName}\n\nThank you.`);
    window.open(`mailto:${config.radiologistEmail}?subject=${s}&body=${b}`, "_self");
    setStatus(`Email client opened → ${config.radiologistEmail}`); setStatusColor(T.email);
  }, [config, image, fileName]);

  const modelsReady = !!unet && !!classifier;
  const canAnalyze = modelsReady && !!imgEl && !running;

  return (
    <div style={{ fontFamily: T.sans, background: T.bg, color: T.text, height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes scanY{0%{top:0%}50%{top:100%}100%{top:0%}}
        @keyframes dotPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.75)}}
        @keyframes slideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes barGrow{from{width:0%}to{width:100%}}
        .btn{border:none;cursor:pointer;font-family:${T.sans};transition:all .18s;outline:none}
        .btn:disabled{opacity:.35;cursor:not-allowed}
        .btn:active:not(:disabled){transform:scale(.97)}
        .sbtn{width:100%;text-align:left;padding:10px 14px;border-radius:6px;font-size:13px;font-weight:600}
        .sbtn:hover:not(:disabled){filter:brightness(1.12)}
        input[type=text]{background:${T.surface};color:${T.text};border:1px solid ${T.border};border-radius:5px;padding:7px 10px;font-size:12px;font-family:${T.mono};width:100%;box-sizing:border-box;outline:none}
        input[type=text]:focus{border-color:${T.accent}60}
      `}</style>

      {/* Hidden file inputs */}
      <input ref={unetFileRef} type="file" multiple accept=".json,.bin" style={{ display: "none" }}
        onChange={(e) => loadModelFiles(e.target.files, setUnet, setUnetState, "U-Net")} />
      <input ref={clsFileRef} type="file" multiple accept=".json,.bin" style={{ display: "none" }}
        onChange={(e) => loadModelFiles(e.target.files, setClassifier, setClsState, "Classifier")} />
      <input ref={imgFileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onUpload} />

      {/* ═══ Header ═══ */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", borderBottom: `1px solid ${T.border}`,
        background: `linear-gradient(180deg,${T.panel}ee,${T.bg})`, flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 8,
            background: `linear-gradient(135deg,${T.accent},${T.call})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: T.bg,
          }}>U</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: "-.02em" }}>Ultrasound Lesion Analysis</div>
            <div style={{ fontSize: 10, color: T.muted, fontFamily: T.mono }}>
              TensorFlow.js · GitHub-Hosted Models · In-Browser Inference
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={() => setShowSettings(!showSettings)} style={{
            padding: "8px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
            background: showSettings ? T.borderHi : "transparent", color: T.muted, border: `1px solid ${T.border}`,
          }}>⚙</button>
          <button className="btn" onClick={onAnalyze} disabled={!canAnalyze} style={{
            padding: "9px 26px", borderRadius: 7, fontSize: 13, fontWeight: 700,
            background: canAnalyze ? `linear-gradient(135deg,${T.accent},${T.accentDim})` : T.faint,
            color: canAnalyze ? T.bg : T.muted,
            boxShadow: canAnalyze ? `0 2px 20px ${T.accent}35` : "none",
          }}>{running ? "Analyzing…" : "⬡ Analyze"}</button>
        </div>
      </header>

      {/* ═══ Body ═══ */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* ── Sidebar ── */}
        <aside style={{
          width: 270, minWidth: 270, flexShrink: 0, background: T.panel,
          borderRight: `1px solid ${T.border}`, display: "flex", flexDirection: "column",
          padding: "16px 16px 12px", gap: 4, overflowY: "auto",
        }}>
          {/* Model section */}
          <SectionLabel>Models</SectionLabel>
          <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
            <ModelIndicator label="U-Net" state={unetState} />
            <ModelIndicator label="Cls" state={clsState} />
          </div>

          {/* Loading bar when downloading */}
          {(unetState === "loading" || clsState === "loading") && (
            <div style={{ height: 3, background: T.surface, borderRadius: 2, overflow: "hidden", marginBottom: 4 }}>
              <div style={{ height: "100%", background: T.accent, borderRadius: 2, animation: "barGrow 2s ease-in-out infinite alternate" }} />
            </div>
          )}

          {/* Manual load buttons (shown if auto-load hasn't succeeded) */}
          {(!unet || !classifier) && (
            <>
              <div style={{ fontSize: 10, color: T.muted, fontFamily: T.mono, marginBottom: 4, lineHeight: 1.5 }}>
                {loadMode === "auto" && !MODEL_URLS.unet.includes("<username>")
                  ? "Auto-downloading from GitHub…"
                  : "Select converted tfjs model files:"}
              </div>
              {!unet && (
                <button className="btn sbtn" onClick={() => { setLoadMode("manual"); unetFileRef.current?.click(); }} style={{
                  background: "transparent", color: T.accent, border: `1.5px dashed ${T.accent}50`, marginBottom: 4,
                }}>↑ Load U-Net files</button>
              )}
              {!classifier && (
                <button className="btn sbtn" onClick={() => { setLoadMode("manual"); clsFileRef.current?.click(); }} style={{
                  background: "transparent", color: T.accent, border: `1.5px dashed ${T.accent}50`, marginBottom: 4,
                }}>↑ Load Classifier files</button>
              )}
            </>
          )}

          <Divider />

          {/* Image section */}
          <SectionLabel>Image</SectionLabel>
          <button className="btn sbtn" onClick={() => imgFileRef.current?.click()} disabled={!modelsReady} style={{
            background: "transparent", color: modelsReady ? T.accent : T.muted,
            border: `1.5px dashed ${modelsReady ? T.accent : T.muted}50`, marginBottom: 4,
          }}>↑ Upload Ultrasound Image</button>
          {fileName && <div style={{ fontSize: 11, color: T.text, fontFamily: T.mono, wordBreak: "break-all", lineHeight: 1.5 }}>{fileName}</div>}

          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <button className="btn" onClick={onSave} disabled={!results} style={{
              flex: 1, padding: "8px 0", borderRadius: 5, fontSize: 12, fontWeight: 600,
              background: T.surface, color: results ? T.text : T.muted, border: `1px solid ${T.border}`,
            }}>⤓ Save</button>
            <button className="btn" onClick={onClear} style={{
              flex: 1, padding: "8px 0", borderRadius: 5, fontSize: 12, fontWeight: 600,
              background: T.surface, color: T.muted, border: `1px solid ${T.border}`,
            }}>✕ Clear</button>
          </div>

          <Divider />

          <SectionLabel>Radiologist Services</SectionLabel>
          <button className="btn sbtn" onClick={onZoom} style={{ background: T.call, color: "#fff", borderRadius: 6, marginBottom: 4 }}>
            📹&ensp;Connect to Live Radiologist
          </button>
          <button className="btn sbtn" onClick={onEmail} style={{ background: T.email, color: "#fff", borderRadius: 6 }}>
            ✉&ensp;Request Image Interpretation
          </button>

          <Divider />

          <SectionLabel>Status</SectionLabel>
          <div style={{ display: "flex", alignItems: "center", fontSize: 12, lineHeight: 1.5, minHeight: 24 }}>
            <Dot color={statusColor} pulse={running || unetState === "loading" || clsState === "loading"} />
            <span>{status}</span>
          </div>

          {results?.regions?.length > 0 && <>
            <Divider />
            <SectionLabel>Findings</SectionLabel>
            {results.regions.map((r, i) => {
              const c = r.classification === "Blood Vessel" ? T.muted : r.classification === "Malignant" ? T.danger : T.success;
              return <div key={i} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "8px 10px",
                borderRadius: 6, marginTop: 4, background: T.surface, border: `1px solid ${c}25`,
                animation: `slideUp .3s ease ${i * .08}s both`,
              }}>
                <Dot color={c} />
                <span style={{ fontSize: 12, fontWeight: 600 }}>{r.classification}</span>
                {r.confidence > 0 && <Badge color={c}>{r.confidence}%</Badge>}
              </div>;
            })}
          </>}

          {showSettings && <>
            <Divider />
            <SectionLabel>Configuration</SectionLabel>
            {[["radiologistName", "Radiologist"], ["radiologistEmail", "Email"],
              ["zoomMeetingUrl", "Zoom URL"], ["zoomMeetingId", "Meeting ID"],
              ["zoomMeetingPass", "Passcode"], ["senderEmail", "Sender"]].map(([k, l]) => (
              <div key={k} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 10, color: T.muted, marginBottom: 3, fontFamily: T.mono }}>{l}</div>
                <input type="text" value={config[k]} onChange={(e) => setConfig(p => ({ ...p, [k]: e.target.value }))} />
              </div>
            ))}
          </>}
        </aside>

        {/* ── Main canvas ── */}
        <main style={{
          flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center",
          background: `radial-gradient(ellipse at 20% 30%,${T.accent}06,transparent 60%),radial-gradient(ellipse at 80% 70%,${T.call}04,transparent 60%),${T.bg}`,
          overflow: "hidden",
        }}>
          <div style={{
            position: "absolute", inset: 0, pointerEvents: "none", opacity: .35,
            backgroundImage: `linear-gradient(${T.faint}60 1px,transparent 1px),linear-gradient(90deg,${T.faint}60 1px,transparent 1px)`,
            backgroundSize: "28px 28px",
          }} />
          {imgEl ? (
            <div style={{ position: "relative", width: "100%", height: "100%", padding: 12 }}>
              <ScanOverlay active={running} />
              <canvas ref={canvasRef} style={{ width: "100%", height: "100%", borderRadius: 6, display: "block" }} />
            </div>
          ) : (
            <div style={{ textAlign: "center", zIndex: 1 }}>
              <div style={{
                width: 72, height: 72, borderRadius: 18, border: `2px dashed ${T.faint}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 18px", fontSize: 28, color: T.faint,
              }}>◎</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: T.muted }}>
                {modelsReady ? "Upload an ultrasound image to begin" : "Waiting for models…"}
              </div>
              <div style={{ fontSize: 11, marginTop: 8, color: T.faint, fontFamily: T.mono, maxWidth: 340, lineHeight: 1.7, margin: "8px auto 0" }}>
                {!modelsReady && "Models will auto-download from GitHub if URLs are configured, or use the sidebar to load files manually."}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
