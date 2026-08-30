/* U2-Net background removal in the browser with onnxruntime-web.
 * - Preprocess: resize to 320x320, RGB, float32, normalize (ImageNet), NCHW
 * - Run model: prefer WebGPU, fallback to WASM
 * - Postprocess: sigmoid (if needed), resize mask to image size, threshold + feather, apply alpha
 */

const MODEL_URL = "./models/model_ceil_zero.onnx"; // put your model here
const INPUT_SIZE = 320;
const imagenetMean = [0.485, 0.456, 0.406];
const imagenetStd  = [0.229, 0.224, 0.225];

const els = {
  file: document.getElementById("fileInput"),
  img: document.getElementById("inputImg"),
  out: document.getElementById("outCanvas"),
  thr: document.getElementById("thr"),
  thrVal: document.getElementById("thrVal"),
  blur: document.getElementById("blur"),
  blurVal: document.getElementById("blurVal"),
  ep: document.getElementById("ep"),
  run: document.getElementById("btnRun"),
  dl: document.getElementById("btnDownload"),
  status: document.getElementById("status"),
};

let session = null;
let inputName = null;

function setStatus(msg) { els.status.textContent = msg; }
function uiReady(ready) {
  els.run.disabled = !ready || !els.img.src;
  els.dl.disabled = !ready || !els.out.width;
}

async function createSession(epChoice) {
  // Choose execution provider
  let providers = ["wasm"];
  if (epChoice === "webgpu") providers = ["webgpu"];
  else if (epChoice === "auto") providers = (await ort.env.webgpu)? ["webgpu","wasm"] : ["wasm"];

  // Enable SIMD+threads for WASM
  //ort.env.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency || 2);
  if (window.crossOriginIsolated) {
    console.log("Page is cross-origin isolated");
    ort.env.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency || 2);
  } else {
    console.log("Page is NOT cross-origin isolated");
    ort.env.wasm.numThreads = 1; // <-- This will not work unless you enable crossOriginIsolated mode
  }
  ort.env.wasm.simd = true;

  setStatus(`Loading model (${providers.join(" - ")})`);
  //session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });
  //setStatus(`Model ready EP: ${session.executionProviders.join(", ")}`);

  // inside createSession
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });
  inputName = session.inputNames[0];
  // FIX: don't use session.executionProviders, it's undefined in web
  setStatus(`Model ready EP: ${providers.join(" -> ")}`);

  uiReady(!!session && !els.run.disabled);
}

function fileToImg(file) {
 return new Promise((resolve, reject) => {
   const url = URL.createObjectURL(file);
   const img = new Image();
   img.onload = () => (resolve(img));
   img.onerror = reject;
   img.src = url;
 });
}

async function fileToBitmap(file) {
  const blobUrl = URL.createObjectURL(file);
  const res = await fetch(blobUrl);
  const blob = await res.blob();
  URL.revokeObjectURL(blobUrl);
  return await createImageBitmap(blob);
}

function drawToCanvas(img, canvas) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
}

function preprocessFromImage(img) {
  // Resize to model size and normalize
  const tmp = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const tctx = tmp.getContext("2d");
  tctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = tctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  // NCHW: [1,3,320,320]
  let p = 0;
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const idx = (y * INPUT_SIZE + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;
      // normalize (ImageNet stats)
      chw[p] = (r - imagenetMean[0]) / imagenetStd[0];                 // R
      chw[p + INPUT_SIZE * INPUT_SIZE] = (g - imagenetMean[1]) / imagenetStd[1]; // G
      chw[p + 2 * INPUT_SIZE * INPUT_SIZE] = (b - imagenetMean[2]) / imagenetStd[2]; // B
      p++;
    }
  }
  const tensor = new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return tensor;
}

function needsSigmoid(min, max) {
  // If mask is already in [0,1], don�t re-sigmoid
  return !(min >= -0.05 && max <= 1.05);
}

function sigmoidArray(arr) {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = 1 / (1 + Math.exp(-arr[i]));
  return out;
}

function selectBestOutput(feeds) {
  // Pick the output with the most elements (commonly "d0" or "output")
  let bestName = session.outputNames[0];
  let bestLen = -1;
  for (const name of session.outputNames) {
    const v = feeds[name];
    const arr = v.data || v; // ort returns Tensor
    const len = (arr.data ? arr.data.length : arr.length) ?? 0;
    if (len > bestLen) { bestLen = len; bestName = name; }
  }
  return bestName;
}

function upsampleMaskToImage(mask320, imgW, imgH) {
  // mask320: Float32Array length 320*320 in [0,1]
  const maskCanvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const mctx = maskCanvas.getContext("2d");
  const imgData = mctx.createImageData(INPUT_SIZE, INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    const v = Math.max(0, Math.min(1, mask320[i]));
    const g = Math.round(v * 255);
    const j = i * 4;
    imgData.data[j] = g;
    imgData.data[j + 1] = g;
    imgData.data[j + 2] = g;
    imgData.data[j + 3] = 255;
  }
  mctx.putImageData(imgData, 0, 0);

  // Draw scaled to original size (bilinear)
  const up = new OffscreenCanvas(imgW, imgH);
  const uctx = up.getContext("2d");
  uctx.imageSmoothingEnabled = true;
  uctx.imageSmoothingQuality = "high";
  uctx.drawImage(maskCanvas, 0, 0, imgW, imgH);
  return up;
}

function applyMaskToCanvas(sourceImg, outCanvas, maskCanvas, threshold, blurPx) {
  outCanvas.width = sourceImg.naturalWidth;
  outCanvas.height = sourceImg.naturalHeight;
  const ctx = outCanvas.getContext("2d");

  // Optional feather on the mask
  const mctx = maskCanvas.getContext("2d");
  if (blurPx > 0) {
    mctx.filter = `blur(${blurPx}px)`;
    const blurred = new OffscreenCanvas(maskCanvas.width, maskCanvas.height);
    blurred.getContext("2d").drawImage(maskCanvas, 0, 0);
    maskCanvas = blurred;
    mctx.filter = "none";
  }

  // Draw original
  ctx.drawImage(sourceImg, 0, 0);

  // Multiply alpha by mask (threshold applied)
  const imgData = ctx.getImageData(0, 0, outCanvas.width, outCanvas.height);
  const maskData = maskCanvas.getContext("2d").getImageData(0, 0, outCanvas.width, outCanvas.height).data;

  for (let i = 0, p = 0; i < imgData.data.length; i += 4, p += 4) {
    // mask gray is in maskData[p] (0..255)
    const m = maskData[p] / 255;
    const a = m >= threshold ? m : 0;
    imgData.data[i + 3] = Math.round(imgData.data[i + 3] * a);
  }
  ctx.putImageData(imgData, 0, 0);
}

async function runOnce() {
  try {
    uiReady(false);
    setStatus("Preprocessing");
    const tensor = preprocessFromImage(els.img);

    setStatus("Running model ...");
    const outputs = await session.run({ [inputName]: tensor });

    // Choose best output tensor (handle models with multiple side outputs)
    const name = selectBestOutput(outputs);
    const outTensor = outputs[name];
    let data = outTensor.data; // Float32Array
    const [n, c, h, w] = outTensor.dims; // expect [1,1,320,320]

    // Flatten to 1D mask
    let mask = data;
    // Apply sigmoid if output not already in [0,1]
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < mask.length; i++) { const v = mask[i]; if (v < min) min = v; if (v > max) max = v; }
    if (needsSigmoid(min, max)) mask = sigmoidArray(mask);

    // Use channel 0 and reshape (h*w)
    const mask320 = new Float32Array(h * w);
    // data layout is NCHW, c==1
    for (let i = 0; i < h * w; i++) mask320[i] = mask[i];

    setStatus("Postprocessing ...");
    const upMask = upsampleMaskToImage(mask320, els.img.naturalWidth, els.img.naturalHeight);
    applyMaskToCanvas(els.img, els.out, upMask, parseFloat(els.thr.value), parseInt(els.blur.value, 10));

    els.dl.disabled = false;
    setStatus("Done");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
  } finally {
    uiReady(true);
  }
}

// 
// els.file.addEventListener("change", async (e) => {
//   const f = e.target.files?.[0];
//   if (!f) return;
//   const img = await fileToImg(f);
//   els.img.src = img.src;
//   els.img.onload = () => {
//     drawToCanvas(els.img, document.createElement("canvas")); // warm up
//     els.run.disabled = !session;
//     setStatus(session ? "Ready" : "Model loading�");
//   };
// });

els.file.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const img = await fileToImg(f);

  // Set onload handler before setting src
  els.img.onload = () => {
    drawToCanvas(els.img, document.createElement("canvas")); // warm up
    els.run.disabled = !session;
    setStatus(session ? "Ready" : "Model loading…");
  };
  els.img.onerror = () => {
    setStatus("Error: Failed to load image.");
    els.run.disabled = true;
  };
  els.img.src = img.src;

  // If image is already loaded (from cache), trigger onload manually
  if (els.img.complete && els.img.naturalWidth !== 0) {
    els.img.onload();
  }
});

els.run.addEventListener("click", runOnce);
els.dl.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "cutout.png";
  a.href = els.out.toDataURL("image/png");
  a.click();
});
els.thr.addEventListener("input", () => (els.thrVal.textContent = els.thr.value));
els.blur.addEventListener("input", () => (els.blurVal.textContent = els.blur.value));
els.ep.addEventListener("change", async () => {
  // Recreate session with new EP
  //els.run.disabled = true;
  await createSession(els.ep.value);
});

// Kick off model load
createSession("auto").catch((e) => setStatus("Model load failed: " + e.message));
