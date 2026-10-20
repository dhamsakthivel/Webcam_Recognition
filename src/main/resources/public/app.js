const video = document.querySelector("#camera");
const canvas = document.querySelector("#frame");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const startButton = document.querySelector("#startCamera");
const classifyButton = document.querySelector("#classify");
const addButton = document.querySelector("#addSample");
const result = document.querySelector("#result");
const labelsEl = document.querySelector("#labels");
const countEl = document.querySelector("#modelCount");
const notice = document.querySelector("#sampleNotice");
let model = [];

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support camera access.");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  document.querySelector("#cameraState").hidden = true;
  startButton.textContent = "Camera ready";
  startButton.disabled = true;
  classifyButton.disabled = false; addButton.disabled = false;
}

// 3×3 mean RGB cells: simple, transparent and entirely local feature extraction.
function cameraFeature() {
  if (!video.videoWidth) throw new Error("Wait for the camera image.");
  canvas.width = 180; canvas.height = 135;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const values = [];
  for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
    let r = 0, g = 0, b = 0, n = 0;
    for (let y = row * 45; y < (row + 1) * 45; y++) for (let x = col * 60; x < (col + 1) * 60; x++) {
      const i = (y * 180 + x) * 4; r += pixels[i]; g += pixels[i + 1]; b += pixels[i + 2]; n++;
    }
    values.push(r / n, g / n, b / n);
  }
  return values;
}

function classify(feature) {
  const distances = Object.entries(model).map(([label, samples]) => {
    const closest = Math.min(...samples.map(sample => Math.sqrt(sample.reduce((sum, x, i) => sum + (x - feature[i]) ** 2, 0) / sample.length)));
    return { label, distance: closest };
  }).sort((a, b) => a.distance - b.distance);
  if (!distances.length) return null;
  const best = distances[0], next = distances[1];
  // Blend closeness and separation from competing labels into a conservative confidence.
  const closeness = Math.max(0, 1 - best.distance / 115);
  const separation = next ? Math.min(1, (next.distance - best.distance) / 45) : 0.75;
  return { ...best, confidence: Math.round(100 * (0.65 * closeness + 0.35 * separation)) };
}

async function loadModel() {
  const [info, savedModel] = await Promise.all([fetch("/api/labels").then(response => response.json()), fetch("/api/model").then(response => response.json())]);
  model = savedModel.samples.reduce((labels, sample) => {
    (labels[sample.label] ??= []).push(sample.feature.split(",").map(Number));
    return labels;
  }, {});
  countEl.textContent = `${info.labels.length} label${info.labels.length === 1 ? "" : "s"} · ${info.sampleCount} view${info.sampleCount === 1 ? "" : "s"}`;
  labelsEl.innerHTML = info.labels.length ? "" : '<p class="hint">No labels yet. Teach the model using the camera above.</p>';
  info.labels.forEach(label => {
    const card = document.createElement("div"); card.className = "label-card";
    card.innerHTML = `<span>${label}</span><button aria-label="Remove ${label}">Remove</button>`;
    card.querySelector("button").onclick = async () => { await fetch("/api/samples", { method: "DELETE", headers: {"Content-Type": "application/x-www-form-urlencoded"}, body: new URLSearchParams({label}) }); delete model[label]; await loadModel(); };
    labelsEl.append(card);
  });
}

startButton.onclick = async () => {
  try { await requestCamera(); } catch (error) { result.textContent = `Camera unavailable: ${error.message} Use localhost and allow the browser camera prompt.`; }
};
document.querySelector("#teachForm").onsubmit = async event => {
  event.preventDefault();
  try {
    const label = document.querySelector("#label").value.trim();
    const feature = cameraFeature();
    await fetch("/api/samples", { method: "POST", headers: {"Content-Type": "application/x-www-form-urlencoded"}, body: new URLSearchParams({ label, feature: feature.join(",") }) }).then(async response => { if (!response.ok) throw new Error((await response.json()).error); });
    (model[label] ??= []).push(feature);
    notice.textContent = `Saved a view of ${label}. Add several more views for a stronger model.`;
    await loadModel();
  } catch (error) { notice.textContent = error.message; }
};
classifyButton.onclick = () => {
  try {
    const prediction = classify(cameraFeature());
    if (!prediction) { result.textContent = "Teach at least one produce label before scanning."; return; }
    result.innerHTML = prediction.confidence >= 70 ? `<strong>${prediction.label}</strong> · ${prediction.confidence}% confidence` : `Not confident enough (${prediction.confidence}%). Add more varied samples or improve lighting.`;
  } catch (error) { result.textContent = error.message; }
};
loadModel();
