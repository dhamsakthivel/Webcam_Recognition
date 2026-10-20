# Fruit Vision — local webcam produce recognition

A small Java 21 application that runs a local browser interface for training and
recognizing fruits and vegetables. It uses the webcam through the browser's
standard `getUserMedia` API, so the user controls permission and can test it
without a cloud account or external image service.

## Run locally

Requirements: JDK 21 or newer. No Maven, Gradle, native webcam drivers, or
third-party dependencies are required.

```bash
javac -d out src/main/java/com/webcamrecognition/FruitVisionServer.java
java -cp out com.webcamrecognition.FruitVisionServer
```

Open [http://localhost:8080](http://localhost:8080), select **Enable camera**,
and approve the browser's camera prompt. The server is intentionally bound to
`127.0.0.1`, so it cannot be reached from another machine.

`localhost` is considered a trustworthy origin by current Chrome, Edge, and
Firefox releases even when it uses HTTP. Camera access may still be blocked in
an embedded browser, an incognito/enterprise policy, or after selecting “Block”
in the permission prompt. The response includes
`Permissions-Policy: camera=(self)` so the locally served page is permitted to
ask for a camera.

## Optional open datasets

The repository includes setup scripts for three image datasets. Downloads are
placed under the ignored `data/datasets/` directory and are never committed.
Review each dataset's license and terms before using it in a product.

On Windows PowerShell:

```powershell
.\scripts\fetch-datasets.ps1
```

On macOS/Linux:

```bash
chmod +x scripts/fetch-datasets.sh
./scripts/fetch-datasets.sh
```

Both scripts shallow-clone
[Fruits-360](https://github.com/fruits-360/fruits-360-100x100). They then
install the latest Kaggle CLI and download `moltean/fruits` and
`kritikseth/fruit-and-vegetable-image-recognition`. The Kaggle steps require a
valid API token in `~/.kaggle/kaggle.json`; credentials are not included in or
requested by this repository.

These datasets are training/evaluation sources, not automatically imported
into the browser model. The included classifier uses a small color-feature
prototype rather than a neural image model; bulk data needs a separately
trained, evaluated TensorFlow Lite or ONNX model before it can improve live
recognition.

## Training a label

1. Enter a fruit or vegetable name.
2. Hold one item prominently in the camera frame and save 8–15 views.
3. Include small changes in angle, distance, and normal lighting.
4. Scan the item using **Identify produce**.

Samples are saved locally in `data/produce-samples.tsv`. Delete a label from
the UI to remove its saved samples.

## Focused scanning and reference captures

While the camera is active, tap/click the product to move the scan guide and
use the scan-area slider to surround it. Fruit Vision loads MediaPipe's
self-segmentation model and excludes detected person pixels from that guide;
everything outside the accepted foreground is blurred. Classification extracts
the feature only from that selected guide, which reduces the influence of a
holder or background. Saving a camera view also writes the focused PNG locally
under `data/reference-images/` so it can be reviewed as a reference.

The MediaPipe runtime and model load from jsDelivr and Google Cloud Storage the
first time camera scanning is enabled. If they cannot be reached, the app
continues with its center-guided blur fallback. Selfie segmentation identifies
people rather than produce, so place one produce item prominently inside the
guide; it is not a general-purpose produce-instance segmentation model.

## Starter examples

The interface begins with red apple, banana, carrot, and broccoli labels. Each
has a small hand-selected baseline color profile and extracts three seed
features from an openly licensed Wikimedia Commons reference image at startup.
The card links show the photographer and license (CC BY 2.0, CC BY-SA 3.0,
public domain, and GFDL 1.2 respectively). Image pixels are processed in the
browser only; they are not uploaded to the local server. Add personal camera
views to a starter label to adapt it to the actual camera, backdrop, and
lighting.

## Recognition approach and limitations

The app extracts a transparent 3×3 grid of mean RGB values from each captured
frame and uses nearest-sample distance to classify it. This keeps all image
processing on the device and makes it easy to understand, but it is a
prototype—not a production object-detection model. The UI only accepts a
prediction at a 70% confidence threshold. That threshold does **not** promise
70% real-world accuracy; neither the starter color profiles nor open-source
image seed samples are sufficient to validate one. A 70% accuracy target requires a
held-out, representative test set for each added produce type.

For higher accuracy, replace the browser feature extractor with an exported
TensorFlow Lite/ONNX image classifier and evaluate it per class under the
intended camera and lighting conditions.
