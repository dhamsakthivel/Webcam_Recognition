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

## Training a label

1. Enter a fruit or vegetable name.
2. Hold one item prominently in the camera frame and save 8–15 views.
3. Include small changes in angle, distance, and normal lighting.
4. Scan the item using **Identify produce**.

Samples are saved locally in `data/produce-samples.tsv`. Delete a label from
the UI to remove its saved samples.

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
