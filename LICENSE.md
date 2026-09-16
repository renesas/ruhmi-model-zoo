## RUHMI Model Zoo

BSD 3-Clause License

Copyright (c) 2026, Renesas Electronics

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Third-Party Models and Data Licenses

The repository source code is released under BSD-3-Clause (see `LICENSE`).
Third-party models and data/assets remain under their original licenses.

| Use Case | Model License | Sample Inputs Included in GitHub |
|----------|---------------|----------------------------------|
| MobileNetV1 (Image Classification) | Apache 2.0 (TensorFlow/Keras weights) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| MobileNetV2 (Image Classification) | Apache 2.0 (TensorFlow/Keras weights) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| MobileNetV3 (Image Classification) | Apache 2.0 (TensorFlow/Keras weights) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| ShuffleNetV2 (Image Classification) | BSD-3-Clause (torchvision weights) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| SqueezeNet 1.1 (Image Classification) | BSD-3-Clause (torchvision weights) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| ResNet8 (Image Classification) | Apache 2.0 (MLCommons Tiny) | CIFAR-10 sample images in `sample_images/` (dataset terms) |
| Visual Wake Words (Image Classification) | Apache 2.0 (MLCommons Tiny) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| YOLO-Fastest 1.1 (Object Detection) | MIT | COCO sample images in `sample_images/` (CC-BY 4.0) |
| YOLOX-Tiny (Object Detection) | Apache 2.0 (Megvii YOLOX) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| BlazeFace Front (Face Detection) | Apache 2.0 (MediaPipe/PINTO) |COCO sample images in `sample_images/` (CC-BY 4.0) |
| Keyword Spotting DS-CNN (Audio Classification) | Apache 2.0 (MLCommons Tiny) | Speech Commands sample audio in `sample_audio/` (CC-BY 4.0) |
| Auto Encoder (Anomaly Detection) | Apache 2.0 (MLCommons Tiny) | No sample audio included (DCASE 2020 CC-BY-SA 4.0; user-provided) |
| MCUNet (Image Classification) | MIT (MIT HAN Lab) | No sample images included (ImageNet Terms of Access prohibit redistribution; user-provided via [image-net.org](https://image-net.org/download.php)) |
| NanoDet-Plus-m (Object Detection) | Apache 2.0 (RangiLyu/nanodet) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| MobileFaceNet (Face Recognition) | Apache 2.0 (foamliu/MobileFaceNet) | No sample images included (LFW is research use only; user-provided) |
| MediaPipe Face Landmark (Facial Landmark) | Apache 2.0 (Google MediaPipe / patlevin/face-detection-tflite MIT) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| PoseNet MobileNetV1-0.5 (Pose Estimation) | Apache 2.0 (tensorflow/tfjs-models) | COCO sample images in `sample_images/` (CC-BY 4.0) |
| YAMNet (Audio Classification) | Apache 2.0 (Google / TensorFlow Hub) | AudioSet sample clips in `sample_audio/` (CC-BY 4.0) |
| RNNoise (Noise Suppression) | Apache 2.0 (Arm ML Model Zoo) | Edinburgh Noisy Speech sample audio in `sample_audio/` (CC-BY 4.0) |
| TinyWav2Letter (Speech Recognition) | Apache 2.0 (Arm ML-zoo) | No sample audio included (Fluent Speech Datasets is research use only; user-provided) |
| AD MicroNet Medium (Anomaly Detection) | Apache 2.0 (Arm ML-zoo) | No sample audio included (DCASE 2020 Task 2 Slider is CC-BY-SA 4.0; user-provided) |

> Verify current upstream license terms before redistribution in your target product, geography, and commercial context.
