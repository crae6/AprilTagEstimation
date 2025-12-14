# AprilTag Enhancement & Detection

This project evaluates how motion blur, low-light, and haze enhancement impacts AprilTag detection performance using different pipelines:

## Installation
Must install pupil april tag dependency in order to run baseline code.
```bash
pip install pupil-apriltags opencv-python numpy
```
## Dataset
Dataset located in dataset folder, then pull all data from pi_cam folder. Contains all data for all conditions.

## Motion Blur

## Dehazing

## Low Light
For Gamma + CLAHE + Retinex method download pupil_clahe.py. Simply run this code on the pi_cam dataset and should see the results.

For Adaptive Shadow Boost method download pupil_adaptive.py. Simply run this code on the pi_cam dataset and should see the results.

In order to run model on low light data only download and run pupil_adaptive_lowlightonly.py.

