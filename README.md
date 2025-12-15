# AprilTag Enhancement & Detection

This project evaluates how motion blur, low-light, and haze enhancement impacts AprilTag detection performance using different pipelines:

## Installation
Must install pupil april tag dependency in order to run baseline code.
```bash
pip install -r pupil-apriltags opencv-python numpy matplotlib torch torchvision
```
## Dataset
Dataset located in dataset folder, then pull all data from pi_cam folder. Contains all data for all conditions.

## Motion Blur
Simply run ```python motion_blur/pipeline.py``` from main directory while dataset exists in file structure. 
To run with motion blurred files only, change ```FILTER_PATTERN = ""``` to ```FILTER_PATTERN = "motion"``` within motion_blur/pipeline.py and rerun. 

## Dehazing
Run the file `dehaze/dehaze_on_data.py` to run a dehazing method on all of the data. Change which one you would like to use at the top of the file. Run `dehaze/dehaze.py` for an example on a singular image of the different methods. Run `dehaze/analyze_results.py` to see the results of the different dehazing implementations.


## Low Light
For Gamma + CLAHE + Retinex method download pupil_clahe.py. Simply run this code on the pi_cam dataset and should see the results.

For Adaptive Shadow Boost method download pupil_adaptive.py. Simply run this code on the pi_cam dataset and should see the results.

In order to run model on low light data only download and run pupil_adaptive_lowlightonly.py.

