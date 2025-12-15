from pupil_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

# Load as uint8 for the detector. Create a float copy only for plotting.
img_uint8 = imread("dataset/pi_cam/pi_cam_11.jpg", pilmode="L").astype(np.uint8)
img = img_uint8.astype(np.float32) / 255.0

# Configure the detector. Adjust parameters here if you want to trade speed/accuracy.
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

detections = detector.detect(
    img_uint8,
    estimate_tag_pose=False,
    camera_params=None,
    tag_size=None,
)

print(f"Found {len(detections)} apriltag(s)")
for det in detections:
    print(
        f"id={det.tag_id:3d}, "
        f"hamming={det.hamming}, "
        f"decision_margin={det.decision_margin:.2f}, "
        f"center=({det.center[0]:.1f}, {det.center[1]:.1f})"
    )

fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")

for det in detections:
    corners = np.array(det.corners)
    # Close the loop by repeating the first corner.
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], "r-")
    ax.scatter(det.center[0], det.center[1], c="cyan", s=20)
    ax.text(det.center[0], det.center[1], str(det.tag_id), color="yellow")

ax.set_title("Pupil AprilTag detections")
ax.axis("off")
plt.show()
