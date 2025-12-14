import cv2
import numpy as np
import os
import time
from pupil_apriltags import Detector as PupilDetector

PI_CAM_DIR = "dataset/pi_cam"
OUT_DIR = "results_pi_cam_full"

TAG_FAMILY = "tag36h11"
TAG_SIZE = 0.20
CAMERA_PARAMS = (2.48502856e+03, 2.48095083e+03, 1.67655746e+03, 1.33820137e+03)

DARK_FACTOR = 0.3
SHADOW_THRESH = 80
BRIGHTEN_MULT = 2.5


def make_lowlight(img, factor=DARK_FACTOR):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adaptive_shadow_boost(img):
    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = gray < SHADOW_THRESH
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

    boosted = img_f.copy()
    boosted[mask_3] *= BRIGHTEN_MULT
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)

    filt = cv2.bilateralFilter(boosted, 9, 75, 75)
    p2, p98 = np.percentile(filt, (2, 98))
    stretched = np.clip((filt - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

    return stretched

class DetectorOptions:
    def __init__(self,
                 families=TAG_FAMILY,
                 nthreads=4,
                 quad_decimate=1.0,
                 quad_blur=0.0,
                 refine_edges=True,
                 debug=False):
        self.families = families
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_blur = quad_blur
        self.refine_edges = refine_edges
        self.debug = debug


class Detector:
    def __init__(self, options=None):
        if options is None:
            options = DetectorOptions()

        self.det = PupilDetector(
            families=options.families,
            nthreads=options.nthreads,
            quad_decimate=options.quad_decimate,
            quad_sigma=options.quad_blur,
            refine_edges=int(options.refine_edges),
            decode_sharpening=0.25,
            debug=int(options.debug)
        )

    def detect(self, gray, return_image=False):
        results = self.det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=TAG_SIZE
        )

        if not return_image:
            return results

        dimg = np.zeros_like(gray)
        for r in results:
            pts = r.corners.astype(int)
            cv2.polylines(dimg, [pts], True, 255, 2)
        return results, dimg

def detect_tags(image, detector):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_t = time.time()
    results, dimg = detector.detect(gray, return_image=True)
    runtime = time.time() - start_t

    overlay = image // 2 + dimg[:, :, None] // 2

    pose_errors = []
    margins = []

    for r in results:

        pts = r.corners.astype(int)
        cx, cy = map(int, r.center)

        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

        cv2.putText(
            overlay,
            f"ID:{r.tag_id} M:{r.decision_margin:.1f}",
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        margins.append(r.decision_margin)
        if r.pose_t is not None:
            pose_errors.append(float(np.linalg.norm(r.pose_t)))

    return results, overlay, runtime, pose_errors, margins


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    detector = Detector(DetectorOptions())

    files = sorted([f for f in os.listdir(PI_CAM_DIR) if f.endswith(".jpg")])
    print(f"Found {len(files)} images.")

    # totals
    total_base = total_boost = 0
    base_time = boost_time = 0.0
    base_pose = boost_pose = 0.0
    base_margins = boost_margins = 0.0
    base_count = boost_count = 0

    for fname in files:
        print(f"\n--- Processing {fname} ---")

        img = cv2.imread(os.path.join(PI_CAM_DIR, fname))
        if img is None:
            continue
        detB, outB, timeB, poseB, margB = detect_tags(img, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_baseline.png"), outB)

        total_base += len(detB)
        base_time += timeB
        base_pose += sum(poseB)
        base_margins += sum(margB)
        base_count += len(poseB)
        low = make_lowlight(img)
        boosted = adaptive_shadow_boost(low)

        detE, outE, timeE, poseE, margE = detect_tags(boosted, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_boosted.png"), outE)

        total_boost += len(detE)
        boost_time += timeE
        boost_pose += sum(poseE)
        boost_margins += sum(margE)
        boost_count += len(poseE)

    print("\n==================== FINAL RESULTS ====================")

    print(f"Baseline total detections: {total_base}")
    print(f"Boosted total detections:  {total_boost}")

    print("\n--- SPEED ---")
    print(f"Baseline avg runtime: {base_time / len(files):.4f}s")
    print(f"Boosted  avg runtime: {boost_time / len(files):.4f}s")

    print("\n--- POSE ACCURACY (lower is better) ---")
    print(f"Baseline avg pose error: {base_pose / (base_count or 1):.4f}")
    print(f"Boosted  avg pose error: {boost_pose / (boost_count or 1):.4f}")

    print("\n--- DECISION MARGIN (higher is better) ---")
    print(f"Baseline avg margin: {base_margins / (base_count or 1):.4f}")
    print(f"Boosted  avg margin:  {boost_margins / (boost_count or 1):.4f}")

    print("========================================================")


if __name__ == "__main__":
    main()
