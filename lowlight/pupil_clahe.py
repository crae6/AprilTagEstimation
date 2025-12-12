import cv2
import numpy as np
import os
from pupil_apriltags import Detector as PupilDetector

PI_CAM_DIR = "dataset/pi_cam"
OUT_DIR = "results_pupil_clahe"
os.makedirs(OUT_DIR, exist_ok=True)


def enhance_lowlight_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_L = np.mean(L)

    if mean_L > 85:
        return img

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(4, 4)
    )
    L_clahe = clahe.apply(L)

    L_final = (0.6 * L + 0.4 * L_clahe).astype(np.uint8)
    enhanced_lab = cv2.merge([L_final, A, B])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def make_detector():
    return PupilDetector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )


def detect_and_annotate(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    annotated = img.copy()

    for r in results:
        pts = r.corners.astype(int)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        cx, cy = map(int, r.center)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        cv2.putText(
            annotated,
            f"ID:{r.tag_id}  M:{r.decision_margin:.1f}",
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return annotated, results

def main():
    detector = make_detector()

    files = sorted(f for f in os.listdir(PI_CAM_DIR) if f.endswith(".jpg"))
    print(f"Found {len(files)} images.")

    total_baseline = 0
    total_enhanced = 0

    for fname in files:
        print("\n------------------------------")
        print("Processing:", fname)
        print("--------------------------------")

        img_path = os.path.join(PI_CAM_DIR, fname)
        img = cv2.imread(img_path)

        if img is None:
            print("Failed to load:", fname)
            continue

        anno_base, det_base = detect_and_annotate(img, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_baseline.png"), anno_base)
        print("Baseline detections:", len(det_base))
        total_baseline += len(det_base)
        enhanced = enhance_lowlight_color(img)
        anno_enh, det_enh = detect_and_annotate(enhanced, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_enhanced.png"), anno_enh)
        print("Enhanced detections:", len(det_enh))
        total_enhanced += len(det_enh)
        h = 350
        resize = lambda x: cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))
        combined = cv2.hconcat([resize(anno_base), resize(anno_enh)])
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_comparison.png"), combined)

    print("\n------------------------------------")
    print("           FINAL RESULTS")
    print("------------------------------------")
    print(f"Total Baseline detections : {total_baseline}")
    print(f"Total Method1 detections : {total_enhanced}")
    print("------------------------------------")
    print("Outputs saved to:", OUT_DIR)
    print("------------------------------------\n")


if __name__ == "__main__":
    main()
