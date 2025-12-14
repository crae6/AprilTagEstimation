import cv2
import numpy as np
import os
import time
from pupil_apriltags import Detector as PupilDetector


PI_CAM_DIR = "dataset/pi_cam"
OUT_DIR = "results_pupil_clahe"
os.makedirs(OUT_DIR, exist_ok=True)

TAG_FAMILY = "tag36h11"
TAG_SIZE = 0.20   
CAMERA_PARAMS = (2.48502856e+03, 2.48095083e+03, 1.67655746e+03, 1.33820137e+03)  


def enhance_lowlight_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    if np.mean(L) > 85:
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
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )

def detect_with_metrics(img, detector):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.time()
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE
    )
    runtime = time.time() - start

    annotated = img.copy()
    pose_errors = []
    margins = []

    for r in results:
        pts = r.corners.astype(int)


        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        cx, cy = map(int, r.center)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"ID:{r.tag_id} M:{r.decision_margin:.1f}",
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 2
        )
        margins.append(r.decision_margin)
        if r.pose_t is not None:
            pose_errors.append(float(np.linalg.norm(r.pose_t)))

    return annotated, results, runtime, pose_errors, margins


def main():
    detector = make_detector()

    files = sorted(f for f in os.listdir(PI_CAM_DIR) if f.endswith(".jpg"))
    print(f"Found {len(files)} images.")

    total_base = 0
    total_enh = 0

    base_time = enh_time = 0.0
    base_pose = enh_pose = 0.0
    base_marg = enh_marg = 0.0

    base_count = enh_count = 0

    for fname in files:
        print("\n------------------------------")
        print("Processing:", fname)
        print("------------------------------")

        img_path = os.path.join(PI_CAM_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load:", fname)
            continue

        anno_base, det_base, tB, poseB, margB = detect_with_metrics(img, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_baseline.png"), anno_base)

        total_base += len(det_base)
        base_time += tB
        base_pose += sum(poseB)
        base_marg += sum(margB)
        base_count += len(poseB)

        print(f"Baseline: {len(det_base)} detections, {tB:.4f} sec")
        enh = enhance_lowlight_color(img)
        anno_enh, det_enh, tE, poseE, margE = detect_with_metrics(enh, detector)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_enhanced.png"), anno_enh)

        total_enh += len(det_enh)
        enh_time += tE
        enh_pose += sum(poseE)
        enh_marg += sum(margE)
        enh_count += len(poseE)

        print(f"Enhanced: {len(det_enh)} detections, {tE:.4f} sec")

        h = 350
        resize = lambda x: cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))
        combined = cv2.hconcat([resize(anno_base), resize(anno_enh)])
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_comparison.png"), combined)

    print("\n====================== FINAL RESULTS ======================")

    print(f"Total Baseline detections : {total_base}")
    print(f"Total Enhanced detections : {total_enh}")

    print("\n--- SPEED ---")
    print(f"Baseline avg runtime: {base_time / len(files):.4f} sec/img")
    print(f"Enhanced avg runtime: {enh_time / len(files):.4f} sec/img")

    print("\n--- POSE ACCURACY (lower = better) ---")
    print(f"Baseline avg pose error: {base_pose / (base_count or 1):.4f}")
    print(f"Enhanced avg pose error: {enh_pose / (enh_count or 1):.4f}")

    print("\n--- DECISION MARGIN (higher = better) ---")
    print(f"Baseline avg margin: {base_marg / (base_count or 1):.4f}")
    print(f"Enhanced avg margin: {enh_marg / (enh_count or 1):.4f}")

    print("============================================================")


if __name__ == "__main__":
    main()
