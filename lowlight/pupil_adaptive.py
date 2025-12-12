import cv2
import numpy as np
import os
from pupil_apriltags import Detector as PupilDetector

DARK_FACTOR = 0.3
SHADOW_THRESH = 80
BRIGHTEN_MULT = 2.5

def make_lowlight(img, factor=0.3):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def adaptive_shadow_boost(img):
    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shadow_mask = gray < SHADOW_THRESH
    shadow_mask_3 = np.repeat(shadow_mask[:, :, None], 3, axis=2)

    boosted = img_f.copy()
    boosted[shadow_mask_3] *= BRIGHTEN_MULT
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)

    filtered = cv2.bilateralFilter(boosted, 9, 75, 75)
    p2, p98 = np.percentile(filtered, (2, 98))
    stretched = np.clip(
        (filtered - p2) * (255.0 / (p98 - p2)),
        0, 255
    ).astype(np.uint8)

    return stretched

class DetectorOptions:
    def __init__(self,
                 families='tag36h11',
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
        results = self.det.detect(gray)

        if not return_image:
            return results

        dimg = np.zeros_like(gray)
        for r in results:
            pts = r.corners.astype(int)
            cv2.polylines(dimg, [pts], True, 255, 2)

        return results, dimg

def detect_tags(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections, dimg = detector.detect(gray, return_image=True)

    overlay = image // 2 + dimg[:, :, None] // 2

    for r in detections:
        pts = r.corners.astype(int)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        cx, cy = r.center
        cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(
            overlay,
            f"ID:{r.tag_id} M:{r.decision_margin:.1f}",
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return detections, overlay

PI_CAM_DIR = "dataset/pi_cam"
OUT_DIR = "results_pi_cam"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    options = DetectorOptions()
    det = Detector(options)
    image_files = sorted(
        f for f in os.listdir(PI_CAM_DIR)
        if f.lower().endswith(".jpg")
    )

    print(f"\nFound {len(image_files)} images in {PI_CAM_DIR}\n")
    total_baseline = 0
    total_lowlight = 0
    total_boosted = 0

    for filename in image_files:
        path = os.path.join(PI_CAM_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"Could not load {filename}")
            continue

        print(f"\n----- Processing {filename} -----")
        base_det, base_overlay = detect_tags(img, det)
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename}_baseline.png"), base_overlay)
        print(f"Baseline detections: {len(base_det)}")
        total_baseline += len(base_det)       

        lowlight = make_lowlight(img, DARK_FACTOR)
        low_det, low_overlay = detect_tags(lowlight, det)
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename}_lowlight.png"), low_overlay)
        print(f"Low-light detections: {len(low_det)}")
        total_lowlight += len(low_det)      
        boosted = adaptive_shadow_boost(lowlight)
        boost_det, boost_overlay = detect_tags(boosted, det)
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename}_boosted.png"), boost_overlay)
        print(f"Boosted detections: {len(boost_det)}")
        total_boosted += len(boost_det)       


    print("\n-----------------------------------------")
    print("           FINAL RESULTS")
    print("-----------------------------------------")
    print(f"Total Baseline detections:     {total_baseline}")
    print(f"Total Method2 detections:      {total_boosted}")
    print("-----------------------------------------")
    print(f"All processed outputs saved in: {OUT_DIR}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
