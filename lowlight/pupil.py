import cv2
import numpy as np
from pupil_apriltags import Detector as PupilDetector

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

        self.options = options
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
        assert len(gray.shape) == 2, "Input must be grayscale"
        assert gray.dtype == np.uint8, "Input must be uint8 grayscale"

        results = self.det.detect(gray)

        if not return_image:
            return results
        dimg = np.zeros_like(gray)
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners.astype(int)
            cv2.polylines(dimg, [np.array([ptA, ptB, ptC, ptD])],
                          True, 255, 2)
        return results, dimg

def detect_tags(image,
                detector,
                camera_params=(600.0, 600.0, 320.0, 240.0),
                tag_size=0.16,
                vizualization=1,
                verbose=1,
                annotation=True):


    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detections, dimg = detector.detect(gray, return_image=True)

    if len(image.shape) == 3:
        overlay = image // 2 + dimg[:, :, None] // 2
    else:
        overlay = gray // 2 + dimg // 2

    if verbose:
        print(f"Detected {len(detections)} tags")

    if annotation and len(image.shape) == 3:
        for r in detections:
            (ptA, ptB, ptC, ptD) = r.corners.astype(int)
            cv2.polylines(overlay, [np.array([ptA, ptB, ptC, ptD])],
                          True, (0, 255, 0), 2)
            cX, cY = r.center
            cv2.circle(overlay, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            cv2.putText(overlay,
                        f"ID:{r.tag_id} M:{r.decision_margin:.1f}",
                        (int(ptA[0]), int(ptA[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

    return detections, overlay


IMAGE_PATH = "pi_cam_11_lowlight.jpg"

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return

    options = DetectorOptions(families='tag36h11')
    det = Detector(options)

    detections, overlay = detect_tags(
        img,
        det,
        camera_params=(600, 600, img.shape[1] / 2, img.shape[0] / 2),
        tag_size=0.16,
        vizualization=1,
        verbose=1,
        annotation=True
    )

    print("Done. Saving baseline overlay to baseline_tinker_overlay.png")
    cv2.imshow("Baseline Tinker-Style AprilTag Detection", overlay)
    cv2.imwrite("baseline_tinker_overlay.png", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
