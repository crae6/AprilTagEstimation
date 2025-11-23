import cv2
import numpy as np

IMAGE_PATH = "IMG_4836.png"
DARK_FACTOR = 0.3     
SHADOW_THRESH = 80   
BRIGHTEN_MULT = 2.5   

def make_lowlight(img, factor=0.3):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adaptive_shadow_boost(img):
    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shadow_mask = gray < SHADOW_THRESH
    shadow_mask_3 = np.repeat(shadow_mask[:, :, np.newaxis], 3, axis=2)
    boosted = img_f.copy()
    boosted[shadow_mask_3] = boosted[shadow_mask_3] * BRIGHTEN_MULT
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(boosted, d=9, sigmaColor=75, sigmaSpace=75)
    p2, p98 = np.percentile(filtered, (2, 98))
    stretched = np.clip((filtered - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

    return stretched

def main():
    original = cv2.imread(IMAGE_PATH)
    if original is None:
        print(f"Could not load image {IMAGE_PATH}")
        return
    print("Image loaded")

    lowlight = make_lowlight(original, DARK_FACTOR)
    boosted = adaptive_shadow_boost(lowlight)

    h = 400
    original_r = cv2.resize(original, (int(original.shape[1]*h/original.shape[0]), h))
    lowlight_r = cv2.resize(lowlight, (int(lowlight.shape[1]*h/lowlight.shape[0]), h))
    boosted_r = cv2.resize(boosted, (int(boosted.shape[1]*h/boosted.shape[0]), h))
    combined = cv2.hconcat([original_r, lowlight_r, boosted_r])

    cv2.imshow("ORIGINAL | LOW-LIGHT | ADAPTIVE SHADOW BOOST", combined)
    cv2.imwrite("comparison_method2.png", combined)
    print("Saved: comparison_method2.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
