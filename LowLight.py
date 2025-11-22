import cv2
import numpy as np

IMAGE_PATH = "IMG_4836.png"
DARK_FACTOR = 0.3


def make_lowlight(img, factor=0.3):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def gamma_correction(l, gamma=0.7):
    l = l.astype(np.float32) / 255.0
    l = np.power(l, gamma) * 255
    return np.uint8(l)

def apply_clahe(l):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(l)

def retinex(l, sigma=30):
    blur = cv2.GaussianBlur(l, (0,0), sigma)
    r = np.log1p(l.astype(np.float32)) - np.log1p(blur.astype(np.float32))
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(r)

def enhance_lowlight_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mu = np.mean(L)
    if mu > 80:
        print("Image is already bright enough, skipping enhancement.")
        return img
    L_g = gamma_correction(L, gamma=0.7)
    L_c = apply_clahe(L_g)
    L_r = retinex(L_c, sigma=30)
    L_final = (0.7 * L_r + 0.3 * L_c).astype(np.uint8)
    lab_enhanced = cv2.merge([L_final, A, B])
    enhanced_color = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced_color

def main():
    original = cv2.imread(IMAGE_PATH)
    if original is None:
        print(f"Could not find {IMAGE_PATH}")
        return
    print("Image loaded")
    lowlight = make_lowlight(original, DARK_FACTOR)
    enhanced_color = enhance_lowlight_color(lowlight)
    h = 400
    original_r = cv2.resize(original, (int(original.shape[1]*h/original.shape[0]), h))
    lowlight_r = cv2.resize(lowlight, (int(lowlight.shape[1]*h/lowlight.shape[0]), h))
    enhanced_r = cv2.resize(enhanced_color, (int(enhanced_color.shape[1]*h/enhanced_color.shape[0]), h))
    combined = cv2.hconcat([original_r, lowlight_r, enhanced_r])

    cv2.imshow("ORIGINAL | LOW-LIGHT | CORRECTED (COLOR)", combined)
    cv2.imwrite("original.png", original_r)
    cv2.imwrite("lowlight.png", lowlight_r)
    cv2.imwrite("enhanced.png", enhanced_r)
    cv2.imwrite("comparison_triplet.png", combined)

    print("✅ Saved comparison_triplet.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
