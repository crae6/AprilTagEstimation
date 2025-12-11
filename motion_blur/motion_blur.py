import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy.ndimage import rotate
from pupil_apriltags import Detector

def motion_psf(shape, length, angle=0):
    '''
    Creates a point-spread-function of the image to describe how the data is 
    blurred by the camera
    '''
    h, w = shape
    psf_size = max(h, w)

    psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    center = psf_size // 2 

    # horizontal line centered
    half = length // 2
    psf[center, center - half:center + half + (length % 2)] = 1.0

    # rotate to angle input
    psf = rotate(psf, angle, reshape=False, order=1, mode='constant')

    # normalize 
    psf /= psf.sum() + 1e-12

    # crop to image 
    y0 = (psf_size - h) // 2
    x0 = (psf_size - w) // 2
    psf_cropped = psf[y0:y0 + h, x0:x0 + w]

    # renormalize 
    psf_cropped /= psf_cropped.sum() + 1e-12

    return psf_cropped

def psf2otf(psf, shape=None):
    '''
    takes the fourier transform of the Point Spread Function to convert it into 
    Optical Transfer Function which will then act as our H in deblurring. 
    '''
    if shape is None:
        shape = psf.shape 

    H = np.zeros(shape, dtype=np.float32)
    ph, pw = psf.shape
    H[:ph, :pw] = psf

    # center the psf
    H = np.fft.ifftshift(H)
    H = np.fft.fft2(H)

    return H

def inverse_filter(blurred, psf, eps=1e-3):
    '''
    filter that takes the inverse of the predicted blur
    '''
    # fourier transform of blurred image
    G = np.fft.fft2(blurred)

    # otf of psf
    H = psf2otf(psf, shape=blurred.shape)

    H_conj= np.conj(H)
    denom = (np.abs(H) ** 2) + eps

    # Stabilization which basically turns it into a wiener filter
    # F_hat = (H_conj / denom) * G
    # f_hat = np.fft.ifft2(F_hat)
    # f_hat = np.real(f_hat)

    # # clip result between 0 and 1
    # f_hat = np.clip(f_hat, 0.0, 1.0)
    # return f_hat

    F_hat = G / (H + 1e-12) # true inverse
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 1)

def wiener_filter(blurred, psf, K=0.01):
    '''
    filter that takes the inverse and adjusts it to account for noise
    '''
    G = np.fft.fft2(blurred)
    H = psf2otf(psf, shape=blurred.shape)

    H_conj = np.conj(H)
    denom = (np.abs(H) ** 2) + K

    F_hat = (H_conj / denom) * G
    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat)

    f_hat = np.clip(f_hat, 0.0, 1.0)
    return f_hat

def detect_and_plot(ax, img_float, title, detector):
    """Run AprilTag detection and visualize the result."""
    img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    detections = detector.detect(
        img_uint8,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )

    ax.imshow(img_float, cmap="gray")
    for det in detections:
        corners = np.array(det.corners)
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], "r-")
        ax.scatter(det.center[0], det.center[1], c="cyan", s=10)
        ax.text(det.center[0], det.center[1], str(det.tag_id), color="yellow")

    ax.set_title(f"{title}\n{len(detections)} tag(s)")
    ax.axis("off")
    return detections

img = imread("motion_blur/apriltag.jpg", pilmode="L").astype(np.float32) / 255.0

psf = motion_psf(img.shape, length=25, angle=20)

H = psf2otf(psf, shape=img.shape)
F = np.fft.fft2(img)
G = H * F
blurred = np.real(np.fft.ifft2(G))
blurred = np.clip(blurred, 0.0, 1.0)
blurred_noisy = np.clip(blurred + 0.0005 * np.random.randn(*blurred.shape), 0.0, 1.0)

inv_rec = inverse_filter(blurred_noisy, psf, eps=10e-3)
wiener_rec = wiener_filter(blurred_noisy, psf, K=0.01)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

fig1 = plt.figure(figsize=(10, 4))
plt.subplot(1, 4, 1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1, 4, 2); plt.imshow(blurred_noisy, cmap="gray"); plt.title("Blurred"); plt.axis("off")
plt.subplot(1, 4, 3); plt.imshow(inv_rec, cmap="gray"); plt.title("Inverse"); plt.axis("off")
plt.subplot(1, 4, 4); plt.imshow(wiener_rec, cmap="gray"); plt.title("Wiener"); plt.axis("off")
fig1.tight_layout()

fig2, axes = plt.subplots(1, 2, figsize=(8, 4))
blur_dets = detect_and_plot(axes[0], blurred_noisy, "Blurred input", detector)
wiener_dets = detect_and_plot(axes[1], wiener_rec, "Wiener filtered", detector)
fig2.suptitle("AprilTag detections on blurred vs. filtered images")
fig2.tight_layout()

def summarize(label, detections):
    print(f"{label}: {len(detections)} detection(s)")
    for det in detections:
        print(
            f"  id={det.tag_id:3d}, "
            f"hamming={det.hamming}, "
            f"decision_margin={det.decision_margin:.2f}"
        )

summarize("Blurred image", blur_dets)
summarize("Wiener filtered image", wiener_dets)

plt.show()
