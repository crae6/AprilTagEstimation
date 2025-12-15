#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import time
from imageio.v2 import imread
try:
    from pupil_apriltags import Detector
except ImportError as exc:
    Detector = None 
    APRILTAG_IMPORT_ERROR = exc
from scipy import ndimage

ALGORITHMS = ["baseline", "wiener", "inverse"]
ALGORITHM_DISPLAY = {
    "baseline": "Original",
    "wiener": "Wiener",
    "inverse": "Inverse",
}

# hardcoded experiment knobs.
FIXED_PSF_LENGTH_PX = 15
FIXED_PSF_ANGLE_DEG = 0.0
WIENER_K = 1e-3  
INVERSE_EPS = 1e-8  # only prevents division-by-zero, otherwise "true" inverse.
SKIP_IF_BASELINE_DETECTS = True
TAG_SIZE_M = 0.19
FILTER_PATTERN = ""

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config() -> SimpleNamespace:
    root = repo_root()
    output_dir = root / "motion_blur" / "results_fixed15_all"
    output_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        csv=root / "dataset" / "data.csv",
        camera_params=root / "pi_cam_params.txt",
        output_dir=output_dir,
        filter_pattern=FILTER_PATTERN,
        tag_size_m=TAG_SIZE_M,
        quad_decimate=0.5,
        quad_sigma=1.0,
        decode_sharpening=0.5,
    )

def _parse_block(block: str, rows: int, cols: int) -> np.ndarray:
    numbers = np.fromstring(block.replace("[", " ").replace("]", " "), sep=" ")
    if numbers.size != rows * cols:
        raise ValueError(f"Expected {rows*cols} values, got {numbers.size}")
    return numbers.reshape(rows, cols)

def load_camera_calibration(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    text = path.read_text()
    cam_match = re.search(r"Camera Matrix:\s*(\[\[.*?\]\])", text, re.S)
    dist_match = re.search(r"Distortion Coefficients:\s*(\[\[.*?\]\])", text, re.S)
    if not cam_match or not dist_match:
        raise ValueError(f"Could not parse camera parameters from {path}")
    camera_matrix = _parse_block(cam_match.group(1), rows=3, cols=3).astype(np.float32)
    distortion = (
        np.fromstring(dist_match.group(1).replace("[", " ").replace("]", " "), sep=" ")
        .astype(np.float32)
        .ravel()
    )
    return camera_matrix, distortion

def read_dataset(csv_path: Path, pattern: str | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for raw_path, depth_str in reader:
            raw_path = raw_path.strip()
            depth_str = depth_str.strip()
            if not raw_path:
                continue
            if pattern and pattern not in raw_path:
                continue
            try:
                depth_mm = float(depth_str)
            except ValueError:
                continue
            rows.append({"raw_path": raw_path, "depth_mm": depth_mm})
    return rows

def resolve_image_path(raw_path: str) -> Path:
    root = repo_root()
    candidates = [
        root / raw_path,
        root / "dataset" / raw_path,
    ]
    parts = Path(raw_path)
    if parts.parts and parts.parts[0] == "images":
        tail = Path(*parts.parts[1:])
        candidates.append(root / "dataset" / tail)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {raw_path}")

def motion_psf(shape: Tuple[int, int], length: int, angle: float = 0.0) -> np.ndarray:
    h, w = shape
    if length <= 1:
        psf = np.zeros((h, w), dtype=np.float32)
        psf[h // 2, w // 2] = 1.0
        return psf

    size = int(max(1, length))
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    kernel = ndimage.rotate(kernel, angle=angle, reshape=False, order=1, mode="constant")
    kernel = np.clip(kernel, 0.0, None)
    kernel /= kernel.sum() + 1e-12

    psf = np.zeros((h, w), dtype=np.float32)
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    y1 = min(y0 + size, h)
    x1 = min(x0 + size, w)
    psf[y0:y1, x0:x1] = kernel[: y1 - y0, : x1 - x0]
    psf /= psf.sum() + 1e-12
    return psf

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    H = np.zeros(shape, dtype=np.complex64)
    ph, pw = psf.shape
    H[:ph, :pw] = psf
    H = np.fft.ifftshift(H)
    return np.fft.fft2(H)

def wiener_filter(blurred: np.ndarray, psf: np.ndarray) -> np.ndarray:
    G = np.fft.fft2(blurred)
    H = psf2otf(psf, blurred.shape)
    H_conj = np.conj(H)
    denom = (np.abs(H) ** 2) + WIENER_K
    recon = np.real(np.fft.ifft2((H_conj / denom) * G))
    return np.clip(recon, 0.0, 1.0)

def inverse_filter(blurred: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Pseudo-inverse filter with light regularization (matches results_fixed15)."""
    G = np.fft.fft2(blurred)
    H = psf2otf(psf, blurred.shape)
    H_conj = np.conj(H)
    denom = (np.abs(H) ** 2) + INVERSE_EPS
    recon = np.real(np.fft.ifft2((H_conj / denom) * G))
    return np.clip(recon, 0.0, 1.0)

def focus_measure(image: np.ndarray) -> float:
    lap = ndimage.laplace(image)
    return float(np.var(lap))

def apply_fixed_psf(image: np.ndarray) -> Dict[str, float | np.ndarray]:
    psf = motion_psf(image.shape, length=FIXED_PSF_LENGTH_PX, angle=FIXED_PSF_ANGLE_DEG)
    recon = wiener_filter(image, psf)
    return {
        "image": recon,
        "length": FIXED_PSF_LENGTH_PX,
        "angle": FIXED_PSF_ANGLE_DEG,
        "score": focus_measure(recon),
        "psf": psf,
    }

def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)

def enhance_for_detection(image: np.ndarray) -> np.ndarray:
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-6)
    filtered = ndimage.median_filter(normalized, size=3)
    return np.clip(filtered, 0.0, 1.0)

def run_detection(
    image: np.ndarray,
    detector: Detector,
    camera_params: Sequence[float],
    tag_size_m: float | None,
) -> List:
    estimate_pose = tag_size_m is not None
    prep = enhance_for_detection(image)
    return detector.detect(
        to_uint8(prep),
        estimate_tag_pose=estimate_pose,
        camera_params=camera_params if estimate_pose else None,
        tag_size=tag_size_m if estimate_pose else None,
    )

def detections_to_serializable(detections: Iterable, depth_m: float) -> List[Dict[str, float]]:
    serializable: List[Dict[str, float]] = []
    for det in detections:
        pose_t = None
        depth_error = None
        if det.pose_t is not None:
            pose_arr = np.asarray(det.pose_t).reshape(-1)
            pose_t = pose_arr.tolist()
            if pose_arr.size >= 3:
                depth_error = float(pose_arr[2] - depth_m)
        serializable.append(
            {
                "id": int(det.tag_id),
                "decision_margin": float(det.decision_margin),
                "hamming": int(det.hamming),
                "center": [float(det.center[0]), float(det.center[1])],
                "pose_t": pose_t,
                "depth_error_m": depth_error,
            }
        )
    return serializable

def detection_statistics(
    detections: List[Dict[str, float]], depth_m: float
) -> Tuple[int, float | None, float | None, float | None]:
    count = len(detections)
    best_depth_error = None
    best_margin = None
    best_est_depth_m = None
    if detections:
        decision_margins = [d["decision_margin"] for d in detections if "decision_margin" in d]
        if decision_margins:
            best_margin = float(max(decision_margins))
        for det in detections:
            err = det.get("depth_error_m")
            if err is None:
                continue
            abs_err = abs(err)
            if best_depth_error is None or abs_err < best_depth_error:
                best_depth_error = float(abs_err)
                if det.get("pose_t") and len(det["pose_t"]) >= 3:
                    best_est_depth_m = float(det["pose_t"][2])
    return count, best_depth_error, best_margin, best_est_depth_m

def select_best_algorithm(
    algo_outputs: Dict[str, Dict[str, float | np.ndarray]]
) -> Tuple[str, Dict[str, float | np.ndarray]]:
    winning_algo = None
    best_count = -1
    best_margin = -math.inf
    for algo, data in algo_outputs.items():
        count = int(data["count"])
        margin_val = data["best_decision_margin"]
        margin = margin_val if margin_val is not None else -math.inf
        if count > best_count or (count == best_count and margin > best_margin):
            winning_algo = algo
            best_count = count
            best_margin = margin
    assert winning_algo is not None
    return winning_algo, algo_outputs[winning_algo]

def annotate_axes(ax, image: np.ndarray, detections: Iterable, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    for det in detections:
        corners = np.array(det.corners)
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], "r-")
        ax.scatter(det.center[0], det.center[1], c="cyan", s=12)
        ax.text(det.center[0], det.center[1], str(det.tag_id), color="yellow", fontsize=8)
    ax.set_title(title)
    ax.axis("off")

def base_name(path: Path) -> str:
    return path.stem

def record_results(
    algo: str,
    image: np.ndarray,
    detections: List,
    depth_m: float,
    algo_outputs: Dict[str, Dict[str, float | np.ndarray]],
    aggregate: Dict[str, Dict[str, object]],
    elapsed_time: float,
) -> None:
    serializable = detections_to_serializable(detections, depth_m)
    count, best_error, best_margin, best_est_depth_m = detection_statistics(serializable, depth_m)
    algo_outputs[algo] = {
        "image": image,
        "detections": detections,
        "serializable": serializable,
        "count": count,
        "best_depth_error_m": best_error,
        "best_decision_margin": best_margin,
    }
    aggregate_entry = aggregate[algo]
    aggregate_entry["detections"] += count
    aggregate_entry["time_s"] += elapsed_time
    if best_error is not None:
        aggregate_entry["errors_mm"].append(best_error * 1000.0)
        if depth_m > 0:
            aggregate_entry["relative_errors"].append(best_error / depth_m)

def process_entries(cfg: SimpleNamespace) -> None:
    if Detector is None:
        raise ImportError(
            "pupil_apriltags is not installed. Install it with `pip install pupil-apriltags`."
        ) from APRILTAG_IMPORT_ERROR
    camera_matrix, _ = load_camera_calibration(cfg.camera_params)
    camera_params = [
        float(camera_matrix[0, 0]),
        float(camera_matrix[1, 1]),
        float(camera_matrix[0, 2]),
        float(camera_matrix[1, 2]),
    ]

    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=cfg.quad_decimate,
        quad_sigma=cfg.quad_sigma,
        refine_edges=1,
        decode_sharpening=cfg.decode_sharpening,
        debug=0,
    )

    entries = read_dataset(cfg.csv, pattern=cfg.filter_pattern)
    if not entries:
        print(f"No matching entries found in {cfg.csv}")
        return

    figures_dir = cfg.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float | str | int | None]] = []
    aggregate = {
        algo: {"detections": 0, "errors_mm": [], "relative_errors": [], "time_s": 0.0}
        for algo in ALGORITHMS
    }

    for entry in entries:
        resolved_path = resolve_image_path(entry["raw_path"])
        depth_mm = entry["depth_mm"]
        depth_m = depth_mm / 1000.0
        print(f"Processing {entry['raw_path']} (depth {depth_mm:.1f} mm)")

        image = imread(resolved_path, pilmode="L").astype(np.float32) / 255.0

        algo_outputs: Dict[str, Dict[str, float | np.ndarray]] = {}
        # Baseline detection first.
        start = time.perf_counter()
        baseline_dets = run_detection(image, detector, camera_params, cfg.tag_size_m)
        elapsed = time.perf_counter() - start
        record_results("baseline", image, baseline_dets, depth_m, algo_outputs, aggregate, elapsed)

        # Default PSF result indicates no filtering (used when skipping).
        psf_result: Dict[str, float | np.ndarray] = {
            "image": image,
            "length": 0,
            "angle": 0.0,
            "score": focus_measure(image),
            "psf": motion_psf(image.shape, length=0, angle=0.0),
        }

        baseline_count = algo_outputs["baseline"]["count"]
        should_skip = SKIP_IF_BASELINE_DETECTS and baseline_count and baseline_count > 0

        if not should_skip:
            start = time.perf_counter()
            psf_result = apply_fixed_psf(image)
            wiener_image = psf_result["image"]
            wiener_dets = run_detection(wiener_image, detector, camera_params, cfg.tag_size_m)
            elapsed = time.perf_counter() - start
            record_results("wiener", wiener_image, wiener_dets, depth_m, algo_outputs, aggregate, elapsed)

            start = time.perf_counter()
            inverse_image = inverse_filter(image, psf_result["psf"])
            inverse_dets = run_detection(inverse_image, detector, camera_params, cfg.tag_size_m)
            elapsed = time.perf_counter() - start
            record_results("inverse", inverse_image, inverse_dets, depth_m, algo_outputs, aggregate, elapsed)
        else:
            # Copy the baseline detections to other algorithms since we skipped filtering.
            record_results("wiener", image, baseline_dets, depth_m, algo_outputs, aggregate, 0.0)
            record_results("inverse", image, baseline_dets, depth_m, algo_outputs, aggregate, 0.0)

        fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(4 * len(ALGORITHMS), 4))  # type: ignore
        axes_list = axes.ravel().tolist() if isinstance(axes, np.ndarray) else [axes]
        for ax, algo in zip(axes_list, ALGORITHMS):
            result = algo_outputs[algo]
            title = f"{ALGORITHM_DISPLAY.get(algo, algo)} ({result['count']} tag(s))"
            annotate_axes(ax, result["image"], result["detections"], title)
        fig.suptitle(
            f"{base_name(resolved_path)} | len={psf_result['length']} px angle={psf_result['angle']:.1f} deg"
        )
        fig.tight_layout()
        figure_path = figures_dir / f"{base_name(resolved_path)}_comparison.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)  # type: ignore

        json_data = {
            "raw_path": entry["raw_path"],
            "resolved_path": str(resolved_path),
            "depth_mm": depth_mm,
            "estimated_kernel": {
                "length_px": psf_result["length"],
                "angle_deg": psf_result["angle"],
                "focus_score": psf_result["score"],
            },
            "algorithms": {
                algo: {
                    "detections": algo_outputs[algo]["serializable"],
                    "count": algo_outputs[algo]["count"],
                    "best_depth_error_m": algo_outputs[algo]["best_depth_error_m"],
                    "best_decision_margin": algo_outputs[algo]["best_decision_margin"],
                }
                for algo in ALGORITHMS
            },
        }
        winner_algo, winner_data = select_best_algorithm(algo_outputs)
        json_data["best_algorithm"] = winner_algo
        json_data["best_algorithm_count"] = int(winner_data["count"])
        json_data["best_algorithm_detections"] = winner_data["serializable"]
        json_path = metrics_dir / f"{base_name(resolved_path)}.json"
        json_path.write_text(json.dumps(json_data, indent=2))

        row: Dict[str, float | str | int | None] = {
            "raw_path": entry["raw_path"],
            "resolved_path": str(resolved_path.relative_to(repo_root())),
            "depth_mm": depth_mm,
            "estimated_length_px": psf_result["length"],
            "estimated_angle_deg": psf_result["angle"],
            "focus_score": psf_result["score"],
            "figure": str(figure_path.relative_to(repo_root())),
            "best_algorithm": winner_algo,
            "best_algorithm_detections": winner_data["count"],
        }
        for algo in ALGORITHMS:
            row[f"detections_{algo}"] = algo_outputs[algo]["count"]
            row[f"best_depth_error_m_{algo}"] = algo_outputs[algo]["best_depth_error_m"]
            row[f"best_decision_margin_{algo}"] = algo_outputs[algo]["best_decision_margin"]
        summary_rows.append(row)

    summary_path = cfg.output_dir / "summary.csv"
    fieldnames = [
        "raw_path",
        "resolved_path",
        "depth_mm",
        "estimated_length_px",
        "estimated_angle_deg",
        "focus_score",
        "figure",
        "best_algorithm",
        "best_algorithm_detections",
    ]
    for algo in ALGORITHMS:
        fieldnames.extend(
            [
                f"detections_{algo}",
                f"best_depth_error_m_{algo}",
                f"best_decision_margin_{algo}",
            ]
        )
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary to {summary_path}")

    for algo in ALGORITHMS:
        label = ALGORITHM_DISPLAY.get(algo, algo)
        total_det = aggregate[algo]["detections"]
        errors = aggregate[algo]["errors_mm"]
        if errors:
            mean_error_m = float(np.mean(errors))
            median_error_m = float(np.median(errors))
            print(
                f"[{label}] total detections: {total_det}, "
                f"mean abs depth error: {mean_error_m:.1f} mm, "
                f"median abs depth error: {median_error_m:.1f} mm"
            )
        else:
            print(f"[{label}] total detections: {total_det}, no pose estimates available.")

    print("\nOverall Depth/Timing Metrics")
    header = f"{'Algorithm':<12}{'MAE (mm)':>12}{'AbsRel':>12}{'Time/Image (s)':>16}"
    print(header)
    print("-" * len(header))
    num_images = len(entries)
    for algo in ALGORITHMS:
        label = ALGORITHM_DISPLAY.get(algo, algo)
        errors = aggregate[algo]["errors_mm"]
        rels = aggregate[algo]["relative_errors"]
        mae = float(np.mean(errors)) if errors else float("nan")
        absrel = float(np.mean(rels)) if rels else float("nan")
        time_per = aggregate[algo]["time_s"] / num_images if num_images else float("nan")
        print(f"{label:<12}{mae:>12.1f}{absrel:>12.3f}{time_per:>16.3f}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = load_config()
    process_entries(config)
