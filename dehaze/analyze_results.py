import numpy as np
import csv
from matplotlib import pyplot as plt

def get_gt(csv_path):
    gt_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_name, depth = row
            gt_dict[image_name] = float(depth)
    return gt_dict

def compute_metrics(pred, gt):
    """
    pred, gt: numpy arrays of same length
    """
    diff = pred - gt

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    mse = np.mean(diff ** 2)

    # Relative errors (important for depth)
    abs_rel = np.mean(np.abs(diff) / gt)
    sq_rel = np.mean((diff ** 2) / gt)

    # Scale-invariant log RMSE (common in depth estimation)
    log_diff = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)

    return {
        "MAE": mae,
        # "RMSE": rmse,
        # "MSE": mse,
        "AbsRel": abs_rel,
        # "SqRel": sq_rel,
        # "SiLog": silog,
    }


def compare_file(pred_dict, gt_dict):
    common_keys = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    pred = np.array([pred_dict[k] for k in common_keys]) * 1000  # convert to mm
    gt = np.array([gt_dict[k] for k in common_keys])
    metrics = compute_metrics(pred, gt)

    return metrics, len(common_keys)


def gt_pred_scatter(pred_dict, gt_dict, color='blue', label='', plot_line=False):
    common_keys = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    pred = np.array([pred_dict[k] for k in common_keys]) * 1000  # convert to mm
    gt = np.array([gt_dict[k] for k in common_keys])

    plt.scatter(gt, pred, alpha=0.5, color=color, label=label)

gt_dict = get_gt("../dataset/data.csv")

baseline = np.load('depths_baseline.npy', allow_pickle=True).item()
dcp = np.load('depths_dcp.npy', allow_pickle=True).item()
ssr = np.load('depths_single_scale.npy', allow_pickle=True).item()
msr = np.load('depths_multi_scale.npy', allow_pickle=True).item()
model = np.load('depths_model.npy', allow_pickle=True).item()
baseline_metrics, baseline_count = compare_file(baseline, gt_dict)
dcp_metrics, dcp_count = compare_file(dcp, gt_dict)
ssr_metrics, ssr_count = compare_file(ssr, gt_dict)
msr_metrics, msr_count = compare_file(msr, gt_dict)
model_metrics, model_count = compare_file(model, gt_dict)


test_image = 'pi_cam/pi_cam_39_haze.jpg'

print(test_image in baseline.keys())
print(test_image in ssr.keys())
print(test_image in dcp.keys())
print(test_image in msr.keys())
print(test_image in model.keys())

keys = baseline.keys()
for key in keys:
    if key not in dcp:
        print(f"Missing in Baseline: {key}")
times = np.array([20.08, 136.68, 517.77, 1833.75, 2844.88])

print(times / 176)

print("Baseline Metrics:", baseline_metrics)
print("DCP Metrics:", dcp_metrics)
print("SSR Metrics:", ssr_metrics)
print("MSR Metrics:", msr_metrics)
print("Model Metrics:", model_metrics)

plt.figure(figsize=(8, 8))


plt.plot([0, 10000], [0, 10000], 'r--', label='y=x')
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.xlabel('Ground Truth Depth (mm)')
plt.ylabel('Predicted Depth (mm)')
plt.title('Ground Truth vs Predicted Depth')
gt_pred_scatter(baseline, gt_dict, color='blue')
gt_pred_scatter(dcp, gt_dict, color='green')
gt_pred_scatter(ssr, gt_dict, color='orange')
gt_pred_scatter(msr, gt_dict, color='purple')
gt_pred_scatter(model, gt_dict, color='red')
plt.legend(['y=x','Baseline', 'DCP', 'SSR', 'MSR', 'FFA Net'])
plt.grid(True)
plt.show()