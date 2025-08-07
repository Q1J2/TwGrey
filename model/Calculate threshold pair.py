import numpy as np
import pandas as pd
import os

features = np.load(r"D:\UCM\UCM100\RFF\processed_features.npy")
gray_corr = np.load(r"D:\UCM\UCM100\RFF\feature_importance.npy")
gamma = gray_corr.copy()

feature_names_path = r"D:\UCM\UCM100\RFF\feature_names.csv"
if os.path.exists(feature_names_path):
    feature_names_df = pd.read_csv(feature_names_path, header=None)
    feature_names = feature_names_df[0].tolist()
    print(f" {len(feature_names)} features")
else:
    feature_names = [f"Feature_{i}" for i in range(len(gray_corr))]

def shadow_set_objective(alpha, beta, gamma):

    significant = gamma[gamma >= alpha]
    insignificant = gamma[gamma <= beta]
    shadow = gamma[(gamma > beta) & (gamma < alpha)]

    V1 = np.sum(1-significant)
    V2 = np.sum(insignificant)
    V3 = len(shadow)

    return abs(V1 + V2 - V3)

def optimize_thresholds(gamma, step=0.005):
    min_objective = float('inf')
    best_alpha, best_beta = 0, 0
    history = []

    gamma_min = np.min(gamma)
    gamma_max = np.max(gamma)

    thresholds = np.arange(gamma_min +step , gamma_max, step)

    for i, beta in enumerate(thresholds):
        for j, alpha in enumerate(thresholds[i + 1:], i + 1):
            objective = shadow_set_objective(alpha, beta, gamma)
            history.append((alpha, beta, objective))

            if objective < min_objective:
                min_objective = objective
                best_alpha = alpha
                best_beta = beta

    return best_alpha, best_beta, min_objective, history

# 主程序调用修改
alpha, beta, min_obj, history = optimize_thresholds(gamma, step=0.005)
print(f"Beat: α={alpha:.3f}, β={beta:.3f}, min={min_obj:.3f}")

# 画图
def select_features(features, gamma, alpha, beta, feature_names):
    feature_mask = np.zeros(len(gamma), dtype=int)

    uncertain_mask = (gamma > beta) & (gamma < alpha)
    feature_mask[uncertain_mask] = 1

    significant_mask = gamma >= alpha
    feature_mask[significant_mask] = 2

    selected_features = features[:, significant_mask]

    return selected_features, feature_mask

alpha, beta, min_obj = optimize_thresholds(gamma, step=0.005)
print(f"Best: α={alpha:.3f}, β={beta:.3f}, min={min_obj:.3f}")

selected_features, feature_mask = select_features(
    features, gamma, alpha, beta, feature_names
)

significant_count = np.sum(feature_mask == 2)
uncertain_count = np.sum(feature_mask == 1)
insignificant_count = np.sum(feature_mask == 0)

print(f" {significant_count} (gamma >= {alpha:.3f})")
print(f" {uncertain_count} ({beta:.3f} < gamma < {alpha:.3f})")
print(f" {insignificant_count} (gamma <= {beta:.3f})")