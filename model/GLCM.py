import pandas as pd
import os
import numpy as np
from skimage import img_as_ubyte, io
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def extract_texture_features(image_path):
    image_array = io.imread(image_path, as_gray=True)

    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        image_array = img_as_ubyte(image_array)
    elif len(image_array.shape) > 2:
        image_array = np.mean(image_array, axis=-1).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image_array, distances=distances, angles=angles, symmetric=True, normed=True)

    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        prop_values = graycoprops(glcm, prop)
        avg_prop_values = np.mean(prop_values)
        features[prop] = avg_prop_values

    hist, _ = np.histogram(image_array, bins=np.arange(0, 257))

    prob = hist / np.sum(hist)
    total_entropy = -np.sum(prob * np.log2(prob + 1e-10))
    features['total_entropy'] = total_entropy

    local_entropy = entropy(image_array, disk(5))
    features['local_entropy_mean'] = np.mean(local_entropy)

    max_correlation, _ = pearsonr(image_array.flatten(), local_entropy.flatten())
    features['max_correlation'] = max_correlation

    mutual_information = mutual_info_regression(image_array.flatten().reshape(-1, 1), local_entropy.flatten())[0]
    features['mutual_information'] = mutual_information
    mean = np.mean(image_array)
    variance = np.var(image_array)
    sum_mean = np.sum(image_array)
    sum_variance = np.var(np.sum(image_array, axis=1))
    sum_entropy = -np.sum(image_array * np.log(image_array + np.finfo(float).eps))
    diff_variance = np.var(np.diff(image_array))
    diff_entropy = -np.sum(np.diff(image_array) * np.log(np.diff(image_array) + np.finfo(float).eps))

    features['mean'] = mean
    features['variance'] = variance
    features['sum_mean'] = sum_mean
    features['sum_variance'] = sum_variance
    features['sum_entropy'] = sum_entropy
    features['diff_variance'] = diff_variance
    features['diff_entropy'] = diff_entropy

    return features



image_dir = r"D:\NWPU\NWPU10"

# 提取特征
feature_list = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        image_path = os.path.join(image_dir, filename)
        print(f"Processing: {image_path}")
        try:
            features = extract_texture_features(image_path)
            features['Filename'] = filename
            feature_list.append(features)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Skipping {filename}, not a valid image file.")

df = pd.DataFrame(feature_list)

cols = ['Filename'] + [col for col in df.columns if col != 'Filename']
df = df[cols]


excel_file = r"D:\NWPU\NWPU100\NWPU100_GLCM.xlsx"

try:
    df.to_excel(excel_file, index=False, engine='openpyxl')
except Exception as e:
    print(f"Error: {e}")
