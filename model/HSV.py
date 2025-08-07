import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

def calculate_hsv_moments(image_path):
    filename = os.path.basename(image_path)

    try:
        with Image.open(image_path) as img:
            image = np.array(img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error:  {image_path} PIL  {e}")
        return [], filename

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)

    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)

    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)

    feature_vector = [h_mean, s_mean, v_mean, h_std, s_std, v_std, h_thirdMoment, s_thirdMoment, v_thirdMoment]

    return feature_vector, filename

image_folder = r"D:\NWPU\NWPU100"
excel_file = r"D:\NWPU\NWPU100\NWPU100_HSV.xlsx"

all_moments = []

for image_filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_filename)
    print(f"Processing: {image_path}")

    if os.path.isfile(image_path) and image_path.lower().endswith(('.tif', '.jpg', '.jpeg', '.bmp')):

        moments, filename = calculate_hsv_moments(image_path)
        if moments:
            all_moments.append([filename] + moments)
        else:
            print(f"Skipping {filename} due to loading error.")
    else:
        print(f"Skipping {image_filename}, not a valid image file.")

columns = ['Filename', 'Hue_Mean', 'Saturation_Mean', 'Value_Mean',
           'Hue_Std', 'Saturation_Std', 'Value_Std',
           'Hue_ThirdMoment', 'Saturation_ThirdMoment', 'Value_ThirdMoment']
df = pd.DataFrame(all_moments, columns=columns)

try:
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print("Excel:", excel_file)
except Exception as e:
    print(f"Error:  Excel : {e}")






