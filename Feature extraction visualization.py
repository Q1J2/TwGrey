import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from scipy.stats import skew
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import matplotlib as mpl
import cv2  
import datetime  
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12


def load_and_prepare_image(image_path):
    if not os.path.exists(image_path):


        raise FileNotFoundError(f"wrong: no- {image_path}")
    if os.path.getsize(image_path) == 0:
        raise ValueError(f"wrong: no - {image_path}")

    img = cv2.imread(image_path)
    if img is not None:
        print(f"OpenCV: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(image_path)
        if img is not None:
            print(f"matplotlib: {image_path}")
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img
    except:
        pass

    try:
        from PIL import Image
        img = Image.open(image_path)
        if img is not None:
            print(f"PIL: {image_path}")
            return np.array(img)
    except:
        pass

    raise RuntimeError(f"no: {image_path}")

def hsv_color_moments_analysis(img, save_dir=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    moments = {}
    for i, channel in enumerate(['Hue', 'Saturation', 'Value']):
        c = [h, s, v][i]
        moments[channel] = {
            'Mean': np.mean(c),
            'Std': np.std(c),
            'Skewness': skew(c.flatten())
        }


    fig = plt.figure(figsize=(12, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.3)

    grid[0].imshow(img)
    grid[0].set_title('Original Image')
    grid[0].axis('off')

    titles = ['Hue Channel', 'Saturation Channel', 'Value Channel']
    channels = [h, s, v]

    for i in range(3):
        grid[i + 1].imshow(channels[i], cmap='viridis' if i == 0 else 'gray')
        grid[i + 1].set_title(titles[i])
        grid[i + 1].axis('off')

    plt.tight_layout()


    if save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"hsv_analysis_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" HSV: {save_path}")
    plt.close(fig)

    print("\nHSV Color Moments:")
    print("{:<12} {:<10} {:<10} {:<10}".format('Channel', 'Mean', 'Std Dev', 'Skewness'))
    for channel in ['Hue', 'Saturation', 'Value']:
        print("{:<12} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            channel,
            moments[channel]['Mean'],
            moments[channel]['Std'],
            moments[channel]['Skewness']))

    return moments


def glcm_texture_analysis(img, save_dir=None):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    texture_features = {}

    for prop in properties:
        texture_features[prop] = graycoprops(glcm, prop)

    fig = plt.figure(figsize=(10, 15))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.3)

    grid[0].imshow(img)
    grid[0].set_title('Original Image')
    grid[0].axis('off')

    gray_img_ax = grid[1]
    gray_img_ax.imshow(gray, cmap='gray')
    gray_img_ax.set_title('Grayscale Image')
    gray_img_ax.axis('off')

    glcm_matrix = glcm[:, :, 0, 0]
    ax = grid[2]
    sns.heatmap(glcm_matrix[:16, :16], cmap='viridis', ax=ax, cbar=False)
    ax.set_title('GLCM Matrix (d=1, θ=0°)')
    ax.set_xlabel('Gray Level j')
    ax.set_ylabel('Gray Level i')

    directions = ['0°', '45°', '90°', '135°']
    for i in range(4):
        if i < 3:
            ax = grid[i + 3]
            glcm_matrix = glcm[:, :, 0, i]
            sns.heatmap(glcm_matrix[:16, :16], cmap='viridis', ax=ax, cbar=False)
            ax.set_title(f'GLCM at {directions[i]}')
            ax.set_xlabel('Gray Level j')
            ax.set_ylabel('Gray Level i')
            ax.tick_params(axis='x', labelrotation=0)
            ax.tick_params(axis='y', labelrotation=0)

    plt.tight_layout()

    if save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"glcm_analysis_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"save: {save_path}")
    plt.close(fig)

    print("\nGLCM Texture Features:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format('Feature', '0°', '45°', '90°', '135°'))
    for prop in properties:
        values = texture_features[prop][0]
        print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            prop, values[0], values[1], values[2], values[3]))

    return texture_features

if __name__ == "__main__":
    SAVE_DIR = r"D:\visualization_results"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f": {SAVE_DIR}")

    image_path = r"D:\NWPU\island\island_053.jpg"

    try:
        print(f": {image_path}")
        img = load_and_prepare_image(image_path)
        print(f": {img.shape}")

        print("\nPerforming HSV Color Moments Analysis...")
        color_moments = hsv_color_moments_analysis(img, save_dir=SAVE_DIR)

        print("\nPerforming GLCM Texture Analysis...")
        texture_features = glcm_texture_analysis(img, save_dir=SAVE_DIR)

    except Exception as e:
        print(f": {str(e)}")

        img = data.brick()
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)

        print("\nPerforming HSV Color Moments Analysis...")
        color_moments = hsv_color_moments_analysis(img, save_dir=SAVE_DIR)

        print("\nPerforming GLCM Texture Analysis...")
        texture_features = glcm_texture_analysis(img, save_dir=SAVE_DIR)