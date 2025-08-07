import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 16

images = [
    (r"D:\NWPU\NWPU40airplane_209.jpg", "Airplane"),
    (r"D:\NWPU\NWPU40airport_128.jpg", "Airport"),
]

fig, axs = plt.subplots(1, 2, figsize=(12,6))

for i, (image_path, label) in enumerate(images):
    img = Image.open(image_path)
    ax = axs[i]

    ax.imshow(img)
    ax.axis("off")

    aspect_ratio = img.width / img.height

    ax.text(0.5, -0.05, f"{label}",
            fontsize=14,
            ha='center',
            va='top',
            transform=ax.transAxes)

plt.subplots_adjust(
    wspace=0.05,
    hspace=0.15,
    left=0.05,
    right=0.95,
    bottom=0.1,
    top=0.95
)

for ax in axs.flat:
    ax.set_aspect('auto')

plt.savefig(r"D:\output_image2.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()