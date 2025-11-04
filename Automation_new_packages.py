import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, color, exposure
import pandas as pd
import os

def analyze_sem_particles_adaptive(
    image_path,
    microns_per_pixel=0.173,
    min_area_um2=0.5,
    max_area_um2=7.5,
    block_size=51,  # Must be odd, try tuning 35-75
    C=5            # Constant subtracted from mean
):
    # Load grayscale and crop bottom 10%
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    h, w = img.shape
    img = img[:int(h * 0.9), :]

    # Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    # Adaptive thresholding (local)
    binary = cv2.adaptiveThreshold(
        img_eq,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )

    # Morphological cleanup
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=15)
    binary = morphology.remove_small_holes(binary, area_threshold=15)
    binary = morphology.binary_closing(binary, morphology.disk(1))

    # Label connected components
    labels = measure.label(binary)

    # Filter by area
    areas_um2 = []
    valid_labels = []
    for region in measure.regionprops(labels):
        area_um2 = region.area * (microns_per_pixel ** 2)
        if min_area_um2 <= area_um2 <= max_area_um2:
            areas_um2.append(area_um2)
            valid_labels.append(region.label)

    # Filter mask with valid particles only
    valid_mask = np.isin(labels, valid_labels)
    filtered_labels = measure.label(valid_mask)

    n_particles = len(areas_um2)
    avg_area = np.mean(areas_um2) if n_particles > 0 else 0.0

    # Visualization overlay (RGB)
    overlay = color.gray2rgb(exposure.equalize_hist(img).astype(np.float32))
    boundaries = morphology.dilation(filtered_labels > 0, morphology.disk(1)) ^ (filtered_labels > 0)
    overlay[boundaries, 0] = 1.0  # Red channel
    overlay[boundaries, 1] = 0.0
    overlay[boundaries, 2] = 0.0

    plt.figure(figsize=(8,8))
    plt.imshow(overlay)
    plt.title(f"Particles: {n_particles}, Avg Area: {avg_area:.3f} µm²\nAdaptive block: {block_size}, C: {C}")
    plt.axis('off')
    plt.show()

    # Save CSV
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_results_adaptive.csv")

    df = pd.DataFrame({'Particle_ID': np.arange(1, n_particles+1), 'Area_um2': areas_um2})
    summary = pd.DataFrame({'Total_Particles': [n_particles], 'Mean_Area_um2': [avg_area]})

    with open(csv_path, 'w', newline='') as f:
        summary.to_csv(f, index=False)
        f.write('\n')
        df.to_csv(f, index=False)

    print(f"\nResults saved to: {csv_path}")
    print(f"Detected particles: {n_particles}")
    print(f"Average particle area: {avg_area:.4f} µm²")

    return filtered_labels, areas_um2, avg_area


# Example usage:
if __name__ == "__main__":
    image_input_path = "C:/Users/D00456326/Desktop/Research Photos/Nanoparticles/NOCOB2.jpg"
    analyze_sem_particles_adaptive(image_input_path)
