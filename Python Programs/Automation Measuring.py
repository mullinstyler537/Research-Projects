from tkinter import Tk, filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- File picker ---
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select SEM image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")]
)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

# Optional: crop image to remove scale bar or edges
height = img.shape[0]
crop_height = int(height * 0.9)
img = img[:crop_height, :]

# --- Ask user for scale information ---
micron_scale = float(input("Enter the scale bar length in microns: "))
pixel_scale = float(input("Enter the corresponding pixel length: "))

nm_per_pixel = (micron_scale * 1000) / pixel_scale

# --- Preprocessing ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# High contrast: threshold to detect white nanoparticles
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Optional: remove tiny noise
kernel = np.ones((2,2), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# --- Edge detection ---
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite("sem_edges_debug.jpg", edges)

# Convert edges to a 3-channel image for overlay
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges_color[:] = (0, 255, 0)  # color edges green

# --- Find contours ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area
min_area_px = 2  # adjust based on smallest nanoparticle
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_px]

# Draw contours on a copy of the original image
img_with_contours = img.copy()
for cnt in filtered_contours:
    cv2.drawContours(img_with_contours, [cnt], -1, (0,0,255), 1)  # red contours

# Overlay edges on the original image
overlay = cv2.addWeighted(img_with_contours, 0.8, edges_color, 0.5, 0)
cv2.imwrite("sem_edges_and_contours_overlay.jpg", overlay)

# --- Calculate areas ---
areas_px = [cv2.contourArea(cnt) for cnt in filtered_contours]
areas_nm2 = np.array(areas_px) * (nm_per_pixel ** 2)
areas_um2 = areas_nm2 / 1e6

# --- Statistics ---
if len(areas_um2) > 0:
    avg_area = np.mean(areas_um2)
    std_area = np.std(areas_um2)
else:
    avg_area = 0
    std_area = 0

# --- Outputs ---
cv2.imwrite("sem_nanoparticles_outlined.jpg", img_with_contours)

print(f"\nDetected {len(areas_um2)} nanoparticles")
print(f"Average area: {avg_area:.4f} µm²")
print(f"Std. dev.:    {std_area:.4f} µm²")

# --- Histogram ---
plt.figure(figsize=(6,4))
plt.hist(areas_um2, bins=30, color='steelblue', edgecolor='black')
plt.xlabel("Particle area (µm²)")
plt.ylabel("Count")
plt.title("Nanoparticle Size Distribution")
plt.tight_layout()
plt.savefig("nanoparticle_size_distribution.png", dpi=300)
# plt.show()  # comment out for automated runs

