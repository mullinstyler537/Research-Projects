import cv2
import numpy as np

#Loads the image in grayscale

def detect_edges(image_path, output_path='C:/Users/D00456326/Desktop/Research Photos/deerwf2 edges.png'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#Checks if the image is loaded properly

    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
#Crop bottom banner 
    h, w = image.shape
    crop_height = int(h * 0.90)  # keep top 90%
    image = image[:crop_height, :]
    
#Applies blur to reduce noise

    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

#Applying edge detection

    edges = cv2.Canny(blurred, 100, 200)

#Save the result
    cv2.imwrite(output_path, edges)
    print(f"Edges saved to {output_path}")


def count_holes(edge_image_path, min_contour_area=10):

#Loads the edged out image

    image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {edge_image_path}")
    
#Threshold the image to make sure it's binary

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#Finds the contours

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        print("No contours found.")
        return 0
    
#Counting the contours or "holes"

    holes = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_contour_area:
            continue
        parent = hierarchy[0][i][3]
        if parent != -1:
            holes += 1

    print(f"Number of pockets detected: {holes}")
    return holes

def get_hole_areas(edge_image_path, microns_per_pixel=1.0, min_contour_area=10):
    
    #Returns a list of areas (in µm²) for each detected hole in the edged image,
    #and prints & returns the average hole area.
    
    image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {edge_image_path}")
    
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        print("No contours found.")
        return [], 0.0

    hole_areas_microns = []

    for i, contour in enumerate(contours):
        area_pixels = cv2.contourArea(contour)
        if area_pixels < min_contour_area:
            continue
        parent = hierarchy[0][i][3]
        if parent != -1:
            area_microns = area_pixels * (microns_per_pixel ** 2)
            hole_areas_microns.append(area_microns)

    if hole_areas_microns:
        avg_area = sum(hole_areas_microns) / len(hole_areas_microns)
    else:
        avg_area = 0.0

    #print(f"Detected {len(hole_areas_microns)} pockets.")
    for idx, area in enumerate(hole_areas_microns, 1):
        print(f"Pocket {idx}: {area:.2f} µm²")

    print(f"Average pocket area: {avg_area:.2f} µm²")

    return hole_areas_microns, avg_area


#Calling the function

image_input_path = "C:/Users/D00456326/Desktop/Research Photos/antsf1.JPG"
edge_output_path = "C:/Users/D00456326/Desktop/Research Photos/antsf1 edges.png"

detect_edges(image_input_path, edge_output_path)
count_holes(edge_output_path)
get_hole_areas(edge_output_path, microns_per_pixel=0.46)



