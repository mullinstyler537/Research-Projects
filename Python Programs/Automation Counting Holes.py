import cv2
import numpy as np

#Loads the image in grayscale

def detect_edges(image_path, output_path='C:/Users/D00456326/Desktop/Research Photos/antsf1 edges.png'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#Checks if the image is loaded properly

    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
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

#Calling the function

image_input_path = "C:/Users/D00456326/Desktop/Research Photos/antsf1.JPG"
edge_output_path = "C:/Users/D00456326/Desktop/Research Photos/antsf1 edges.png"

detect_edges(image_input_path, edge_output_path)
count_holes(edge_output_path)
