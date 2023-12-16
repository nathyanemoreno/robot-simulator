import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


def create_maze_from_image(image_path):
    # Read the original image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with zeros (black)
    mask = np.zeros_like(gray)

    # Draw the main contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND the original image with the mask
    result_main = cv2.bitwise_and(image, image, mask=mask)

    # Create a new mask for the inner contours
    inner_mask = np.zeros_like(gray)

    # Identify inner contours based on hierarchy
    for i, contour in enumerate(contours):
        # Check if the contour has a parent (inner contour)
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(inner_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Bitwise AND the original image with the inner mask
    result_inner = cv2.bitwise_and(image, image, mask=inner_mask)

    # Merge the main and inner contours
    result_merged = cv2.add(result_main, result_inner)

    # Display the results
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(result_main, cv2.COLOR_BGR2RGB))
    plt.title('Main Contour')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(result_inner, cv2.COLOR_BGR2RGB))
    plt.title('Inner Contour')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_merged, cv2.COLOR_BGR2RGB))
    plt.title('Merged Contours')

    plt.show()



    original_image = Image.open(image_path)

    # Step 2: Convert to grayscale
    grayscale_image = original_image.convert('L')

    # Step 3: Thresholding
    threshold_value = 128
    binary_image = grayscale_image.point(
        lambda x: 255 if x > threshold_value else 0)

    # Step 4: Analyze Maze Structure (replace with your own analysis)
    maze_structure = analyze_maze_structure(binary_image)

    return maze_structure


def analyze_maze_structure(binary_image):
    # Placeholder for maze analysis
    # You can implement your own image processing techniques here
    # to extract the maze structure and create a representation.

    # Example: Convert the binary image to a NumPy array
    maze_array = np.array(binary_image)

    # Placeholder: Perform your own analysis based on the maze_array

    return maze_array


if __name__ == "__main__":
    image_path = "recorded.png"
    maze_representation = create_maze_from_image(image_path)

    # Print or use maze_representation as needed
    print(maze_representation)
