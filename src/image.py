import cv2
import numpy as np
import matplotlib.pyplot as plt

# A1E2A0


def hex_to_hsv(hex_color):
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to BGR
    bgr = np.array([int(hex_color[i:i+2], 16)
                   for i in (0, 2, 4)][::-1])  # Reverse order for BGR

    # Convert BGR to HSV
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    print(hsv)
    return hsv


def get_contours_on_image(image, lower_bound, upper_bound):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = hex_to_hsv(lower_bound)
    upper_bound = hex_to_hsv(upper_bound)

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the red color mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_map_contours(image_path):
    # Read the original image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edges
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with zeros (black)
    mask = np.zeros_like(gray)

    # Draw the main contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND the original image with the mask
    result_main = cv2.bitwise_and(image, image, mask=mask)
    return (result_main, mask)


def find_red_square(image):
    contours = get_contours_on_image(image, '#FF0000', '#8A2410')

    # Filter contours to find squares (or rectangles)
    red_squares = []
    center = ()
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        print(approx)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate the aspect ratio of the contour's bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Tweak the aspect ratio threshold based on your specific case
            if 0.8 <= aspect_ratio <= 1.1:
                red_squares.append(approx)

                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    center = (cX, cY)

    # Draw the red squares on a copy of the original image
    image_with_red_squares = image.copy()
    cv2.drawContours(image_with_red_squares, red_squares, -
                     1, (0, 255, 0), cv2.FILLED)

    return (image_with_red_squares, center)


def find_blue_balls(image):
    contours = get_contours_on_image(image, '#00aaff', '5500ff')

    blue_balls = []
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > 0.5:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            centers.append((cX, cY))

            blue_balls.append(contour)

    image_with_blue_balls = image.copy()
    cv2.drawContours(image_with_blue_balls, blue_balls, -
                     1, (0, 255, 0), cv2.FILLED)

    return (image_with_blue_balls, centers)


def find_origin_and_objective(image, square_center):
    contours = get_contours_on_image(image, '#B5FEB4', '#B5FEB4')

    # Filter contours to find squares (or rectangles)
    areas = []
    objectives = []
    origin_with_contours = image.copy()

    # Create a blank image for the result
    result_main = np.zeros_like(image)

    for i, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            areas.append(approx)

        result = cv2.pointPolygonTest(approx.astype(int), square_center, False)

        if result >= 0:
            origin = approx
            color = (255, 255, 0)
        else:
            objectives.append(contour)
            color = (0, 255, 255)

        # Draw the contours on the result image
        result_main = cv2.drawContours(
            result_main, [contour], -1, color, cv2.FILLED)

    # Overlay the contours on the original image
    final_result = cv2.addWeighted(image, 1, result_main, 0.5, 0)

    return (final_result, origin, objectives)


f0, _ = get_map_contours('assets/f0.png')
f1, _ = get_map_contours('assets/f1.png')
f2, _ = get_map_contours('assets/f2.png')
f0_square, f0_square_center = find_red_square(f0)
f1_square, f1_square_center = find_red_square(f1)
f2_square, f2_square_center = find_red_square(f2)
# with_balls, centers = find_blue_balls(map)
# img, origin, objectives = find_origin_and_objective(square, square_center)


print(f0_square_center, f1_square_center, f2_square_center)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(f0_square, cv2.COLOR_BGR2RGB))
plt.title('Red Squares')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(f1_square, cv2.COLOR_BGR2RGB))
plt.title('Blue Balls')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(f2_square, cv2.COLOR_BGR2RGB))
plt.title('Blue Balls')

plt.show()
