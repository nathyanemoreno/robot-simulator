from io import BytesIO
import os
from matplotlib import contour
import matplotlib.pyplot as plt
from math import log
import cv2
from matplotlib.colors import hex2color
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import pyautogui
import imageio
import numpy as np
import subprocess
from PIL import Image

SCREENSHOT_TIME = 0.03


class Environment:
    def __init__(self, image=None):
        self.state = image
        self.driver = None

    def init_browser(self, url, div_xpath):
        # Create a new Firefox profile
        # firefox_profile = self.create_firefox_profile()

        # Specify the new profile for the Firefox WebDriver
        # firefox_options = webdriver.FirefoxOptions()
        # firefox_options.profile = firefox_profile

        options = Options()
        options.set_preference("media.webspeech.synth.enabled", False)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument("--disable-notifications")

        # specify the path to your geckodriver
        service = Service(executable_path="/home/nathyane/bin/geckodriver")
        self.driver = webdriver.Firefox(options=options, service=service)

        # Set the window size (width x height)
        self.driver.set_window_size(800, 600)

        # Set the window position (x, y)
        self.driver.set_window_position(0, 0)

        # self.driver = webdriver.Firefox(options=options)
        self.driver.get(url)

        # XPath of the specific div to record
        self.div_xpath = div_xpath

    def create_firefox_profile():
        # Create a new Firefox profile using the -CreateProfile command-line option
        try:
            subprocess.run(['firefox', '-CreateProfile', 'web_robot_profile'])
            # The profile directory is typically in ~/.mozilla/firefox/
            profile_directory = '/home/nathyane/.mozilla/firefox/web_robot_profile'
        except Exception as e:
            print(f"Error during screen processing: {e}")
        return webdriver.FirefoxProfile(profile_directory)

    def read_screen(self):
        try:
            element = self.driver.switch_to.active_element
            location = element.location
            size = element.size
            left = int(location['x']) + 20
            top = int(location['y']) + 40

            # width = int(size['width']) - 40
            # height = int(size['height']) - 80
            right = left + size['width'] - 40
            bottom = top + size['height'] - 80

            # screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot = self.driver.get_screenshot_as_png()
            screenshot = Image.open(BytesIO(screenshot))
            screenshot = screenshot.crop((left, top, right, bottom))

            # Convert the screenshot to a NumPy array
            image_array = np.array(screenshot)

            self.state = image_array

            # Convert NumPy array to PIL Image
            # image = Image.fromarray(image_array)

            # image.save('recorded.png')
            # # Append the frame to the video
            # self.video_writer.append_data(frame)
            return image_array
        except Exception as e:
            print(f"Error during screen processing: {e}")

    def start_play(self):
        try:
            print("STARTING")
            ActionChains(self.driver).click().perform()
            self.driver.switch_to.parent_frame()
        except:
            raise Exception("Could not enter game")

    def press_arrow_key(self, key):
        actions = ActionChains(self.driver)
        actions.key_down(key).perform()

    def release_key(self, key):
        actions = ActionChains(self.driver)
        actions.key_up(key).perform()

    def run_simulation(self):
        # Wait for "Y" input to begin movement
        is_ready = False
        while not is_ready:
            is_ready = input("Ready? (Y/N) ")
            is_ready = is_ready == "Y"

        try:
            while True:
                # Process the web page to extract relevant information
                self.read_screen()
                print("PROCESSING")
                self.press_arrow_key(Keys.ARROW_DOWN)

                # Adjust the delay based on the speed of your simulation
                time.sleep(SCREENSHOT_TIME)

        except KeyboardInterrupt:
            print("Simulation stopped.")
        finally:
            # Close the web driver when done
            # self.driver.quit()

            # Close the video writer
            # self.video_writer.close()
            print("END")

    def hex_to_hsv(self, hex_code):
        # Convert hex code to BGR
        bgr = np.uint8(
            [[[int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:], 16)]]])

        # Convert BGR to HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        return hsv[0, 0, :]

    def find_contours_by_color(self, image, hex_color, hue_tolerance=30):
        # Convert the image to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert hex code to HSV
        hsv_color = self.hex_to_hsv(hex_color)

        # Define a lower and upper bound for the color in HSV
        lower_bound = np.array([hsv_color[0] - hue_tolerance, 100, 100])
        upper_bound = np.array([hsv_color[0] + hue_tolerance, 255, 255])

        # Create a mask for the specified color
        mask = cv2.inRange(image_bgr, lower_bound, upper_bound)

        # Find contours in the masked image
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def __get_contours_on_image(self, image, lower_bound, upper_bound):
        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # # Define lower and upper bounds for red color in HSV
        # lower_red1 = np.array([0, 100, 100])
        # upper_red1 = np.array([10, 255, 255])
        # lower_red2 = np.array([160, 100, 100])
        # upper_red2 = np.array([180, 255, 255])

        # # Create masks for red color
        # mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine the masks
        # mask = cv2.bitwise_or(mask_red1, mask_red2)

        lower_bound = self.hex_to_hsv(lower_bound)
        upper_bound = self.hex_to_hsv(upper_bound)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours in the red color mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def get_map_contours(self, image, erosion_width):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edges
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with zeros (black)
        mask = np.zeros_like(gray)

        for i, contour in enumerate(contours):
            approx = cv2.approxPolyDP(
                contour, 0.01*cv2.arcLength(contour, True), True)

            contours_eroded = None
            if len(approx) >= 5:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Create an empty mask
                mask_yellow = np.zeros_like(gray)
                mask_red = np.zeros_like(gray)

                img = cv2.drawContours(
                    mask_yellow, contours, -1, (0, 255, 255), thickness=cv2.FILLED)

                # Create a mask for eroded contours
                for contour in contours:
                    # Create a separate mask for each contour
                    contour_mask = np.zeros_like(gray)
                    cv2.drawContours(
                        contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

                    # Apply erosion to the contour mask
                    contour_mask_eroded = cv2.erode(
                        contour_mask, None, iterations=erosion_width)

                    # Combine the eroded contour mask into the red mask
                    mask_red = cv2.bitwise_or(mask_red, contour_mask_eroded)

                    # Combine the yellow and red contours
                    result = cv2.bitwise_or(
                        image, image, mask=(mask_yellow | mask_red))

                    # Find contours in the eroded image
                    contours_eroded, _ = cv2.findContours(
                        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw the new contours on the result image
                    cv2.drawContours(result, contours_eroded, -1,
                                     (255, 0, 0), thickness=2)  # Draw in blue

                # Bitwise AND the original image with the mask
                return contours_eroded[0], img

    # def get_map_contours(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edges
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with zeros (black)
        mask = np.zeros_like(gray)

        for i, contour in enumerate(contours):
            approx = cv2.approxPolyDP(
                contour, 0.01*cv2.arcLength(contour, True), True)

            if len(approx) >= 5:
                img = cv2.drawContours(
                    mask, contour, -1, (0, 255, 255), thickness=cv2.FILLED)
                print(contour)
                return contour, img
            else:
                raise Exception("Could not determine map area")

    def find_red_square(self, image):
        # Convert image to HSV for better color manipulation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv_image, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

        # join my masks
        red_mask = mask0+mask1

        # Find contours in the red mask
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find and draw filled red squares
        red_squares = []
        img = image.copy()

        for contour in contours:
            center = ()
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Tweak the aspect ratio threshold based on your specific case
            if 0.9 <= aspect_ratio <= 1.1:
                red_squares.append((x, y, w, h))

                # Calculate the center of the square
                cX = x + w // 2
                cY = y + h // 2
                center = (cX, cY)

                # Draw filled red square
                cv2.rectangle(img, (x, y), (x + w, y + h),
                              (255, 0, 0), cv2.FILLED)

            return img, center, (w, h)

    def find_blue_balls(self, image):
        # Convert image to HSV for better color manipulation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define the lower and upper bounds for the blue color in HSV
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # Create a mask for blue pixels in the image
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Find contours in the blue mask
        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find and draw filled blue circles
        centers = []
        img = image.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > 0.5:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                centers.append((cX, cY))
                # print((cX, cY))

                # Draw filled blue circle
                cv2.circle(img, (cX, cY), int(
                    np.sqrt(area / np.pi)), (0, 0, 255), cv2.FILLED)

        return img, centers

    def find_spots(self, image):
        contours = self.__get_contours_on_image(image, '#B5FEB4', '#B5FEB4')

        # Filter contours to find squares (or rectangles)
        spots = []

        # Create a blank image for the result
        # result_main = np.zeros_like(image)
        img = image.copy()

        for i, contour in enumerate(contours):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                spots.append(approx)
            # Draw the contours on the result image
            cv2.drawContours(
                img, [contour], -1, (255, 255, 0), cv2.FILLED)

        # Overlay the contours on the original image
        img = cv2.addWeighted(image, 1, img, 0.5, 0)

        return img, spots

    def find_origin(self, spots, square_center):
        for i, spot in enumerate(spots):
            try:
                result = cv2.pointPolygonTest(
                    spot.astype(int), square_center, False)

                if result >= 0:
                    # Draw the contours on the result image
                    # result_main = cv2.drawContours(
                    #     result_main, [spot], -1, (255, 255, 0), cv2.FILLED)
                    return spot
            except Exception as e:
                # raise Exception("Could not find origin", e)
                print("Could not find origin")
                continue

    def find_safe_spots(self, spots, origin):
        return (600, 300)
        # if(origin in spots):

    def find_objectives(self, image):
        # Convert image to HSV for better color manipulation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define the lower and upper bounds for the yellow color in HSV
        lower_yellow = np.array([22, 93, 0])
        upper_yellow = np.array([45, 255, 255])

        # Create a mask for yellow pixels in the image
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Find contours in the yellow mask
        contours, _ = cv2.findContours(
            yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yellow_balls = []
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

                yellow_balls.append(contour)

        image_with_yellow_balls = image.copy()
        cv2.drawContours(image_with_yellow_balls, yellow_balls, -
                         1, (0, 255, 0), cv2.FILLED)

        return centers

    def find_origin_and_objective(self, image, square_center):
        contours = self.__get_contours_on_image(image, '#B5FEB4', '#B5FEB4')

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

            result = cv2.pointPolygonTest(
                approx.astype(int), square_center, False)

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

    def get_environment(self, image):
        square_img, red_square_center, _ = self.find_red_square(image)
        obstacles_img, obstacles = self.find_blue_balls(square_img)
        _, spots = self.find_spots(image)
        origin = self.find_origin(spots, red_square_center)
        objectives = self.find_objectives(image)
        safe_spots = self.find_safe_spots(spots, origin)
        # print("Square: ", red_square_center)
        # print("Obstacles: ", blue_balls_centers)
        # print("Origin: ", origin)
        # print("Objectives: ", objectives)
        # print("Obstacles: ", obstacles)
        return red_square_center, safe_spots, obstacles
    # blue_balls_centers, spots, origin, objectives
        # return obstacles_img
