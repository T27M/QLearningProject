import PIL
from PIL import Image
import numpy as np
import time
import cv2
import sys
import os
from core.configbase import ConfigBase

# RGB Colors
PACMAN_COLOR = (210, 164, 74)  # Orange/Yellow

WALL_AND_PICKUP_COLOR = (228, 111, 111)  # Redish
BACKGROUND_COLOR = (0, 28, 136)  # Dark Blue

GHOST_RED = (200, 72, 72)
GHOST_ORANGE = (180, 122, 48)
GHOST_BLUE = (84, 184, 153)
GHOST_PURPLE = (198, 89, 179)

# Grey Colors
PACMAN_GREY = (167, 255)

GHOST_RED_G = (110, 255)
GHOST_ORANGE_G = (130, 255)
GHOST_BLUE_G = (150, 255)
GHOST_PURPLE_G = (131, 255)

WALL_AND_PICKUP_COLOR_G = (145, 255)
BACKGROUND_COLOR_G = (31, 255)

pickup_height = 2
power_pickup_height = 7
pickup_width = 3
min_pickup_x = 8
max_pickup_x = 149

min_pickup_y = 7
max_pickup_y = 164

# First 4 power-up, rest are fruit
pickup = [(8, 15), (8, 147), (148, 146), (148, 14),
          (8, 7), (8, 31), (8, 55), (8, 103), (8, 127),
          (8, 139), (8, 163), (16, 7), (16, 31), (16, 43),
          (16, 55), (16, 67), (16, 79), (16, 91), (16, 103),
          (16, 115), (16, 127), (16, 163), (24, 7), (24, 31),
          (24, 79), (24, 127), (24, 163), (32, 7), (32, 19),
          (32, 31), (32, 43), (32, 55), (32, 79), (32, 103),
          (32, 115), (32, 127), (32, 139), (32, 151), (32, 163),
          (40, 31), (40, 55), (40, 79), (40, 103), (40, 163),
          (48, 7), (48, 19), (48, 31), (48, 55), (48, 79),
          (48, 103), (48, 115), (48, 127), (48, 139), (48, 163),
          (56, 7), (56, 31), (56, 43), (56, 55), (56, 67),
          (56, 79), (56, 91), (56, 103), (56, 139), (56, 163),
          (64, 7), (64, 31), (64, 55), (64, 103), (64, 115),
          (64, 127), (64, 139), (64, 151), (64, 163), (72, 7),
          (72, 31), (72, 55), (72, 103), (72, 127), (72, 163),
          (84, 7), (84, 31), (84, 103), (84, 127), (84, 163),
          (92, 7), (92, 31), (92, 55), (92, 103), (92, 115),
          (92, 127), (92, 139), (92, 151), (92, 163), (100, 7),
          (100, 31), (100, 43), (100, 55), (100, 67), (100, 79),
          (100, 91), (100, 103), (100, 139), (100, 163), (108, 7),
          (108, 19), (108, 31), (108, 55), (108, 79), (108, 103),
          (108, 115), (108, 127), (108, 139), (108, 163), (116, 31),
          (116, 55), (116, 79), (116, 103), (116, 163), (124, 7),
          (124, 19), (124, 31), (124, 43), (124, 55), (124, 79),
          (124, 103), (124, 115), (124, 127), (124, 139), (124, 151),
          (124, 163), (132, 7), (132, 31), (132, 79), (132, 127),
          (132, 163), (140, 7), (140, 31), (140, 43), (140, 55),
          (140, 67), (140, 79), (140, 91), (140, 103), (140, 115),
          (140, 127), (140, 163), (148, 7), (148, 31), (148, 55),
          (148, 103), (148, 127), (148, 139), (148, 163)]

# [(41, (167, 255)), (60, (110, 255)), (60, (130, 255)), (160, (0, 255))]


class FeatureProcessor(ConfigBase):
    def __init__(self, config):
        super().__init__(config)

        self.__play_area_width = self._config['play_area_width']
        self.__play_area_height = self._config['play_area_height']

    def __find_grey(self, img, color):
        x_bit = color[0]
        y_bit = color[1]

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                x_val, y_val = img.getpixel((x, y))
                if x_val == x_bit and y_val == y_bit:
                    return (x, y)

    # Extract feature vector
    def extract_features(self, observation):

        # Parse image and crop
        img = Image.fromarray(observation, 'RGB')

        img.show()
        input('')

        img = img.convert('LA')
        img = img.crop((0, 0, self.__play_area_width, self.__play_area_height))

        # Find pickups
        pickup_vector = self.__check_pickup_squares(img)

        if(True):
            # Find Pacman
            pixel_pacman = self.__find_grey(img, PACMAN_GREY)

            # Find Ghosts
            pixel_red_ghost = self.__find_grey(img, GHOST_RED_G)
            pixel_orange_ghost = self.__find_grey(img, GHOST_ORANGE_G)
            pixel_blue_ghost = self.__find_grey(img, GHOST_BLUE_G)
            pixel_purple_ghost = self.__find_grey(img, GHOST_PURPLE_G)

            if self._debug:
                print(self.__get_colors(img))
                print("Pacman pixel at: " + str(pixel_pacman))

                if pixel_red_ghost is not None:
                    print("Red Ghost pixel at: " + str(pixel_red_ghost))
                else:
                    print("Red Ghost flicker")

                if pixel_orange_ghost is not None:
                    print("Orange Ghost pixel at: " + str(pixel_orange_ghost))
                else:
                    print("Orange Ghost flicker")

                if pixel_blue_ghost is not None:
                    print("Blue Ghost pixel at: " + str(pixel_blue_ghost))
                else:
                    print("Blue Ghost flicker")

                if pixel_purple_ghost is not None:
                    print("Purple Ghost pixel at: " + str(pixel_purple_ghost))
                else:
                    print("Purple Ghost flicker")

            if pixel_pacman is None:
                pixel_pacman = (0, 0)

            if pixel_red_ghost is None:
                pixel_red_ghost = (0, 0)

            if pixel_orange_ghost is None:
                pixel_orange_ghost = (0, 0)

            if pixel_blue_ghost is None:
                pixel_blue_ghost = (0, 0)

            if pixel_purple_ghost is None:
                pixel_purple_ghost = (0, 0)

            entities = [pixel_pacman[0], pixel_pacman[1],
                        pixel_red_ghost[0], pixel_red_ghost[1],
                        pixel_orange_ghost[0], pixel_orange_ghost[1],
                        pixel_blue_ghost[0], pixel_blue_ghost[1],
                        pixel_purple_ghost[0], pixel_purple_ghost[1]]

        return entities + pickup_vector

    def __check_pickup_squares(self, frame):
        pickups = []

        for i in range(len(pickup)):

            p = pickup[i]

            if self.__is_wall_or_pickup(frame, p[0], p[1]):
                pickups.append(1)
            else:
                pickups.append(0)

        return pickups

    def __find_pickup_sqaures(self, frame):
        # Top left pixel
        pickup_location = []

        # Start top left and scan vertically
        for x in range(min_pickup_x, max_pickup_x):
            for y in range(min_pickup_y, max_pickup_y):

                if not self.__is_wall_or_pickup(frame, x, y):
                    continue

                # Check surrounding pixels
                up_is_bg = self.__is_background(frame, x, y - 1)
                left_is_bg = self.__is_background(frame, x - 1, y)

                # Width check
                w_is_bg = not self.__is_wall_or_pickup(
                    frame, x + pickup_width + 1, y)

                if not up_is_bg or not left_is_bg or not w_is_bg:
                    continue

                # Height check
                h_is_bg = self.__is_background(frame, x, y + pickup_height)

                # Powerup height check
                #p_h_is_bg = self.is_background(frame, x, y + power_pickup_height + 1)

                if h_is_bg:
                    #print("Pickup: " + str(x) + " " + str(y))
                    pickup_location.append((x, y))

        return pickup_location

    def __is_wall_or_pickup(self, frame, x, y):
        x_val, y_val = frame.getpixel((x, y))
        return self.__is_color(x_val, y_val, WALL_AND_PICKUP_COLOR_G)

    def __is_background(self, frame, x, y):
        x_val, y_val = frame.getpixel((x, y))
        return self.__is_color(x_val, y_val, BACKGROUND_COLOR_G)

    def __is_color(self, x_val, y_val, color):
        return color[0] == x_val and color[1] == y_val

    # Helper method to identify color codes
    def __get_colors(self, img):
        colors = img.getcolors()
        colors = sorted(colors)
        colors = [i for i in colors if i[0] > 300]
        print(colors)
