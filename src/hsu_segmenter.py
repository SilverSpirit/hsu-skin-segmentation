import numpy as np
import cv2
import consts as *

class HsuSegmenter:
    def __init__(self):
        pass # nothing for now
    
    def get_mask(self, img):
        return img # this will return the mask, currently returns the image


    def get_center_b_chroma(self, Y):
        center = 108
        if Y < kl:
            center += ((kl - Y) * (118-108)) / (kl - ymin)
        elif kh < Y:
            center += ((Y - kh) * (118 - 108)) / (ymax - kh)

        return center