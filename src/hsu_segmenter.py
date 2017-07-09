import numpy as np
import cv2
from constants import *

class HsuSegmenter:
    def __init__(self):
        pass # nothing for now
    
    def get_mask(self, img):
        ycrcb_img = self.preprocess(img)
        return img # this will return the mask, currently returns the image
    
    def preprocess(self, img):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb_img)
        
        # ranges from https://en.wikipedia.org/wiki/YCbCr#YCbCr
        cv2.normalize(y, y, 16, 235, cv2.NORM_MINMAX) 
        cv2.normalize(cr, cr, 16, 240, cv2.NORM_MINMAX)
        cv2.normalize(cb, cb, 16, 240, cv2.NORM_MINMAX)
        return cv2.merge((y, cr, cb))

    # def get_center_b_chroma(self, Y): ## CORRECT THIS
        # center = 108
        # if Y < kl:
            # center += ((kl - Y) * (118-108)) / (kl - ymin)
        # elif kh < Y:
            # center += ((Y - kh) * (118 - 108)) / (ymax - kh)
        # return center
