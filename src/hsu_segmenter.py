import numpy as np
import cv2
from consts import *

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
        # cv2.normalize(y, y, 16, 235, cv2.NORM_MINMAX)
        # cv2.normalize(cr, cr, 16, 240, cv2.NORM_MINMAX)
        # cv2.normalize(cb, cb, 16, 240, cv2.NORM_MINMAX)
        y = (self.normalize_range(y, 16, 235)).astype(int)
        cr = (self.normalize_range(cr, 16, 240)).astype(int)
        cb = (self.normalize_range(cb, 16, 240)).astype(int)

        print(y)
        print(np.min(y), np.max(y), '\n')
        print(np.min(cr), np.max(cr), '\n')
        print(np.min(cb), np.max(cb))
        return cv2.merge((y, cr, cb))

    def normalize_range(self, col_channel, new_min, new_max, 
                        old_min = 0, old_max = 255):
        return (((col_channel - old_min) / (old_max - old_min)) 
                * (new_max - new_min) + new_min)

    # def lighting_correction(self, img):
        # b, g, r = cv2.split(img)
        # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # top_y = 0.95 * np.amax(ycrcb_img[:, :, 0])
        # ind = np.where(ycrcb_img[:, :, 0] >= top_y, True, False)
        
        # if np.sum(ind) >= 100:
            # avg_b = np.sum(b[ind]) / len(b[ind])
            # avg_g = np.sum(g[ind]) / len(g[ind])
            # avg_r = np.sum(r[ind]) / len(r[ind])
            # self.normalize_range()
        
