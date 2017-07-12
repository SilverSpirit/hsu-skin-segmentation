import numpy as np
import cv2
from consts import *

class HsuSegmenter:
    def __init__(self):
        pass # nothing for now
    
    def get_mask(self, img):
        ycrcb_img = self.preprocess(img)
        rows, cols, channs = np.shape(ycrcb_img)
        #print(rows, cols, channs)
        y, cr, cb = cv2.split(ycrcb_img)
        cb_prime = np.copy(cb)
        cr_prime = np.copy(cr)
        for i in range(0, rows):
            for j in range(0, cols):
                y_val = y[i, j]
                if y_val < kl or y_val > kh:
                    cb_prime[i, j] = self.get_trans_chroma(cb[i,j], y_val, 'cb')
                    cr_prime [i, j]= self.get_trans_chroma(cr[i,j], y_val, 'cr')
        
        
        x = theta_cos * (cb_prime - cx) + theta_sin * (cr_prime - cy)
        y = -theta_sin * (cb_prime - cx) + theta_cos * (cr_prime - cy)
        
        # print(cb_prime) 
        # print(cr_prime) 
        # print(y)
        # print(x) 
        eval_mat = ((((x - ecx) / a) ** 2 + ((y - ecy) / b) ** 2))
        return np.where(eval_mat <= 1, 255, 0).astype(np.uint8)
        
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

        #print(y)
        #print(np.min(y), np.max(y), '\n')
        #print(np.min(cr), np.max(cr), '\n')
        #print(np.min(cb), np.max(cb))
        return cv2.merge((y, cr, cb))
        # return y, cr, cb
        
    def lighting_compensation(self, img):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb_img)
        rows, cols = np.shape(y)
        area = rows * cols 
        skin_count = np.sum(self.get_mask_vec(img).astype(bool))
        bright_ind = np.where(y >= 0.95 * 255, True, False)
        top_count = np.sum(bright_ind)
        
        if skin_count > 0.5 * area or top_count < 100:
            print('not correcting')
            return img
        
        b, g, r = cv2.split(img)
        mean_b = int(np.mean(b[bright_ind]))
        mean_g = int(np.mean(g[bright_ind]))
        mean_r = int(np.mean(r[bright_ind]))

        b = np.array(self.normalize_range(b, 0, 255, 0, mean_b)).astype(np.uint8)
        g = np.array(self.normalize_range(g, 0, 255, 0, mean_g)).astype(np.uint8)
        r = np.array(self.normalize_range(r, 0, 255, 0, mean_r)).astype(np.uint8)
        return cv2.merge((b,g,r))
        

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
        
    def get_trans_chroma(self, chroma, y, chann):
        if chann == 'cb':
            center_chroma_b = 108
            if y < kl:
                center_chroma_b += ((kl - y) * (118 - 108)) / (kl - ymin)
            elif y > kh:
                center_chroma_b += ((y - kh) * (118 - 108)) / (ymax - kh)

            spread_of_cluster_b = 0
            if y < kl:
                spread_of_cluster_b += wlcb + ((y - ymin) * (wcb - wlcb) / (kl - ymin))
            elif y > kh:
                spread_of_cluster_b += whcb + ((ymax - y) * (wcb - whcb) / (ymax - kh))

            return (chroma - center_chroma_b) * (wcb / spread_of_cluster_b) + 108

        elif chann == 'cr':
            center_chroma_r = 154
            if y < kl:
                center_chroma_r -= ((kl - y) * (154 - 144)) / (kl - ymin)
            elif y > kh:
                center_chroma_r += ((y - kh) * (154 - 132)) / (ymax - kh)

            spread_of_cluster_r = 0
            if y < kl:
                spread_of_cluster_r += wlcr + ((y - ymin) * (wcr - wlcr) / (kl - ymin))
            elif y > kh:
                spread_of_cluster_r += whcr + ((ymax-y) * (wcr - whcr) / (ymax - kh))
            return (chroma - center_chroma_r) * (wcr / spread_of_cluster_r) + 154
    
    def get_mask_vec(self, img):
        ycrcb_img = self.preprocess(img)
        rows, cols, channs = np.shape(ycrcb_img)
        #print(rows, cols, channs)
        y, cr, cb = cv2.split(ycrcb_img)
        cb_prime = np.copy(cb)
        cr_prime = np.copy(cr)
        
        low_ind = np.where(y < kl, True, False)
        high_ind = np.where(y > kh, True, False)
        change_ind = np.logical_or(low_ind, high_ind)
        
        center_chroma_b = 108 * np.ones(np.shape(y))
        center_chroma_b[low_ind] = center_chroma_b[low_ind] + (((kl - y[low_ind]) * (118 - 108)) / (kl - ymin))
        center_chroma_b[high_ind] = center_chroma_b[high_ind] + (((y[high_ind] - kh) * (118 - 108)) / (ymax - kh))

        spread_of_cluster_b = np.zeros(np.shape(y))
        spread_of_cluster_b[low_ind] = wlcb + ((y[low_ind] - ymin) * (wcb - wlcb) / (kl - ymin))
        spread_of_cluster_b[high_ind] = whcb + ((ymax - y[high_ind]) * (wcb - whcb) / (ymax - kh))

        cb_prime[change_ind] = (cb_prime[change_ind] - center_chroma_b[change_ind]) * (wcb / spread_of_cluster_b[change_ind]) + 108

        center_chroma_r = 154 * np.ones(np.shape(y))
        center_chroma_r[low_ind] = center_chroma_r[low_ind] - (((kl - y[low_ind]) * (154 - 144)) / (kl - ymin))
        center_chroma_r[high_ind] = center_chroma_r[high_ind] + (((y[high_ind] - kh) * (154 - 132)) / (ymax - kh))

        spread_of_cluster_r = np.zeros(np.shape(y))
        spread_of_cluster_r[low_ind] = wlcr + ((y[low_ind] - ymin) * (wcr - wlcr) / (kl - ymin))
        spread_of_cluster_r[high_ind] = whcr + ((ymax - y[high_ind]) * (wcr - whcr) / (ymax - kh))

        cr_prime[change_ind] = (cr_prime[change_ind] - center_chroma_r[change_ind]) * (wcr / spread_of_cluster_r[change_ind]) + 154
        
        
        x = theta_cos * (cb_prime - cx) + theta_sin * (cr_prime - cy)
        y = -theta_sin * (cb_prime - cx) + theta_cos * (cr_prime - cy)
        
        # print(cb_prime) 
        # print(cr_prime) 
        # print(y)
        # print(x) 
        eval_mat = ((((x - ecx) / a) ** 2 + ((y - ecy) / b) ** 2))
        return np.where(eval_mat <= 1, 255, 0).astype(np.uint8)

    def conv_rgb_ycbcr(self, image):
        b, g, r = cv2.split(image)
        y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).astype(int)
        cb = (128 - 0.148 * r - 0.291 * g + 0.439 * b).astype(int)
        cr = (128 + 0.439 * r - 0.368 * g - 0.071 * b).astype(int)

