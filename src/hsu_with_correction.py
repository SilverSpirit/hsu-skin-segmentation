import cv2
import numpy as np
from consts import *

class ImprovedHSUSegmenter():
    def lighting_correction(self, img):
        non_skin_upper_bound = np.array([80, 80, 80], np.uint8)
        skin_upper_bound = np.array([255, 255, 255], np.uint8)

        mask = cv2.inRange(img, non_skin_upper_bound, skin_upper_bound, cv2.THRESH_TOZERO)
        non_skin_thresh = cv2.bitwise_and(img, img, mask = mask)

        # cv2.imshow("images", np.hstack([img, non_skin_thresh]))
        cv2.imshow('Skin Region 1', non_skin_thresh)
        # cv2.waitKey(0)

        im2 = np.copy(non_skin_thresh)
        b, g, r = cv2.split(im2)
        # print(np.min(b), np.min(g), np.min(r))
        # print(np.max(b), np.max(g), np.max(r))
        # print(np.shape(b), np.shape(r))

        cond1 = np.logical_and(np.logical_and(np.where(r<230, True, False),np.where(b<230, True, False)), np.where(g<230, True, False))
        cond2 = np.logical_not(np.logical_and(np.where(b<g, True, False), np.where(g<r, True, False)))
        im2[np.logical_and(cond1, cond2)] = 0

        cv2.imshow('Skin Region 2', im2)
        cv2.waitKey(0)
        return im2

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

        # print(y)
        # print(np.min(y), np.max(y), '\n')
        # print(np.min(cr), np.max(cr), '\n')
        # print(np.min(cb), np.max(cb))
        return cv2.merge((y, cr, cb))
        # return y, cr, cb

    def get_mask_vec(self, img):
        ycrcb_img = self.preprocess(img)
        rows, cols, channs = np.shape(ycrcb_img)
        # print(rows, cols, channs)
        y, cr, cb = cv2.split(ycrcb_img)
        cb_prime = np.copy(cb)
        cr_prime = np.copy(cr)

        kl, kh = 125, 170
        low_ind_b = np.where(y < kl, True, False)
        high_ind_b = np.where(y > kh, True, False)
        change_ind_b = np.logical_or(low_ind_b, high_ind_b)

        center_low_ind_b = np.where(y < kl, True, False)
        center_high_ind_b = np.where(y >= kl, True, False)
        center_chroma_b = 108 * np.ones(np.shape(y))
        center_chroma_b[center_low_ind_b] = center_chroma_b[center_low_ind_b] + (((kl - y[center_low_ind_b]) * (118 - 108)) / (kl - ymin))
        # center_chroma_b[high_ind_b] = center_chroma_b[high_ind_b] + (((y[high_ind_b] - kh) * (118 - 108)) / (ymax - kh))
        center_chroma_b[center_high_ind_b] = -0.0007 * np.square(y[center_high_ind_b]-170) + 120

        spread_of_cluster_b = np.zeros(np.shape(y))
        spread_of_cluster_b[low_ind_b] = wlcb + ((y[low_ind_b] - ymin) * (wcb - wlcb) / (kl - ymin))
        # spread_of_cluster_b[high_ind_b] = whcb + ((ymax - y[high_ind_b]) * (wcb - whcb) / (ymax - kh))
        spread_of_cluster_b[high_ind_b] = 0.0156 * np.square(y[high_ind_b]-200) + 115

        cb_prime[low_ind_b] = (cb_prime[low_ind_b] - center_chroma_b[low_ind_b]) * (
        46.97 / spread_of_cluster_b[low_ind_b]) + center_chroma_b[low_ind_b]
        cb_prime[high_ind_b] = (cb_prime[high_ind_b] - center_chroma_b[high_ind_b]) * (
            33.98 / spread_of_cluster_b[high_ind_b]) + center_chroma_b[high_ind_b]

        kl, kh = 125, 200
        low_ind_r = np.where(y < kl, True, False)
        high_ind_r = np.where(y > kh, True, False)
        change_ind_r = np.logical_or(low_ind_r, high_ind_r)


        center_low_ind_r = np.where(y<kl, True, False)
        center_high_ind_r = np.where(y>=kl, True, False)
        center_chroma_r = 154 * np.ones(np.shape(y))
        center_chroma_r[center_low_ind_r] = center_chroma_r[center_low_ind_r] - (((kl - y[center_low_ind_r]) * (154 - 144)) / (kl - ymin))
        # center_chroma_r[high_ind_r] = center_chroma_r[high_ind_r] + (((y[high_ind_r] - kh) * (154 - 132)) / (ymax - kh))
        center_chroma_r[center_high_ind_r] = -0.0017 * np.power((y[center_high_ind_r] - 150), 2) + 150

        spread_of_cluster_r = np.zeros(np.shape(y))
        spread_of_cluster_r[low_ind_r] = wlcr + ((y[low_ind_r] - ymin) * (wcr - wlcr) / (kl - ymin))
        # spread_of_cluster_r[high_ind_r] = whcr + ((ymax - y[high_ind_r]) * (wcr - whcr) / (ymax - kh))
        spread_of_cluster_r[high_ind_r] = (0.6 * y[high_ind_r]) + 285

        cr_prime[low_ind_r] = (cr_prime[low_ind_r] - center_chroma_r[low_ind_r]) * (
        38.76 / spread_of_cluster_r[low_ind_r]) + center_chroma_r[low_ind_r]
        cr_prime[high_ind_r] = (cr_prime[high_ind_r] - center_chroma_r[high_ind_r]) * (
            19.75 / spread_of_cluster_r[high_ind_r]) + center_chroma_r[high_ind_r]

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

    def normalize_range(self, col_channel, new_min, new_max,
                        old_min = 0, old_max = 255):
        return (((col_channel - old_min) / (old_max - old_min))
                * (new_max - new_min) + new_min)

