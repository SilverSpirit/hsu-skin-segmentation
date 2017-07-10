import numpy as np
import cv2

class HSVSegmenter():
    def get_mask(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        h = self.normalize_range(h, 0, 1)
        s = self.normalize_range(s, 0, 1)
        mask = np.copy(h)
        # %convert the RGB color image to HSV image
        # im_hsv = rgb2hsv(im);
        # temp = im_hsv(:,:,1);

        # %classify the pixels that fall outside the threshold levels for skin color 
        # %in H and S color space as 0, classify rest as 1
        # temp(((0<temp) & (temp <= 0.1)) & im_hsv(:,:,2)<0.65 ) = 1;
        # temp(im_hsv(:,:,2)>0.72) = 0;
        # temp(im_hsv(:,:,2)<0.15) = 0;
        # temp(temp>=0.96) = 1;
        # temp(temp<0.96) = 0;
        # figure, imagesc(temp);
        # colormap(gray);
        ind_1 = np.logical_and(np.where(0 < mask, True, False), 
                np.where(mask <= 0.1, True, False))
        ind_2 = np.where(s < 0.65, True, False)
        ind = np.logical_and(ind_1, ind_2)
        mask[ind] = 1
        mask[np.where(s > 0.72, True, False)] = 0
        mask[np.where(s < 0.15, True, False)] = 0
        mask[np.where(mask >= 0.96, True, False)] = 1
        mask[np.where(mask < 0.96, True, False)] = 0
        mask = self.normalize_range(mask, 0, 255, 0, 1).astype(np.uint8)
        return mask
    
    def normalize_range(self, col_channel, new_min, new_max, 
                    old_min = 0, old_max = 255):
        return (((col_channel - old_min) / (old_max - old_min)) 
                * (new_max - new_min) + new_min)