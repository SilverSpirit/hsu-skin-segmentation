import sys
import cv2
from hsu_segmenter import HsuSegmenter

def main():
    f_name = sys.argv[1]
    input_img = cv2.imread(f_name)
    h = HsuSegmenter()
    output_img = h.get_mask(input_img)
    output_img_vec = h.get_mask_vec(input_img)
    cv2.imshow('Output', output_img)
    cv2.imshow('Output vec', output_img_vec)
    cv2.waitKey(0)
    # h.conv_rgb_ycbcr(input_img)


if __name__ == '__main__':
    main()