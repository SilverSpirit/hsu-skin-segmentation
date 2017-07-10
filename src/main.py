import sys
import cv2
from hsu_segmenter import HsuSegmenter
from hsv_segmenter import HSVSegmenter

def main():
    # f_name = sys.argv[1]
    # input_img = cv2.imread(f_name)
    # h = HsuSegmenter()
    # h1 = HSVSegmenter()
    # output_img = h.get_mask_vec(input_img)
    # output_img_hsv = h1.get_mask(input_img)
    # cv2.imshow('Output', output_img)
    # cv2.imshow('Output hsv', output_img_hsv)
    # cv2.waitKey(0)
    # h.conv_rgb_ycbcr(input_img)

    take_cam_input()

def hsv_seg(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lower = (0, 0, 0)
    hsv_upper = (255, 255, 255)
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
    return mask

def take_cam_input():
    cap = cv2.VideoCapture(0)
    #h = HsuSegmenter()
    h = HSVSegmenter()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        #output = h.get_mask_vec(frame)
        output = h.get_mask(frame)
        cv2.imshow('Cam Output', output)
        if cv2.waitKey(10) == ord('q'):
            break


if __name__ == '__main__':
    main()