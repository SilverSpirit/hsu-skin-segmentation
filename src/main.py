import sys
import cv2
from hsu_segmenter import HsuSegmenter

def main():
    # f_name = sys.argv[1]
    # input_img = cv2.imread(f_name)
    # h = HsuSegmenter()
    # output_img = h.get_mask(input_img)
    # output_img_vec = h.get_mask_vec(input_img)
    # cv2.imshow('Output', output_img)
    # cv2.imshow('Output vec', output_img_vec)
    # cv2.waitKey(0)
    # h.conv_rgb_ycbcr(input_img)

    take_cam_input()


def take_cam_input():
    cap = cv2.VideoCapture(0)
    h = HsuSegmenter()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        output = h.get_mask_vec(frame)
        cv2.imshow('Cam Output', output)
        if cv2.waitKey(10) == ord('q'):
            break


if __name__ == '__main__':
    main()