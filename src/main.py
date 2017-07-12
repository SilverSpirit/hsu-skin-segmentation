import sys
import cv2
from hsu_segmenter import HsuSegmenter
from hsv_segmenter import HSVSegmenter
from hsu_with_correction import ImprovedHSUSegmenter

def main():
    f_name = sys.argv[1]
    input_img = cv2.imread(f_name)
    h = HsuSegmenter()
    
    output_img = h.get_mask_vec(input_img)
    cv2.imshow('Output without lighting', output_img)
    
    input_img_2 = h.lighting_compensation(input_img)
    output_img_1 = h.get_mask_vec(input_img_2)
    cv2.imshow('Lighting corrected output', output_img_1)
    # cv2.imshow('Lighting corrected output', input_img_2)

    # h1 = ImprovedHSUSegmenter()
    # output_img_1 = h1.get_mask_vec(input_img)
    # cv2.imshow('Output 1', output_img_1)
    cv2.waitKey(0)
    
    # take_cam_input()

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