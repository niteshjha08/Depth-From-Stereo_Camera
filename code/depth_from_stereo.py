#!/usr/bin/python3
from dis import dis
import cv2
import numpy as np
import matplotlib.pyplot as plt

from misc_utils import *
from projection_utils import *
from feature_matching_utils import *
from epipolar_utils import *
# from alt_codes import *

def main(folder_name):
    # Load images and camera matrix
    images = get_images(images_dir)
    K1, K2, baseline = load_camera_matrix(folder_name)

    img0_color = images[0]
    img1_color = images[1]

    img0= cv2.cvtColor(img0_color, cv2.COLOR_BGR2GRAY)
    img1= cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img0, None)
    kp2, des2 = sift.detectAndCompute(img1, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    good_matches = matches[0:100]


    good_matches = get_feature_matches(kp1, kp2, good_matches)
    match_img = draw_feature_matching(img0_color, img1_color, good_matches)
    cv2.namedWindow("match_img",cv2.WINDOW_KEEPRATIO)
    cv2.imshow("match_img",match_img)
    cv2.waitKey(0)


    F_best, best_matches = RANSAC(good_matches, 0.02, 1000)
    match_img_inliers = draw_feature_matching(img0_color, img1_color, best_matches)
    cv2.namedWindow("match_img_inliers",cv2.WINDOW_KEEPRATIO)
    cv2.imshow("match_img_inliers",match_img_inliers)
    print("inlier shape: ", np.shape(best_matches))
    cv2.waitKey(0)

    E = get_essential_matrix(F_best, K1, K2)

    R,C = get_camera_pose(E)

    pts_3D_all = get_3D_pts(best_matches, R, C, K1, K2)
    R, C = get_best_pose(pts_3D_all, R, C)

    pts1 = best_matches[:,:2]
    pts2 = best_matches[:,2:]


    h1, w1 = img0_color.shape[:2]
    h2, w2 = img1_color.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F_best, imgSize=(w1, h1))

    print("H1 and H2 are:\n", H1, H2)

    img0_rect = cv2.warpPerspective(img0_color, H1, (w1, h1))
    img1_rect = cv2.warpPerspective(img1_color, H2, (w2, h2))
    pts1_rect = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H1).reshape(-1,2)
    pts2_rect = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H2).reshape(-1,2)
    F_rectified = rectify_F(F_best, H1, H2)

    img0_rect_copy = img0_rect.copy()
    img1_rect_copy = img1_rect.copy()

    eplines_1 = cv2.computeCorrespondEpilines(pts1_rect.reshape(-1,1,2),1,F_rectified)
    ep_lines_img1_rect = drawlines(img0_rect_copy, pts1_rect, eplines_1)
    cv2.namedWindow('eplines1_rect', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('eplines2_rect', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('merged', cv2.WINDOW_KEEPRATIO)

    dataset_name = folder_name.split('/')[-1]

    eplines_2 = cv2.computeCorrespondEpilines(pts2_rect.reshape(-1,1,2),1,F_rectified)
    ep_lines_img2_rect = drawlines(img1_rect_copy, pts2_rect, eplines_2)
    cv2.imshow("eplines2_rect",ep_lines_img2_rect)
    cv2.imwrite('../Results/eplines_2' +dataset_name+ ".png", ep_lines_img2_rect)
    cv2.imshow("eplines1_rect",ep_lines_img1_rect)
    cv2.imwrite('../Results/eplines_1' +dataset_name+ ".png", ep_lines_img1_rect)


    merged = np.hstack((ep_lines_img1_rect, ep_lines_img2_rect))
    cv2.imshow('merged',merged)
    cv2.imwrite('../Results/eplines_both' +dataset_name+ ".png", merged)

    cv2.waitKey(0)

    disparity_map = get_disparity_map(img0_rect,img1_rect)

    
    plt.imshow(disparity_map, cmap='hot', interpolation='nearest')
    plt.savefig('../Results/disparity_image_heat' +dataset_name+ ".png")
    plt.imshow(disparity_map, cmap='gray', interpolation='nearest')
    plt.savefig('../Results/disparity_image_gray' +dataset_name+ ".png")

    f = K1[0,0]
    depth_map = get_depth_map(disparity_map, baseline, f)

    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('../Results/depth_image' +dataset_name+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('../Results/depth_image_gray' +dataset_name+ ".png")

if __name__ == "__main__":
    images_dir = r'/home/nitesh/programming/ENPM673/Depth-Using-Stereo-main/Data/Project 3/pendulum'
    main(images_dir)

