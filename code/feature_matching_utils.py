import cv2
import numpy as np
import os
import random

from epipolar_utils import get_FundamentalMatrix

def get_feature_matches(kp1, kp2, chosen_matches):
    good_matches = []
    for i, match in enumerate(chosen_matches):
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        good_matches.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    good_matches = np.array(good_matches).reshape(-1, 4)

    return good_matches

def draw_feature_matching(img1, img2, good_matches):
    combined_img = np.hstack((img1, img2))
    for match in good_matches:
        pt1 = match[:2]
        pt2 = match[2:]
        cv2.line(combined_img, np.int0(pt1), (int(pt2[0]+img1.shape[1]), int(pt2[1])), (0,255,0),2)
    return combined_img

def RANSAC(matches, epsilon, iters):
    best_F = []
    best_matches = []
    best_matches_count = 0
    num_features = matches.shape[0]
    for i in range(iters):
        random_indices = np.random.choice(num_features, size = 8)
        random_features = matches[random_indices,:]
        
        F = get_FundamentalMatrix(random_features)
        valid_matches = []
        ct = 0
        for j in range(num_features):
            error = get_error(F,matches[j])

            if error < epsilon:
                ct+=1
                valid_matches.append(j)
        if len(valid_matches) > best_matches_count:
            best_matches = valid_matches
            best_matches_count = len(valid_matches)
            best_F = F

    best_F = best_F/best_F[2][2]
    best_features = matches[best_matches,:]

    return best_F, best_features

def get_error(F, feature):
    
    x1 = np.array([feature[0],feature[1],1]).T
    x2 = np.array([feature[2],feature[3],1])
    x1F = np.matmul(x1, F)
    error = np.abs(np.matmul(x1F,x2))
    return error
