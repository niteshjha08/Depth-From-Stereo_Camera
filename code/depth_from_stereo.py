import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
import os
import random

def get_images(images_dir):
    files = []
    file_names = os.listdir(images_dir)
    img_names = []
    for name in file_names:
        if(name.endswith(tuple(['.png','.jpg','.jpeg']))):
            img_names.append(name)
    images = []
    for name in img_names:
        img = cv2.imread(os.path.join(images_dir, name))
        images.append(img)

    return images

def get_FundamentalMatrix(features):
    if(features.shape[0]<8):
        print("Less than 8 features found!")
        return None
    else:
        pts1 = features[:,:2]
        pts2 = features[:,2:]

        A = np.zeros((features.shape[0],9))
        for i in range(features.shape[0]):
            A[i] = [pts1[i,0]* pts2[i,0], pts1[i,0]*pts2[i,1], pts1[i,0], pts1[i,1]*pts2[i,0], pts1[i,1]*pts2[i,1], pts1[i,1],\
                    pts2[i,0], pts2[i,1], 1]
        
        # Finding rank 3 F using SVD
        U, D, Vt = np.linalg.svd(A)
        V = Vt.T
        F = V[:,-1].reshape(3,3)

        # Fixing rank of F to 2
        U, D, Vt = np.linalg.svd(F)
        D[2,2] = 0
        UD = np.dot(U,D)
        F = np.dot(UD, Vt)
        return F

def getError(F, feature):
    x1 = np.array([feature[0],feature[1],1])
    x2 = np.array([feature[2],feature[3],1])
    
    error = np.dot(feature[0], feature[1],1)
    pass

def RANSAC(matches, epsilon, iters):
    best_F = []
    best_matches = []
    num_features = matches.shape[0]
    for i in range(iters):
        random_indices = random.randint(num_features, size = 8)
        random_features = matches[random_indices,:]

        F = get_FundamentalMatrix(random_features)
        for j in range(num_features):
            error = getError(F, features[j])


def main(images_dir):
    images = get_images(images_dir)
    # print("number of images: ",len(images))
    img1 = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(images[1],cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(desc1,desc2)
    matches = sorted(matches, key = lambda x :x.distance)
    # print(np.shape(matches))
    # print(matches[0][:2])
    # print(matches[0][2:4])

    chosen_matches = matches[0:100]

    good_matches = []
    for i, m1 in enumerate(chosen_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        good_matches.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    good_matches = np.array(good_matches).reshape(-1, 4)

    print(good_matches.shape[0])
    print(good_matches[0])
    print(good_matches[0][2:4])
    print(type(good_matches))


if __name__=="__main__":
    images_dir = r'./../data/pendulum/'
    main(images_dir)


