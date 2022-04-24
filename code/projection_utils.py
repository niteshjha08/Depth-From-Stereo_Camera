import cv2
import numpy as np
import os
import random
def get_3D_pts(best_features, R, C, K1, K2):
    pts3D_all = []
    I = np.identity(3)
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1))))
    for i in range(4):
        P2 = np.dot(K2, np.dot(R[i], np.hstack((I, -C[i].reshape(3,1)))))

        pt1 = best_features[:,:2]
        pt2 = best_features[:,2:]
        pt = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
        pts3D_all.append(pt)

    return pts3D_all
