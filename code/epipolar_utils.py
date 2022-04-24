import cv2
import numpy as np
import os
import random

def drawlines(img, pts, lines):
    lines = lines.reshape(-1,3)
    for pt, line in zip(pts, lines):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [img.shape[1], -(line[2] + line[0]*img.shape[1])/line[1]])
        cv2.line(img, (x0, y0), (x1,y1),color, 1)
        cv2.drawMarker(img, np.int32(pt), color, 1, 10, 2,)
    return img

def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T


def get_FundamentalMatrix(features):
    if(features.shape[0]<8):
        return None
    else:
        x1 = features[:,:2]
        x2 = features[:,2:]
        x1, T1 = normalize(x1)
        x2, T2 = normalize(x2)


        A = np.zeros((features.shape[0],9))
        for i in range(features.shape[0]):
            x_1,y_1 = x1[i,0], x1[i,1]
            x_2,y_2 = x2[i,0], x2[i,1]

            A[i] = [x_2 * x_1, x_2 * y_1, x_2, y_2 * x_1, y_2 * y_1, y_2, x_1, y_1, 1]
        
        # Finding rank 3 F using SVD
        U, D, Vt = np.linalg.svd(A)
        V = Vt.T
        F = V[:,-1].reshape(3,3)

        # Fixing rank of F to 2
        U, D, Vt = np.linalg.svd(F)
        D = np.diag(D)
        D[2,2] = 0
        UD = np.dot(U,D)
        F = np.dot(UD, Vt)
        F = np.dot(T2.T, np.dot(F,T1))
        return F

def get_essential_matrix(F, K1, K2):
    E = np.dot(K1.T, np.dot(F,K2))
    U, D, Vt = np.linalg.svd(E)
    D = np.diag([1,1,0])
    E = np.dot(U, np.dot(D, Vt))
    return E

def get_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    C1 = U[:,2]
    R1 = np.dot(U, np.dot(W, Vt))
    if(np.linalg.det(R1)<0):
        R1 = -R1
        C1 = -C1

    C2 = -U[:,2]
    R2 = np.dot(U, np.dot(W, Vt))
    if(np.linalg.det(R2)<0):
        R2 = -R2
        C2 = -C2

    C3 = U[:,2]
    R3 = np.dot(U, np.dot(W.T, Vt))
    if(np.linalg.det(R3)<0):
        R3 = -R3
        C3 = -C3

    C4 = -U[:,2]
    R4 = np.dot(U, np.dot(W.T, Vt))
    if(np.linalg.det(R4)<0):
        R4 = -R4
        C4 = -C4

    C = [C1, C2, C3, C4]
    R = [R1, R2, R3, R4]

    return R, C


def get_best_pose(pts_3D_all, R, C):
    max_positive_z = 0
    idx = None
    for i in range(len(R)):
        R1 = R[i]
        C1 = C[i]
        r3 = R1[2,:]
        pts_3D = pts_3D_all[i]
        pts_3D = (pts_3D/pts_3D[3,:])[:3,:]
        
        positive_z = get_positive_z_count(pts_3D,C1,r3)
        
        if positive_z > max_positive_z:
            max_positive_z = positive_z
            idx = i
    return R[idx], C[idx]

def get_positive_z_count(pts_3D,C,r3):
    positive_z = 0
    for i in range(pts_3D.shape[1]):
        X = pts_3D[:,i]
        val = np.dot(r3, (X - C))
        if val>0:
            positive_z+=1

    return positive_z

def rectify_F(F, H1, H2):
    H1_inv = np.linalg.inv(H1)
    H1_inv = H1_inv/H1_inv[2,2]
    H2_inv = np.linalg.inv(H2)
    H2_inv = H2_inv/H2_inv[2,2]
    F_rect = np.dot(H2_inv.T, np.dot(F, H1_inv))
    return F_rect

def draw_feature_matching(img1, img2, good_matches):
    combined_img = np.hstack((img1, img2))
    for match in good_matches:
        pt1 = match[:2]
        pt2 = match[2:]
        # print(pt1, pt2)
        cv2.line(combined_img, np.int0(pt1), (int(pt2[0]+img1.shape[1]), int(pt2[1])), (0,255,0),2)
    return combined_img

def get_ssd(img1, img2):
    ssd = np.sum(np.square(img1 - img2))
    return ssd
    
def match_window(img1_window, img2, x, y, search_range, window):
    x_start = max(0, x - search_range)
    x_end = min(x+ search_range, img2.shape[1] - window)

    min_ssd = None
    min_ssd_x = 0
    for x in range(x_start, x_end):
        img2_window = img2[y:y+window, x:x+window]
        ssd = get_ssd(img1_window, img2_window)

        if min_ssd is None:
            min_ssd = ssd
        
        if ssd<min_ssd:
            min_ssd = ssd
            min_ssd_x = x

    return min_ssd_x

def get_disparity_map(img1, img2, window = 10, search_range = 50):
    disparity_map = np.zeros(img1.shape[:2])
    h,w = img1.shape[:2]
    
    for y in range(window, h - window):
        for x in range(window, w - window):
            print(x,y)
            img1_window = img1[y:y+window, x:x+window]
            matching_x = match_window(img1_window, img2, x, y, search_range, window)
            disparity_map[y,x] = np.abs(matching_x - x)
    return disparity_map

def get_depth_map(disparity_map, baseline, focal_length):

    depth = (float(baseline * focal_length) / (disparity_map + 1e-15)).astype(np.float32)
    depth[depth > 1e5] = 1e5
    depth = np.uint8(depth * 255 / np.max(depth))

    return depth