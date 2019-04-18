#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:51:35 2019
@author: aayush
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import gridspec
import matplotlib.animation as animation
import scipy.linalg
from sklearn.metrics import r2_score
from PIL import Image  
from scipy import ndimage
import matplotlib.gridspec as gridspec
import pptk
#%%
'''
This function takes the keypoints and descriptors of multiple images and 
outputs the matched keypoints after Lowe's ratio test followed by 
homography transformation using RANSAC 
'''
def matchedImage(kp1,des1,kp2,des2,img1,img2):    
    flann=cv2.BFMatcher()    
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    # store all the good matches as per Lowe's ratio test.
    matchDx=[]
    matchDy=[]
    good = []
    matchesidx=[]
    #works at 0.8 and 5
    for j,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[j]=[1,0]
            matchDx.append(kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0])
            matchDy.append(kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1])
            good.append(m)
            matchesidx.append(j)
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    #this steps eliminates the random matches in an image
    #all the movements has to be in the same direction and similar scale
    #error matches after Lowe's ratio is reduced by finding a transformation which satifies all 
    #feature point movement 
    matched_src=[]
    matched_dst=[]
    if len(src_pts)>0:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)
      #  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        if mask is not None:
            for i in range (len(mask)):
                if mask[i]==0:
                    matchesMask[matchesidx[i]]=[0,0]
                else:
#                    matched_src.append(([src_pts[i][0][1],src_pts[i][0][0]]))
#                    matched_dst.append(([dst_pts[i][0][1],dst_pts[i][0][0]]))
                    matched_src.append(([src_pts[i][0][0],src_pts[i][0][1]]))
                    matched_dst.append(([dst_pts[i][0][0],dst_pts[i][0][1]]))
                    
            draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
            img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return np.array(matched_dst).reshape(-1,2),np.array(matched_src).reshape(-1,2),img4

def drawFlow(img, src_pts, dst_pts,j):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i, (new, old) in enumerate(zip(dst_pts, src_pts)):
        a,b = new.ravel()
        c,d = old.ravel()
        tip = tuple(np.float32(np.array([a, b]) ))
        cv2.line(img, (a,b),(c,d), (0,255,0), 1)
        cv2.line(img, (a,b), tip, (0,0,255), 4)
    cv2.imshow('Flow Field', img)
    cv2.imwrite('tracking_2/'+str(j)+'.png',img)
    cv2.waitKey(1)

##USER DEFINED INSTRINSIC PARAMETER
K=np.array([[7.070493000000e+02,0.000000000000e+00,6.040814000000e+02],[0.000000000000e+00, 7.070493000000e+02,1.805066000000e+02],[0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])
principal_focal=(K[0,2],K[1,2])
focal_length=K[0][0]

detectorname='SURF'
if detectorname=='SURF':
    detector = cv2.xfeatures2d.SURF_create(extended=False, hessianThreshold=50, upright=True, nOctaves=4, nOctaveLayers=4)

filepath='./Kitti_Seq_07/'
#filepath='./p1/'
img1=np.array(Image.open('%s%06d.png'% (filepath,0)))
#img1=np.array(Image.open('%s%06d.png'% (filepath,600)))
#transposing image for better visualization
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).T
mask1=np.ones((np.shape(img1)))
final_position=[]
final_rotation=[]
points_3d_x=[]
points_3d_y=[]
points_3d_z=[]

des_list=[]
#%%
#mask[200:400,300:]=1
(kp_initial, desc_initial)=detector.detectAndCompute(img1,mask=np.uint8(mask1))
R_next=np.eye(3)
T_next=np.zeros((3,1))
for i in range (0,2):
    print (i)
    img2=np.array(Image.open('%s%06d.png'% (filepath,i)))
    img2_orig=img2
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).T
    (kp_final, desc_final)=detector.detectAndCompute(img2,mask=np.uint8(mask1))

    matched_src_T,matched_dst_T,img4=matchedImage(kp_initial,desc_initial,kp_final,desc_final,img1,img2)
#    matched_src,matched_dst,img4=matchedImage(kp_initial,desc_initial,kp_final,desc_final,img1,img2)
#
    ##to visualize the maps
    cv2.imwrite('tracking_1/'+str(i)+'.png',img4.swapaxes(0,1))
    cv2.imshow('IMG',img4.swapaxes(0,1))
    cv2.imshow('IMG_1',img1.T)
    #transposing back image points which was transposed for the image
    matched_dst=np.zeros((matched_dst_T.shape))
    matched_src=np.zeros((matched_src_T.shape))
    matched_dst[:,0]=matched_dst_T[:,1]
    matched_dst[:,1]=matched_dst_T[:,0]
    matched_src[:,0]=matched_src_T[:,1]
    matched_src[:,1]=matched_src_T[:,0]
    matched_dst,matched_src=np.float32(matched_dst),np.float32(matched_src)
    
    #computation of the fundamental and the essential matrix and recovering pose
    # use of in built 8 point algorithm and RANSAC
    F, mask = cv2.findFundamentalMat(matched_src,matched_dst,method = cv2.FM_8POINT + cv2.FM_RANSAC)
    E=np.dot(K.T,np.dot(F,K))
    points, R_new, T_new, mask = cv2.recoverPose(E, matched_src, matched_dst, focal=focal_length, pp = principal_focal)

#    Essential matrix directly can be used directly    
#    E, mask = cv2.findEssentialMat(matched_src, matched_dst, focal=focal_length, pp = principal_focal, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#    points, R, t, mask = cv2.recoverPose(E, matched_src, matched_dst, focal=focal_length, pp = principal_focal)

#    Fundamental and the Essential matrix and Recovering pose self made functions need to set parameters accordingly 
#    did not produce effective results
#    F=self_fundamental(matched_src,matched_dst,1000,1000,0)      
#    E=np.dot(K.T,np.dot(F,K))
#    R,t=self_recover_pose(E) ###it outputs same result as recover pose

    T_next=T_next+(np.dot(R_next,T_new))
    R_next=(np.dot(R_new,R_next))
    final_position.append(T_next)
    final_rotation.append(R_next)
    drawFlow(img2_orig,matched_src,matched_dst,i)    
    plt.plot(np.array(final_position)[:,0],np.array(final_position)[:,2])
    plt.savefig('1.png')
    plt.savefig('trajectory_path/'+str(i)+'.png',dpi=250)
    img_show=np.array(Image.open('1.png'))
    cv2.imshow('trajectory',img_show)
    
    P0 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
    P0 = K.dot(P0)
    P1 = np.hstack((R_new, T_new))
    P1 = K.dot(P1)
    point1=matched_src.reshape(2, -1)
    point2=matched_dst.reshape(2, -1)
    points3d=cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)
#    points3d=(points3d/points3d[:,3].reshape(-1,1))
#    points3d=self_triangulate_point(P0,P1,point1,point2)  ##can use this function as well
    points_3d_x.append(np.array(points3d[:,0])+np.array(final_position)[i,0])
    points_3d_y.append(np.array(points3d[:,1]))
    points_3d_z.append(np.array(points3d[:,2])+np.array(final_position)[i,2])
#    P=np.array([np.hstack(points_3d_x),np.hstack(points_3d_z),np.hstack(points_3d_y)]).T
#    v = pptk.viewer(P) 
#    v.set(point_size=0.001)
#    if i>0:
#        v.capture('3d_pointcloud/'+str(i-1)+'.png')
#        v.record('recording', poses)
    if i%10==0:  #Key frame is kept as every 10 th frame 
        des_list.append(desc_final)
    img1=img2
    (kp_initial, desc_initial)=(kp_final, desc_final)
    key = cv2.waitKey(20) & 0xFF
    if key ==  27:
        break       
cv2.destroyAllWindows()

#%%
#to visualize the 3D model using pptk 
import pptk
P=np.array([np.hstack(points_3d_x),np.hstack(points_3d_z),np.hstack(points_3d_y)]).T
v = pptk.viewer(P) 
#attr6 = pptk.rand(1, 3)
#v.attributes(attr6)
v.set(point_size=0.001)
#v.capture('screenshot.png')

#%%
for j in range (600,1101):
    print (j)
    P=np.array([np.hstack(points_3d_x[:j]),np.hstack(points_3d_z[:j]),np.hstack(points_3d_y[:j])]).T
    v = pptk.viewer(P)
    v.set(r=np.float32(500))
    time.sleep(1)
#    v.record('recording')
    v.capture('3d_pointcloud/'+str(j-1)+'.png')
    time.sleep(1)
    v.close()
    
#%%
for j in range (156,600):
    print (j)
    P=np.array([np.hstack(points_3d_x[:j]),np.hstack(points_3d_z[:j]),np.hstack(points_3d_y[:j])]).T
    v = pptk.viewer(P)
#    v.set(r=np.float32(100))
#    v.set(lookat=np.float32([-40,0,200]))
    time.sleep(1)
#    v.record('recording')
    v.capture('3d_pointcloud/'+str(j-1)+'.png')
    time.sleep(1)
    v.close()
#%%
#to visualize the trajectory
plt.figure(2)
plt.plot(np.array(final_position)[:,0],np.array(final_position)[:,2])
plt.savefig('1.png')
img_show=np.array(Image.open('1.png'))
cv2.imshow('trajectory',img_show)
#%%

#CREATE A VISUAL BAG OF WORDS ON THE BASIS OF DESCRIPTORS with a dictionary size of 1000
#CREATE A CONFUSION MATRIX BASED ON COSINE SIMILAIRTY OF THE HISTOGRAMS 
#LOOP CLOSURE IS IDENTIFIED BASED ON THE POINTS WHERE COSINE SIMILARITY IS GREATER THAN 0.9
import scipy.cluster.vq as vq
dataarrayBOW=(np.vstack(des_list))
dictionarySize=1000
BOW=cv2.BOWKMeansTrainer(dictionarySize)
BOW.add(dataarrayBOW)
dictionary = BOW.cluster()

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = np.histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

histogramofwords=np.zeros((1101,dictionarySize))
for i in range (0,1101):
    print (i)
    img2=np.array(Image.open('%s%06d.png'% (filepath,i)))
    img2_orig=img2
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).T
    (kp_final, desc_final)=detector.detectAndCompute(img2,mask=np.uint8(mask1))
    histogramofwords[i,:]=computeHistograms(dictionary, desc_final)
#%%
plt.figure(figsize=(20,20))    
from sklearn.metrics.pairwise import cosine_similarity
confusion_matrix=cosine_similarity(histogramofwords)
#%%
plt.figure(3,figsize=(20,20))
plt.imshow(confusion_matrix)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png',dpi=250)
plt.figure(4)
plt.imshow((confusion_matrix>0.9),cmap='gray')
plt.title('Cosine similarity over 0.9')
plt.savefig('confusion_matrix_2.png',dpi=250)
plt.figure(5)
plt.imshow((confusion_matrix>0.9),cmap='gray')
plt.ylim([50,0])
plt.xlim([1000,1100])
plt.title('Loop closure')
plt.savefig('confusion_matrix_3.png',dpi=250)

#%%

for j in range (100,1101):
    print (j)
    img1=np.array(Image.open('3d_pointcloud/'+str(j)+'.png'))
    img2=np.array(Image.open('trajectory_path/'+str(j)+'.png'))
    img3=np.array(Image.open('tracking_1/'+str(j)+'.png'))
    img4=np.array(Image.open('tracking_2/'+str(j)+'.png'))
    plt.figure(3,figsize=(8,5))
    plt.subplot(221)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(img3)
    plt.subplot(222)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(img4)
    plt.subplot(223)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(img2)
    plt.subplot(224)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(img1)
    plt.tight_layout()
    plt.savefig('diff2/'+str(j)+'.png')
    plt.close('all')
    
    #%%
    
def self_triangulate_point(P1,P2,x1,x2):
    #x1=np.hstack((x1, np.ones(len(x1)).reshape(-1,1))).T
    #x2=np.hstack((x2, np.ones(len(x2)).reshape(-1,1))).T
    points_triangulated=[]
    for j in range(len(x1[0])):
        firstpoint=x1[:,j]
        secondpoint=x1[:,j]
        M = np.zeros((6,6)) 
        M[:3,:4] = P1 
        M[3:,:4] = P2 
        M[:2,4] = -firstpoint
        M[2,4] = -1 
        M[3:5,5] = -secondpoint
        M[5,5] = -1
        U,S,V = np.linalg.svd(M)
        X = V[-1,:4]
        points_triangulated.append(X)#/X[3])
    return np.array(points_triangulated)

def self_recover_pose(E):
    U, S, Vt = np.linalg.svd(E)
    #9.13 Hartley and Zisserman
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,1.0]).reshape(3, 3)
    R = U.dot(W.T).dot(Vt)
    T = U[:, 2]
    return R,T.reshape(-1,1)

def compute_fundamental(x1,x2):
    A = np.zeros((8,9))
    x1=np.hstack((x1, np.ones(len(x1)).reshape(-1,1))).T
    x2=np.hstack((x2, np.ones(len(x2)).reshape(-1,1))).T
    for i in range(8):
        #[x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1]
        A[i] = [x1[0,i]*x2[0,i],x1[0,i]*x2[1,i],x1[0,i]*x2[2,i], x1[1,i]*x2[0,i],x1[1,i]*x2[1,i],x1[1,i]*x2[2,i], x1[2,i]*x2[0,i],x1[2,i]*x2[1,i],x1[2,i]*x2[2,i] ]
    #General least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    #making F Rank 2
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    return F

def self_fundamental(points1,points2,iterations,thr,inlierRatio):
#    points1=np.fliplr(points1.T)
#    points2=np.fliplr(points2.T)
    
    F_new=np.zeros((3,3))
    for i in range (iterations):
        index=np.arange(len(points1))
        np.random.shuffle(index)
        a=index[:8]
        corr1=points1[a]
        corr2=points2[a]
        F=compute_fundamental(corr1,corr2)
      #  F=F/F[2,2]
        points1_homogeneous = np.hstack((points1,np.ones((len(points1),1)))).T
        points2_homogeneous = np.hstack((points2,np.ones((len(points2),1)))).T
        
        x_obtained = np.dot(points2_homogeneous.T,np.dot(F,points1_homogeneous))
        error = np.sum(np.square(x_obtained),axis=1)  
#        print (error)
        inlier  = error< thr
        inliers = np.sum(inlier)

        print (inliers/float(len(points1)))
      #  F_new=F  
        if inliers/float(len(points1))>=inlierRatio:
            inlierRatio=inliers/float(len(points1))
            F_new=F        
    return F_new/F_new[2,2]
