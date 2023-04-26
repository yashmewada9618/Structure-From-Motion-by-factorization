import cv2
import glob
import os
import numpy as np
import open3d as o3d

"""
Author: Yash Mewada {mewada.y@northeastern.edu}
Created: April 21st 2023
"""
class SFM:
    def __init__(self,pathTofiles):
        """
        Initialise the class and parse the locations of all the images.
        """
        self.path = pathTofiles
        input = pathToFiles + str('/hotel/')
        os.chdir(input)
        self.images = []
        for file in list(glob.glob("*.png")):
            self.images.append(os.getcwd() + str('/') + file)
        self.images = np.sort(self.images)
        # print(self.images)
        self.F_list = []
        self.E_list = []
        self.inliers1 = []
        self.inliers2 = []
        self.of_params = dict(winSize = (15,15),
                         maxLevel = 2,
                         criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.0001))
        self.feature_params = dict( maxCorners = 170,
                       qualityLevel = 0.0001,
                       minDistance = 7,
                       blockSize = 7)
        # print(self.images)
    
    def estimateOpticalFlow(self):
        """
        Estimate the optical flow based on the Shitomashi and lucas kanade tracking features.
        """

        os.chdir(self.path + str('/sift_output/'))
        index = 0

        p0 = cv2.imread(self.images[0])
        prev_gray = cv2.cvtColor(p0,cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **self.feature_params)
        # prev_pts = prev_pts.reshape(-1,1,2)
        prev_pts = np.squeeze(prev_pts)
        print(prev_pts)
        # correspondences = [pts1]
        color = np.random.randint(0,255,(500,3))
        # prev_gray = None
        # prev_pts = None
        # print(color.shape)
        mask = np.zeros_like(p0)
        W = np.zeros((len(self.images)*2,prev_pts.shape[0]))
        for i in range(len(self.images)-1):
            print(self.images[i+1])
            img = cv2.imread(self.images[i+1])
            curr_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            curr_pts,st,err = cv2.calcOpticalFlowPyrLK(prev_gray,curr_gray,prev_pts,None,**self.of_params)
            st = st.reshape(curr_pts.shape[0], )

            # curr_pts1 = np.squeeze(curr_pts)
            # print(curr_pts)
            if curr_pts is not None:
                curr_pts = curr_pts[st==1]
                prev_pts = prev_pts[st==1]
            
            for k in range(curr_pts.shape[0]):
                W[2*i:2*i+2, k] = curr_pts[k]

            # correspondences.append(good_new)
            for i, (new, old) in enumerate(zip(curr_pts, prev_pts)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                img = cv2.circle(img, (int(a), int(b)), 5, color[i].tolist(), -1)
            img_new = cv2.add(img, mask)
            prev_gray = curr_gray
            prev_pts = curr_pts.copy()
            
            # cv2.imwrite(str(index) + ".png",img_new)
            index += 1
            cv2.imshow("gray",img_new)
            cv2.waitKey(0)
        # W = np.array(correspondences)
        W_mean = np.mean(W, axis=1, keepdims=True)
        W_reg = W - W_mean

        # Compute SVD of W
        U, D, VT = np.linalg.svd(W_reg,full_matrices=False)
        # D[3:] = 0  # set singular values 5 to 2m to 0
        # Keep the first 3 eigenvalues/,full_matrices=Falseeigenvectors
        U_3 = U[:, :3]
        D_3 = np.diag(np.sqrt(D[:3]))
        V_T_3 = VT[:3, :]

        # Compute M and S
        M = np.dot(U_3, D_3)
        S = np.dot(D_3, V_T_3)
        
        R_i = M[:100, :]
        R_j = M[100:200, :]
        G1, c1 = self.mat_G_maker(np.zeros((202, 6)), np.zeros((202, 1)), R_i, R_i, same=1)
        G2, c2 = self.mat_G_maker(np.zeros((202, 6)), np.zeros((202, 1)), R_j, R_j, same=1)
        G3, c3 = self.mat_G_maker(np.zeros((202, 6)), np.zeros((202, 1)), R_i, R_j, same=0)
        G = np.vstack((G1, G2, G3))
        c = np.vstack((c1, c2, c3))
        c = c.squeeze()
        GTG_inv = np.linalg.pinv(np.dot(G.T, G))
        l = np.dot(np.dot(GTG_inv, G.T), c)

        L = np.zeros((3, 3))

        L[0, 0] = l[0]
        L[1, 1] = l[3]
        L[2, 2] = l[5]

        L[0, 1] = l[1]
        L[1, 0] = L[0, 1]

        L[0, 2] = l[2]
        L[2, 0] = L[0, 2]

        L[1, 2] = l[4]
        L[2, 1] = L[1, 2]
        d_, V_ = np.linalg.eig(L)
        D = np.diag(d_)
        D[D < 0] = 0.00001

        print("new L close enough?:", np.allclose(L, np.dot(V_, np.dot(D, V_.T))))
        L = np.dot(V_, np.dot(D, V_.T))

        Q = np.linalg.cholesky(L)
        # print(Q)
        R_true = np.dot(M, Q)
        S_true = np.dot(np.linalg.inv(Q), S)
        # print(S_true.T)
        return S_true.T


    def g(self,v1, v2):
        [a, b, c] = v1 
        [x, y, z] = v2

        res = [a*x, 2*a*y, 2*a*z, b*y, 2*b*z, c*z]
        return res
    

    def mat_G_maker(self,G, c, R1, R2, same=0):
        c.fill(same)
        n = R1.shape[0]
        for i in range(n):
            r1 = R1[i, :]
            r2 = R2[i, :]
            res = self.g(r1, r2)
            G[i, :] = res

        return G, c

    def estimateFMwOpticalFlow(self):
        """
        Estimate fundamental matrix using Shi-Tomasi features and optical flow tracker
        """
        W = np.matrix([])
        
        for i in range(len(self.images)-1):
            img1 = cv2.imread(self.images[i])
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(self.images[i+1])
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            corners1 = cv2.goodFeaturesToTrack(gray1, mask=None, **self.feature_params)
            corners2, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners1, None, **self.of_params)

            if corners2 is not None:
                corners1 = np.int32(corners1[st == 1])
                corners2 = np.int32(corners2[st == 1])
            F, mask = cv2.findFundamentalMat(corners1, corners2, cv2.FM_RANSAC, 1.0, 0.99, 3500)
            F = cv2.normalize(F, None)
            self.F_list.append(F)
            pts1 = corners1[mask.ravel() == 1]
            pts2 = corners2[mask.ravel() == 1]
            self.inliers1.append(pts1)
            self.inliers2.append(pts2)
            print(pts1.T)
            # W.append(pts1)
            # W.append(pts2)

        print(np.shape(W))
        np.save('fundamental_Mtx.npy', self.F_list)
        np.save('inliers1.npy', self.inliers1)
        np.save('inliers2.npy', self.inliers2)

        self.inliers1 = np.array(self.inliers1, dtype=object)
        self.inliers2 = np.array(self.inliers2, dtype=object)



    def estimateEssentialMatrix(self):
        
        """
        Estimate essntial matrix from the fundamental matrix and decompose it into
        the rotation and translation vectors
        """

        for F in self.F_list:
            E = np.matmul(F.T,np.matmul(np.diag([1,1,0]),F))
            U,S,Vt = np.linalg.svd(E)
            S[0] = S[1] = (S[0] + S[1]) / 2
            S[2] = 0
            E = U @ np.diag(S) @ Vt
            # E = cv2.normalize(E,None)
            self.E_list.append(E)
        np.save('essentialMtrx.npy',self.E_list)
        self.E_list = np.array(self.E_list)
    
    def recoverPose(self):
        """
        Recover the pose of the 3D point by using the fundamental matrix
        and the inliers of respective frames
        """

        assert len(self.inliers1) == len(self.inliers2)
        points_3d = np.array([])
        for i in range(len(self.inliers1)):
            _,R,t,mask = cv2.recoverPose(self.E_list[i],np.float64(self.inliers1[i]), np.float64(self.inliers2[i]))
            P1 = np.eye(3,4)
            P2 = np.hstack((R,t))
            pts_3d_homogeneous = cv2.triangulatePoints(P1, P2, np.float64(self.inliers1[i]).T, np.float64(self.inliers2[i]).T)
            # pts_3d = (pts_3d_homogeneous / np.tile(pts_3d_homogeneous[-1,:], (4, 1)))[:3,:].T
            pts_3d = cv2.convertPointsFromHomogeneous(pts_3d_homogeneous.T)
            pts_3d = pts_3d.squeeze()[:, :3]
            pts_3d = pts_3d.T
            points_3d = np.append(points_3d,pts_3d)
            # points_3d.append(pts_3d)

        # print(points_3d)
        return np.reshape(points_3d,(-1,3))
    
    def write_ascii_Ply(self,points,filename):
        """
        Write an ASCII output PLY file with the 3D x, y, z coordinates of the points separated by commas.
        """
        # print(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype('float64'))
        o3d.io.write_point_cloud(filename, pcd,write_ascii=True)
    
    def visualisePly(self,filename):
        pcd = o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pcd])




if __name__ == "__main__":
    pathToFiles = '/home/yash/Documents/Computer_VIsion/CV_OptionalProject/hotel'
    Plyfile = '/home/yash/Documents/Computer_VIsion/CV_OptionalProject/output.ply'

    p4 = SFM(pathToFiles)

    pts_3d = p4.estimateOpticalFlow()
    # p4.estimateFMwOpticalFlow()
    # p4.estimateEssentialMatrix()
    # pts_3d = p4.recoverPose()
    p4.write_ascii_Ply(pts_3d,Plyfile)
    p4.visualisePly(Plyfile)
