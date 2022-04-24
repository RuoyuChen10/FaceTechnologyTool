# -*- coding: utf-8 -*-  

"""
Created on 2021/12/10

@author: Ruoyu Chen
Refer to the code from https://github.com/Naurislv/facial_image_averaging
Refer to the idea from https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
"""

# Standard imports
import os
import math
import sys
import glob

# Local imports
from face_landmarks import detect_landmarks

# Dependecy imports
import cv2
import numpy as np

class AverageFace():
    """
    demo of generate the average face
    """
    def __init__(self, weight, height):
        super(AverageFace, self).__init__()
        # weight and height is the image size for generate
        self.weight = weight
        self.height = height

    def similarityTransform(self, inPoints, outPoints):
        """
        Compute similarity transform given two sets of two points.
        OpenCV requires 3 pairs of corresponding points.
        We are faking the third one.
        """
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)  
    
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        # tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        tform, inliers = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
        
        return tform

    def rectContains(self, rect, point):
        """
        Check if a point is inside a rectangle
        """
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True

    def calculateDelaunayTriangles(self, rect, points):
        """
        Calculate delanauy triangle
        """
        # Create subdiv
        subdiv = cv2.Subdiv2D(rect)
        
        # Insert points into subdiv
        for p in points:
            subdiv.insert((p[0], p[1]))
        # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
        triangleList = subdiv.getTriangleList()
        # Find the indices of triangles in the points array
        delaunayTri = []
        
        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])        
            
            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):                    
                        if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)                            
                if len(ind) == 3:                                                
                    delaunayTri.append((ind[0], ind[1], ind[2]))
        return delaunayTri

    def constrainPoint(self, p, w, h) :
        p =  (min(max(p[0], 0), w-1), min(max(p[1], 0), h-1))
        return p

    def applyAffineTransform(self, src, srcTri, dstTri, size):
        """
        Apply affine transform calculated using srcTri and dstTri to src and
        output an image of size.
        """
        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        
        return dst

    def warpTriangle(self, img1, img2, t1, t2) :
        """
        Warps and alpha blends triangular regions from img1 and img2 to img
        """
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = [] 
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        
        size = (r2[2], r2[3])

        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        
        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

    def face_average(self, images, allPoints):
        """
        main function
        images: Input a list [image1,image2]; each image: shape[W,H,C], opencv-python format, channel order: BGR; the image intensity of each pixel ranges from 0 to 1.
        allPoints: point
        """
        # Eye corners
        eyecornerDst = [ (np.int(0.3 * self.weight ), np.int(self.height / 3)), (np.int(0.7 * self.weight ), np.int(self.height / 3)) ]
        
        imagesNorm = []
        pointsNorm = []
        
        # Add boundary points for delaunay triangulation
        boundaryPts = np.array([(0,0), (self.weight/2, 0), (self.weight-1, 0), (self.weight-1, self.height/2), (self.weight-1, self.height-1), (self.weight/2, self.height-1), (0, self.height-1), (0, self.height/2) ])
        
        # Initialize location of average points to 0s
        pointsAvg = np.array([(0,0)]* (len(allPoints[0]) + len(boundaryPts)), np.float32())
        
        n = len(allPoints[0])
        numImages = len(images)
        
        # Warp images and trasnform landmarks to output coordinate system,
        # and find average of transformed landmarks.
        
        for i in range(0, numImages):

            points1 = allPoints[i]

            # Corners of the eye in input image
            eyecornerSrc  = [allPoints[i][36], allPoints[i][45]] 
            
            # Compute similarity transform
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst)
            
            # Apply similarity transformation
            img = cv2.warpAffine(images[i], tform, (self.weight,self.height))

            # Apply similarity transform on points
            points2 = np.reshape(np.array(points1), (68,1,2))        
            
            points = cv2.transform(points2, tform)
            
            points = np.float32(np.reshape(points, (68, 2)))
            
            # Append boundary points. Will be used in Delaunay Triangulation
            points = np.append(points, boundaryPts, axis=0)
            
            # Calculate location of average landmark points.
            pointsAvg = pointsAvg + points / numImages
            
            pointsNorm.append(points)
            imagesNorm.append(img)
        
        # Delaunay triangulation
        rect = (0, 0, self.weight, self.height)
        dt = self.calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # Output image
        output = np.zeros((self.height, self.weight, 3), np.float32())

        # Warp input images to average image landmarks
        for i in range(0, len(imagesNorm)) :
            img = np.zeros((self.height, self.weight, 3), np.float32())
            try:
                # Here may exist bug
                # Transform triangles one by one
                for j in range(0, len(dt)) :
                    tin = [] 
                    tout = []
                    
                    for k in range(0, 3) :                
                        pIn = pointsNorm[i][dt[j][k]]
                        pIn = self.constrainPoint(pIn, self.weight, self.height)
                        
                        pOut = pointsAvg[dt[j][k]]
                        pOut = self.constrainPoint(pOut, self.weight, self.height)
                        
                        tin.append(pIn)
                        tout.append(pOut)
                    
                    self.warpTriangle(imagesNorm[i], img, tin, tout)


                # Add image intensities for averaging
                output = output + img
            except:
                numImages -= 1

        # Divide by numImages to get average
        output = output / numImages

        return output