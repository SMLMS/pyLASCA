#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:09:48 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt


class Transformator:
    def __init__(self):
        self._time = int()
        self._pxlSize = float()
        self._rawData = pd.DataFrame()
        self._landmarks = pd.DataFrame()
        self._landmarkType = str()
        self._affineMatrix = np.zeros([3,3], float)
        
    @property
    def time(self) -> int:
        return(self._time)
    
    @time.setter
    def time(self, value: int):
        self._time = value
    
    @property
    def pxlSize(self) -> float:
        return(self._pxlSize)
    
    @pxlSize.setter
    def pxlSize(self, value: float):
        self._pxlSize = value
    
    @property
    def rawData(self) -> pd.DataFrame:
        return(self._rawData.copy())
    
    @rawData.setter
    def rawData(self, df: pd.DataFrame):
        self._rawData = df.copy()
        
    @property
    def landmarks(self) -> pd.DataFrame:
        return(self._landmarks.copy())
    
    @landmarks.setter
    def landmarks(self, df: pd.DataFrame):
        self._landmarks = df.copy()
        
    @property
    def landmarkType(self) -> str():
        return(self._landmarkType)
    
    @landmarkType.setter
    def landmarkType(self, value: str):
        if (value == "cross" or value == "square"):
            self._landmarkType = value
        else:
            raise ValueError("Could not set instance variable landmarksType to %s. Must be 'cross' or 'square'." %(value))
            
    @property
    def affineMatrix(self) -> np.ndarray:
        return(self._affineMatrix.copy())
        
    def __str__(self):
        message = str("Time: %i\n" % (self.time))
        message += str("\nrawData:\n")
        message += str(self.rawData.head())
        message += str("\nlandmarktype: %s\n" % (self.landmarkType))
        message += str("landmarks:\n")
        message += str(self.landmarks)
        message += str("\naffineMatrix:\n")
        message += str(self.affineMatrix)
        return(message)
        
    def __del__(self):
        print("Released insatnce of Transformator class from heap.")

    def importRawData(self, fileName: str, sheetName: str):
        # =============================================================================
        # load data
        # =============================================================================
        try:
            sheetData = pd.read_excel(io = fileName, sheet_name = sheetName)
        except:
            print("could not read rawData from sheet %s in file %s" %(sheetName, fileName))
        applType = str()
        if (self.landmarkType == "cross"):
            applType = "i.c."
        elif (self.landmarkType == "square"):
            applType =  "topic"
        else:
            raise ValueError("Error in importRawData function of transformator class. Instance variable landmarksType must be 'cross' or 'square'.") 
        
        filteredData = sheetData[sheetData["t_[min]"] == self.time]
        filteredData = filteredData[filteredData["appl_type"] == applType]
# =============================================================================
#         if (filteredData["appl_side"].values[0] == "left"):
#             #alpha = np.array([0,0,45,45,90,90,135,135,180,180,225,225,270,270,315,315])
#             alpha = np.array([180,180,135,135,90,90,45,45,0,0,315,315,270,270,225,225])
#         elif (filteredData["appl_side"].values[0] == "right"):
#             alpha = np.array([180,180,135,135,90,90,45,45,0,0,315,315,270,270,225,225])
#         else:
#             raise ValueError("Error in importRawData function of transformator class. Instance variable landmarksType must be 'cross' or 'square'.") 
# =============================================================================
        
        filteredData = pd.melt(filteredData, id_vars=['effect'], value_vars=["d1_[cm]",
                                                                             "d2_[cm]",
                                                                             "d3_[cm]",
                                                                             "d4_[cm]",
                                                                             "d5_[cm]",
                                                                             "d6_[cm]",
                                                                             "d7_[cm]",
                                                                             "d8_[cm]"])
        
        alpha = np.array([180,180,225,225,270,270,315,315,0,0,45,45,90,90,135,135])
        filteredData["alpha"] = alpha
        self.rawData = pd.DataFrame({"probandID" : np.full(16, sheetName),
                                     "appl_type": np.full(16, applType),
                                     "effect": filteredData["effect"],
                                     "t_[min]": np.full(16, self.time),
                                     "alpha": filteredData["alpha"],
                                     "d_[mm]": 10.0 * filteredData["value"]
                                     })
        self._rawData = self._rawData.sort_values(by = ["effect", "alpha"])
        self._rawData.reset_index(drop=True, inplace=True)
        
    def importLandmarks(self, fileName: str):
        # =============================================================================
        # load data
        # =============================================================================
        try:
            self.landmarks = pd.read_csv(filepath_or_buffer = fileName)
        except:
            print("could not read landmarks from %s" %fileName)
            
    def identifyLandmarkType(self):
        cross = np.array([0,90,180,270])
        if ((self.landmarks["alpha"] == cross).all()):
            self.landmarkType = "cross"
        else:
            self.landmarkType = "square"
            
    def rotateLandmarks(self, deg: int):
        alpha = self.landmarks["alpha"] + deg
        idx = 360 <= alpha
        alpha[idx] = alpha[idx] - 360
        self._landmarks["alpha"] = alpha
        self._landmarks = self._landmarks.sort_values(by = ["alpha"])
        self._landmarks.reset_index(drop=True, inplace=True)
        
    def rotateRawData(self, deg: int):
        alpha = self.rawData["alpha"] + deg
        idx = 360 <= alpha
        alpha[idx] = alpha[idx] - 360
        self._rawData["alpha"] = alpha
        self._rawData = self._rawData.sort_values(by = ["effect", "alpha"])
        self._rawData.reset_index(drop=True, inplace=True)
            
    def createAffineMatrix(self):
        entries = 4
        if (self.landmarkType == "cross"):
            x_array = np.array([0.0, 10.0, 0.0, -10.0])
            y_array = np.array([10.0, 0.0, -10.0, 0.0])
        elif (self.landmarkType == "square"):
            x_array = np.array([15.0, 15.0, -15.0, -15.0])
            y_array = np.array([15.0, -15.0, -15.0, 15.0])
        else:
            raise ValueError("Error in importRawData function of transformator class. Instance variable landmarksType must be 'cross' or 'square'.")
        mat = np.zeros ([2*entries,6], float)
        mat[0:entries,0] = x_array
        mat[0:entries,1] = y_array
        mat[0:entries,2] = 1
        mat[entries:2*entries,3] = x_array
        mat[entries:2*entries,4] = y_array
        mat[entries:2*entries,5] = 1        

        vec = self.landmarks["x_mm"].append(self.landmarks["y_mm"])
        
        x = np.linalg.lstsq(mat,vec,rcond=None)[0]
        self._affineMatrix = np.zeros([3,3], float)
        self._affineMatrix[0,0] = x[0]
        self._affineMatrix[0,1] = x[1]
        self._affineMatrix[0,2] = x[2]
        self._affineMatrix[1,0] = x[3]
        self._affineMatrix[1,1] = x[4]
        self._affineMatrix[1,2] = x[5]
        self._affineMatrix[2,2] = 1
        
    def createRotationMatrix(self):
        print(self._landmarks)
        dxc = np.sum(self.landmarks["x_mm"])/4.0
        dyc = np.sum(self.landmarks["y_mm"])/4.0
        
        dx1 = self.landmarks["x_mm"][0] - self.landmarks["x_mm"][2]
        dy1 = self.landmarks["y_mm"][0] - self.landmarks["y_mm"][2]
        phi1 = np.arctan2(dx1, dy1) * 180.0 / np.pi
        
        dx2 = self.landmarks["x_mm"][1] - self.landmarks["x_mm"][3]
        dy2 = self.landmarks["y_mm"][1] - self.landmarks["y_mm"][3]
        phi2 = np.arctan2(dx2, dy2) * 180.0 / np.pi
        
        phi = ((np.max([phi1,phi2])-90.0 + np.min([phi1,phi2]))/2.0)
        
        Phi = phi * np.pi/180.0
        self._affineMatrix = np.zeros([3,3], float)
        self._affineMatrix[0,0] = np.cos(Phi)
        self._affineMatrix[0,1] = np.sin(Phi)
        self._affineMatrix[0,2] = dxc
        self._affineMatrix[1,0] = -1.0*np.sin(Phi)
        self._affineMatrix[1,1] = np.cos(Phi)
        self._affineMatrix[1,2] = dyc
        self._affineMatrix[2,2] = 1
        
    def createTransformationMatrix(self):
        if (self.landmarkType == "cross"):
            self.createRotationMatrix()
        elif (self.landmarkType == "square"):
            self.createAffineMatrix()
        else:
            raise ValueError("Error in importRawData function of transformator class. Instance variable landmarksType must be 'cross' or 'square'.")
        

# =============================================================================
#     def createAffineMatrix(self):
#         if (self.landmarkType == "cross"):
#             x_array = np.array([0.0, 10.0, 0.0, -10.0])
#             y_array = np.array([10.0, 0.0, -10.0, 0.0])
#         elif (self.landmarkType == "square"):
#             x_array = np.array([30.0, 30.0, -30.0, -30.0])
#             y_array = np.array([30.0, -30.0, -30.0, 30.0])
#         else:
#             raise ValueError("Error in importRawData function of transformator class. Instance variable landmarksType must be 'cross' or 'square'.")
#             
#         entries = 4
#         idx = np.arange(entries)
#         idx = idx[1:] - (idx[:, None] >= idx[1:])
#         mat = np.zeros ([2*(entries-1),6], float)
#         aMat = np.zeros([entries,3,3], float)
#         for i in range(entries):
#             mat[0:entries-1,0] = x_array[idx[i,:]]
#             mat[0:entries-1,1] = y_array[idx[i,:]]
#             mat[0:entries-1,2] = 1
#             mat[entries-1:2*(entries-1),3] = x_array[idx[i,:]]
#             mat[entries-1:2*(entries-1),4] = y_array[idx[i,:]]
#             mat[entries-1:2*(entries-1),5] = 1        
# 
#             vec = self.landmarks["x_mm"][idx[i,:]].append(self.landmarks["y_mm"][idx[i,:]])
#         
#             x = np.linalg.lstsq(mat,vec,rcond=None)[0]
#         
#             aMat[i,0,0] = x[0]
#             aMat[i,0,1] = x[1]
#             aMat[i,0,2] = x[2]
#             aMat[i,1,0] = x[3]
#             aMat[i,1,1] = x[4]
#             aMat[i,1,2] = x[5]
#             aMat[i,2,2] = 1
#         
#         self._affineMatrix = np.mean(aMat, axis = 0)
# =============================================================================

        
    def transformToCartesian(self):
        self._rawData["x_[mm]"] = self._rawData["d_[mm]"] * np.cos(self._rawData["alpha"] * np.pi/180.0)
        self._rawData["y_[mm]"] = self._rawData["d_[mm]"] * np.sin(self._rawData["alpha"] * np.pi/180.0)
            
    def transformToImage(self):
        x = np.zeros([16,1], float)
        y = np.zeros([16,1], float)
        points = np.ones([16,3], float)
        points[:,0] = self.rawData["x_[mm]"]
        points[:,1] = self.rawData["y_[mm]"]
        for i in range(np.shape(points)[0]):
            t_point = np.dot(self.affineMatrix, points[i,:])
            x[i] = t_point[0]
            y[i] = t_point[1]
        self._rawData["x_img_[mm]"] = x
        self._rawData["y_img_[mm]"] = y
        
    def transfromToPxl(self):
        self._rawData["x_img_[pxl]"] = self.rawData["x_img_[mm]"] * (1.0/self.pxlSize)
        self._rawData["y_img_[pxl]"] = self.rawData["y_img_[mm]"] * (1.0/self.pxlSize)
        
    def transform(self):
        self.transformToCartesian()
        self.createTransformationMatrix()
        self.transformToImage()
        self.transfromToPxl()
            
    def plotTimePoint(self):
        appl = self.rawData["probandID"].values[0]
        aType = self.rawData["appl_type"].values[0]
        
        x_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["x_[mm]"].values[:]
        y_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["y_[mm]"].values[:]
        x_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["x_[mm]"].values[:]
        y_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["y_[mm]"].values[:]
        
        x_allodyn_tf = self.rawData[self.rawData["effect"] == "allodynia"]["x_img_[pxl]"].values[:]
        y_allodyn_tf = self.rawData[self.rawData["effect"] == "allodynia"]["y_img_[pxl]"].values[:]
        x_hyperal_tf = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["x_img_[pxl]"].values[:]
        y_hyperal_tf = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["y_img_[pxl]"].values[:]

        fig, (ax1, ax2) = plt.subplots(1,2)  # Create a figure and an axes.
        ax1.fill(x_hyperal, y_hyperal, facecolor='none', edgecolor='cyan', linewidth=3, label="sec_hyperalgesia") 
        ax1.fill(x_allodyn, y_allodyn, facecolor='none', edgecolor='purple', linewidth=3, label="allodynia")  # Plot some data on the axes.
        ax1.set_aspect('equal', 'box')
        ax1.set_title("%s with %s at t = %i min" %(appl, aType, self.time))  # Add a title to the axes.
        ax1.set_xlabel("abscissa [mm]")
        ax1.set_ylabel("ordinate [mm]")
        ax1.legend()  # Add a legend.
        
        ax2.fill(x_hyperal_tf, y_hyperal_tf, facecolor='none', edgecolor='cyan', linewidth=3, label="sec_hyperalgesia") 
        ax2.fill(x_allodyn_tf, y_allodyn_tf, facecolor='none', edgecolor='purple', linewidth=3, label="allodynia")  # Plot some data on the axes.
        ax2.set_aspect('equal', 'box')
        ax2.set_title("%s with %s at t = %i min" %(appl, aType, self.time))  # Add a title to the axes.
        ax2.set_xlabel("abscissa [pxl]")
        ax2.set_ylabel("ordinate [pxl]")
        ax2.legend()  # Add a legend.
        plt.show()
        
    def plotTimePointOnImage(self, img: np.ndarray):
        appl = self.rawData["probandID"].values[0]
        aType = self.rawData["appl_type"].values[0]
        
        x_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["x_img_[pxl]"].values[:]
        y_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["y_img_[pxl]"].values[:]
        x_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["x_img_[pxl]"].values[:]
        y_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["y_img_[pxl]"].values[:]

        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.imshow(img)
        ax.fill(x_hyperal, y_hyperal, facecolor='none', edgecolor='cyan', linewidth=3, label="sec_hyperalgesia") 
        ax.fill(x_allodyn, y_allodyn, facecolor='none', edgecolor='purple', linewidth=3, label="allodynia")  # Plot some data on the axes.
        
        ax.set_aspect('equal', 'box')
        ax.set_title("%s with %s at t = %i min" %(appl, aType, self.time))  # Add a title to the axes.
        ax.set_xlim([0,752])
        ax.set_ylim([0,580])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.xticks([])
        plt.yticks([])
        ax.legend()  # Add a legend.
        plt.show()
        
    def saveContoursOnImage(self, img: np.ndarray, fileName: str):
        
        x_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["x_img_[pxl]"].values[:]
        y_allodyn = self.rawData[self.rawData["effect"] == "allodynia"]["y_img_[pxl]"].values[:]
        x_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["x_img_[pxl]"].values[:]
        y_hyperal = self.rawData[self.rawData["effect"] == "sec_hyperalgesia"]["y_img_[pxl]"].values[:]

        xa2 = int(x_allodyn[-1])
        ya2 = int(y_allodyn[-1])
        xh2 = int(x_hyperal[-1])
        yh2 = int(y_hyperal[-1])
        for i in range(len(x_allodyn)):
            xa1 = int(x_allodyn[i])
            ya1 = int(y_allodyn[i])
            xh1 = int(x_hyperal[i])
            yh1 = int(y_hyperal[i])
            cv.line(img,(xa1,ya1),(xa2,ya2),(255,0,255),3)
            cv.line(img,(xh1,yh1),(xh2,yh2),(255,255,0),3)
            xa2 = xa1
            ya2 = ya1
            xh2 = xh1
            yh2 = yh1
        cv.imwrite(filename = fileName, img =img)
# =============================================================================
#         source_window = 'Source'
#         cv.namedWindow(source_window)
#         cv.imshow("test", img)
#         while(True):
#             k = cv.waitKey(33)
#             if k == -1:  # if no key was pressed, -1 is returned
#                 continue
#             else:
#                 break
#         cv.destroyAllWindows()
#         cv.waitKey(1)
# =============================================================================
        
        
    def saveTransformationResults(self, fileName: str):
        self.rawData.to_csv(path_or_buf = fileName, index = False)
        print("Transformation results written to  %s" %(fileName))
        
    def saveAffineMatrix(self, fileName: str):
        np.savetxt(fileName, self.affineMatrix , delimiter=",")
        print("Affine matrix written to  %s" %(fileName))
        
    
def main():
    fileName_landmarks = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 02/Tag 1/SCTCAPS 02 #1 15min (Colour)_Landmarks_200529.csv"
    #fileName_landmarks = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 01/Tag 2/SCTCAPS 01 #2  15min (Colour)_Landmarks_sig_56_200605.csv"
    fileName_rawData = "/Users/malkusch/PowerFolders/LaSCA/mechanic/sctcaps_01-19_Datensatz.xlsx"
    sheetName = "sctcaps01" 
    tf = Transformator()
    tf.time = 15
    #tf.pxlSize = 0.24099596478356566
    tf.pxlSize = 0.2444488261188555
    tf.importLandmarks(fileName = fileName_landmarks)
    tf.identifyLandmarkType()
    tf.importRawData(fileName = fileName_rawData, sheetName = sheetName)
    #tf.transform()
    #tf.plotTimePoint()
    #fileName_img = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 01/Tag 1/SCTCAPS 01 #1 15min (Colour).jpg"
    #fileName_img = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 01/Tag 2/SCTCAPS 01 #2  15min (Colour).jpg"
    #rawImage = cv.imread(fileName_img, cv.IMREAD_COLOR)
    #rawImage = cv.cvtColor(rawImage, cv.COLOR_BGR2RGB)
    #tf.plotTimePointOnImage(img = rawImage)
    #rawImage = cv.flip(rawImage, -1)
    tf.rotateLandmarks(deg = 90)
    
    #print(tf)

     
if __name__ == '__main__':
    main()