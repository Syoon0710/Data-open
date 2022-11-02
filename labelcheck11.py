# -*- coding: utf-8 -*-
"""
Analyze Segmented Scan file
and Save individual tooth file

Created on Sun Jul 10 16:27:22 2022

@author: tmkang
"""

"""
1. load segmented obj file
2. analyze obj file : #group, group ID, group data index
3. analyze group data : position, max, min, mid, range in x/y/z direction
4. save individual tooth data (.obj)
5. save group information in a .JSON file
input : segmented scan file
output : tooth_file_name = output_file_name+"-nn.obj"
         json_file_name = output_file_name+".json"
"""
import sys
import os
import trimesh as tm
import numpy as np
import json
from collections import OrderedDict

mVert=np.empty((0,6),float)  # x,y,z,r,g,b
mVNorm=np.empty((0,3),float) # x,y,z
mGroup=np.empty((0,3),int)  # groupID, face0, Nface
mFace=np.empty((0,3),int)   # v1//vn1 v2//vn2 v3//vn3
bBox=np.empty((0,6),float)  # xmin, xmax, ymin, ymax, zmin, zmax

nVert=0
nGroup=0
nFace=0
UpLo=0    # if dataloc="upper", UpLo=1 / if dataloc="lower", UpLo=0

FDI=np.array([[0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28],
              [0,48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]])
   
#------------------------------- Load Segmented OBJ file and analyze 
def load_Seg_file(input_path, input_file_name):
    global mVert, mVNorm, mGroup, mFace, bBox, nVert, nGroup, nFace
    
    print("LABELCHECK-1-OBJ file loading")
    # read OBJ file
    objpathfile=os.path.join(input_path,input_file_name).replace("\\","/")
    if not os.path.isfile(objpathfile):
        print("ERROR-file not found")
    fpObj = open(objpathfile, 'r', encoding='utf-8')
    OBJlines = fpObj.readlines()
    fpObj.close()

    meshSeg = tm.load(objpathfile,process=False)
    # parsing OBJ file
    kg=0
    kf=0

    vt=np.empty((0,6),float)  # x,y,z,r,g,b
    vnt=np.empty((0,3),float) # x,y,z
    gt=np.empty((0,3),int)  # groupID, face0, Nface
    ft=np.empty((0,3),int)   # v1//vn1 v2//vn2 v3//vn3
    vt=np.concatenate((meshSeg.vertices,meshSeg.visual.vertex_colors.astype(float)/255),axis=1)
    vnt=meshSeg.vertex_normals
    ft=meshSeg.faces+1

    for line in OBJlines:
        elm=line[0]
        if elm=="g":      # if group
            if kg>0 :
                gt[kg-1,2]=kf-gt[kg-1,1]
            elm1=line.strip().split(" ")
            gt=np.append(gt,[[int(elm1[1][7:]),kf,0]],axis=0)
            kg=kg+1            
        elif elm=="f":        # if face
            #ft=np.append(ft,[[int(elm[1]),int(elm[3]),int(elm[5])]],axis=0)
            kf=kf+1
    gt[kg-1,2]=kf-gt[kg-1,1]
    
    nVert=len(vt)
    nGroup=kg
    nFace=kf
    
    mVert=vt
    mVNorm=vnt
    mGroup=gt
    mFace=ft

#---------------------------------------- Analyze group data
def analyze_Group_data():
    global mVert, mVNorm, mGroup, mFace, bBox, nVert, nGroup, nFace
    bbxmin=0
    bbxmax=0
    bbymin=0
    bbymax=0
    bbzmin=0
    bbzmax=0

    print("LABELCHECK-2-OBJ file analyze !")

    bbt=np.empty((0,6),float)  # xmin, xmax, ymin, ymax, zmin, zmax
    # group bbox calc
    for i in range(nGroup):
        for j in range(mGroup[i,2]):
            if j==0 :
                bbxmin=mVert[mFace[mGroup[i,1]+j,0]-1,0]
                bbxmax=bbxmin
                bbymin=mVert[mFace[mGroup[i,1]+j,0]-1,1]
                bbymax=bbymin
                bbzmin=mVert[mFace[mGroup[i,1]+j,0]-1,2]
                bbzmax=bbzmin
            if bbxmin>mVert[mFace[mGroup[i,1]+j,0]-1,0] :
                bbxmin=mVert[mFace[mGroup[i,1]+j,0]-1,0]
            if bbxmax<mVert[mFace[mGroup[i,1]+j,0]-1,0] :
                bbxmax=mVert[mFace[mGroup[i,1]+j,0]-1,0]
            if bbymin>mVert[mFace[mGroup[i,1]+j,0]-1,1] :
                bbymin=mVert[mFace[mGroup[i,1]+j,0]-1,1]
            if bbymax<mVert[mFace[mGroup[i,1]+j,0]-1,1] :
                bbymax=mVert[mFace[mGroup[i,1]+j,0]-1,1]
            if bbzmin>mVert[mFace[mGroup[i,1]+j,0]-1,2] :
                bbzmin=mVert[mFace[mGroup[i,1]+j,0]-1,2]
            if bbzmax<mVert[mFace[mGroup[i,1]+j,0]-1,2] :
                bbzmax=mVert[mFace[mGroup[i,1]+j,0]-1,2]
        bbt=np.append(bbt,[[bbxmin,bbxmax, bbymin, bbymax, bbzmin, bbzmax]],axis=0)
    bBox=bbt

#----------------------------------------Save tooth information
def save_JSON_file(output_path, output_file_name):
    global mVert, mVNorm, mGroup, mFace, bBox, nVertexUp, nGroupUp, nFaceUp
    global UpLo, FDI

    #global jpt
    print("LEBELCHECK-4-JSON file save !")
    json_file_name=output_file_name.split(".")[0]+".json"
    outfile=os.path.join(output_path,json_file_name).replace("\\","/")

    json_data = OrderedDict()
    
    json_data={}
    json_data['tooth'] = []

    for j in range(nGroup):
        gID=mGroup[j,0]
        xCenter=float((bBox[j,1]+bBox[j,0])/2)
        xSize=float((bBox[j,1]-bBox[j,0]))
        yCenter=float((bBox[j,3]+bBox[j,2])/2)
        ySize=float((bBox[j,3]-bBox[j,2]))
        zCenter=float((bBox[j,5]+bBox[j,4])/2)
        zSize=float((bBox[j,5]-bBox[j,4]))
    
        json_data["tooth"].append({
            "number": int(FDI[UpLo,gID]),
            "label": int(gID),
            "keypoint":[xCenter,yCenter,zCenter],
            "size":[xSize,ySize,zSize],
            "diag":"normal"
            })
    jfp=open(outfile, 'w', encoding="utf-8")
    json.dump(json_data, jfp) #, ensure_ascii=False, indent="\t")
    jfp.close()
 
def labelcheck(input_path, input_file_name, output_path, output_file_name, dataloc):
    global UpLo
    
    UpLo=1
    if dataloc=="upper":
        UpLo=0
        
    load_Seg_file(input_path, input_file_name)
    analyze_Group_data()
    save_JSON_file(output_path, output_file_name)
