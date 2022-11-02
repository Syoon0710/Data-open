import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
from labelcheck11_open import *
import vedo
import pandas as pd
from scipy.spatial import distance_matrix
import shutil
import time
from pygco import cut_from_graph
import random
import sys
import scipy.io as sio
from pytz import timezone
from datetime import datetime

# from sklearn.svm import SVC # uncomment this line if you don't install thudersvm
# from thundersvm import SVC # comment this line if you don't install thudersvm
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from tqdm import tqdm
from esay_mesh_vtk import Easy_Mesh

import logging

logger = None

def vtp_to_obj(vtpname,result_path,resultname):
    mesh = Easy_Mesh(vtpname)
    mesh.to_obj('Sample_')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    path = os.path.dirname(os.path.abspath(__file__))
    rootpath = path + '/vtptoobj'
    datapath = result_path +'/'+ resultname
    
    vertex_list = list()
    vertex_list_check = list()
    vertex_dir_list = list()
    face_list = list()
    face_dir_list = list()
    color = [[10,0,10],
            [10,10,0],
            [4,10,10],
            [10,10,4],
            [8,10,10],
            [10,8,10],
            [10,8,6],
            [10,6,8],
            [8,10,6],
            [8,6,10],
            [6,8,10],
            [10,8,2],
            [10,4,4],
            [4,10,4],
            [4,4,10]]

    def check_dir(dir):
        file_list = os.listdir(dir)

        for file in file_list:        
            if os.path.isdir(dir+r"/"+file) == False :
                if file[-4:] == '.obj':
                    split_file = file.split('.')
                    if  len(split_file) == 2:
                        face_dir_list.append(dir+r"/"+file)
                    else:
                        vertex_dir_list.append(dir+r"/"+file)

    check_dir(rootpath)

    with open(face_dir_list[0],'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == 'v':
            vertex_list.append(line)
            vertex_list_check.append(0)
        else:
            break

    color_ind = -1
    for file in face_dir_list:
        color_ind += 1
        r = random.randint(0,10) /10
        r = color[color_ind][0] / 10
        g = random.randint(0,10)  / 10
        g = color[color_ind][1]  / 10
        b = random.randint(0,10)  / 10
        b = color[color_ind][2]  / 10
        with open(file,'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line[0] == 'v':
                continue
            elif line[0] == 'g':
                face_list.append('# mm_gid 0\n')
                face_list.append(line)
            elif line[0] == 'f':
                face_list.append(line)
                split_line = line.split()
                f_l = list()
                first = split_line[1].split('//')
                first = int(first[0])
                f_l.append(first)
                sec = split_line[2].split('//')
                sec = int(sec[0])
                f_l.append(sec)
                thr = split_line[3].split('//')
                thr = int(thr[0])
                f_l.append(thr)
                for f in f_l:
                    if vertex_list_check[f - 1] == 0:
                        vertex_list[f - 1] = vertex_list[f - 1].strip()
                        vertex_list[f - 1] = vertex_list[f - 1] + ' ' + str(r) + '00000 ' + str(g) + '00000 ' + str(b) + '00000\n'
                        vertex_list_check[f - 1] = 1
                    else:
                        continue

    with open(datapath, 'w') as out:
        ind = 0
        for vertex in vertex_list:
            ver_split = vertex.split()
            num = float(ver_split[1])
            ver_split[1] = str(round(num,6))
            num = float(ver_split[2])
            ver_split[2] = str(round(num,6))
            num = float(ver_split[3])
            ver_split[3] = str(round(num,6))
            out.write(ver_split[0] + ' ' + ver_split[1]+ ' ' + ver_split[2]+ ' ' + ver_split[3]+ ' ' + ver_split[4]+ ' ' + ver_split[5]+ ' ' + ver_split[6] + '\n')

        for face in face_list:
            #if face_list_check[ind] == 1:
            #    ind+=1
            #    continue
            out.write(face)
            ind+=1
    shutil.rmtree(rootpath)
    # shutil.rmtree('./temp')

def main(input_path,input_data,result_path,result_name,sex,age,dataloc):
    # Logger
    global logger
    logging.basicConfig(filename="AI.log", filemode="a+", format="[%(asctime)s][%(levelname)s] - %(message)s", level=logging.INFO)
    logger = logging.getLogger()
    
    log("START main - {} {} {} {} {} {} {}".format(input_path, input_data, result_path, result_name, sex, age,dataloc))
    try:

        temp_path = input_path
        new_file_name = input_data

        gpu_id = 0
        torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

        upsampling_method = 'KNN'

        model_path = './models'
        if dataloc == 'U':
            model_name = 'MeshSNet_G_Upper_best.tar'
        elif dataloc == 'L':
            model_name = 'MeshSNet_G_Lower_best.tar'
        # mesh_path = input_path  # need to modify
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        num_classes = 15
        num_channels = 15

        # set model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MeshSegNet_G(num_classes=num_classes, num_channels=num_channels).cuda()
        model = nn.DataParallel(model, device_ids = [0,1,2])
        
        if(device.type =="cuda")and(torch.cuda.device_count()>1):
            print('Multi GPU activate')
        # load trained model
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        model = nn.DataParallel(model, device_ids = [0])
        model.to(f'cuda:{model.device_ids[0]}')
        #cudnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


        model.eval()
        with torch.no_grad():


            start_time = time.time()

            print('Predicting Sample filename: {}'.format(input_data))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(input_path, input_data))

            # pre-processing: downsampling
            if mesh.NCells() > 50000:
                print('\tDownsampling...')
                target_num = 30000
                ratio = target_num/mesh.NCells() # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
                predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
            else:
                mesh_d = mesh.clone()
                predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            cells = np.zeros([mesh_d.NCells(), 9], dtype='float32')
            for i in range(len(cells)):
                cells[i][0], cells[i][1], cells[i][2] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            original_cells_d = cells.copy()

            mean_cell_centers = mesh_d.centerOfMass()
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2)
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            mesh_d.celldata['Normal'] = mesh_normals

            # preprae input
            points = mesh_d.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            barycenters = mesh_d.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # computing A_S and A_L

            arr = np.array([X.shape[0], X.shape[0]],dtype=np.int32)

            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = torch.cdist(torch.from_numpy(X[:, 9:12]),torch.from_numpy(X[:, 9:12]))

            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))
            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(f'cuda:{model.device_ids[0]}', dtype=torch.float32)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(f'cuda:{model.device_ids[0]}', dtype=torch.float32)
            A_L = torch.from_numpy(A_L).to(f'cuda:{model.device_ids[0]}', dtype=torch.float32)

            # tensor_prob_output = model(X, A_S, A_L,A_U)
            tensor_prob_output = model(X, A_S, A_L)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label']=predicted_labels_d

            # refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

            # unaries
            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            # parawise
            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            #edges
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            cells = original_cells_d.copy()
            barycenters = mesh_d.cellCenters() # don't need to copy
            cell_ids = np.asarray(mesh_d.faces())

            lambda_c = 30
            edges = np.empty([1, 3], order='C')
            for i_node in range(cells.shape[0]):
                # Find neighbors
                nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
                nei_id = np.where(nei==2)
                for i_nei in nei_id[0][:]:
                    if i_node < i_nei:
                        cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                        if cos_theta >= 1.0:
                            cos_theta = 0.9999
                        theta = np.arccos(cos_theta)
                        phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                        if theta > np.pi/2.0:
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                        else:
                            beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            edges = np.delete(edges, 0, 0)
            edges[:, 2] *= lambda_c*round_factor
            edges = edges.astype(np.int32)

            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            # output refined result
            mesh3 = mesh_d.clone()
            mesh3.celldata['Label']=refine_labels

            # get fine_cells
            cells = np.zeros([mesh.NCells(), 9], dtype='float64')
            for i in range(len(cells)):
                cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            fine_cells = cells

            barycenters = mesh3.cellCenters() # don't need to copy
            fine_barycenters = mesh.cellCenters() # don't need to copy

            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(barycenters, np.ravel(refine_labels))
            fine_labels = neigh.predict(fine_barycenters)
            fine_labels = fine_labels.reshape(-1, 1)

            # seg 결과(_Segmented.obj), json도 같은이름으로
            mesh.celldata['Label'] = fine_labels
            vedo.write(mesh3, os.path.join(temp_path, '{}_seg.vtp'.format(new_file_name[:-4])))
            vtp_to_obj(temp_path+'/'+'{}_seg.vtp'.format(new_file_name[:-4]),temp_path,f"{new_file_name[:-4]}_seg.obj")

        # labelcheck 호출 ( seg_obj 파일 이름, json파일이름, )
        labelcheck(temp_path, f"{new_file_name[:-4]}_seg.obj", temp_path, f"{new_file_name[:-4]}.json", dataloc)
        # input_path : 

        # 결과 obj, json을 지정 폴더에 zip파일로 저장 > 원래 인풋 파일네임_segmented.zip
        curDir = os.getcwd()
        os.chdir(input_path)
        result_zip = zipfile.ZipFile(os.path.join(result_path, result_name), 'w')
        files = os.listdir(temp_path)
        result_zip.write(f"{new_file_name[:-4]}_seg.obj", compress_type = zipfile.ZIP_DEFLATED)
        result_zip.write(f"{new_file_name[:-4]}.json", compress_type = zipfile.ZIP_DEFLATED)
        result_zip.close()
        os.chdir(curDir)
    except Exception as e:
        log(e.__str__(), True)
        
    log("FINISHED main")
    
def log(logMessage, exc_info = False):
    global logger
    
    if exc_info :
        logger.error(logMessage, exc_info=True)
    else:
        logger.info(logMessage)
    
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])