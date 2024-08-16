import numpy as np
import networkx as nx

import os
from pathlib import Path
import math
import random

import matplotlib.pyplot as plt





# include functions from ect_utils
from ect_utils import ECC



mypath = '../data/ALLleaves/'
mypath_output = '../data/ALLleaves_ECT_rotate/'

def rotate_around_point(point, radians, origin=(0, 0)):
                """Rotate a point around an origin (default = (0,0)).
                """
                x, y = point
                ox, oy = origin

                rx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
                ry = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

                return rx, ry

def rotate(pos, theta):
    pos_rotate = {}
    for p in pos:
        point = pos[p]
        pos_rotate[p]=(rotate_around_point(point, theta, origin = (0,0)))
    return pos_rotate

options = {
    "node_size": 0,
    "node_color": "black",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 2,
    'with_labels':False,
}

# From previously computed bounding box
r = 2.9092515639765497
# ADD in list of species already computed ECT for
#done_species = ['Arabidopsis_asymmetry', 'rapa', 'Co2','Co3','Co4','Co5','Co6', 'GrapeMay28.2011','GrapeMay29.2011','GrapeMay30.2011','GrapeMay31.2011','Geneva_2013','Geneva_2015', 'Geneva_2016', 'MtVernon_scans', 'B2','B3','chamber','field','wild','Tomato_asymmetry','Tomato_entire', 'V2']
done_species = ['Alstroemeria', 'Apple', '1Arabidopsis', 'napus', 'Co1', 'Cotton', 'GrapeJune1.2011', 'Grass','Ivy', 'Passiflora', 'Pepper', 'Potato','B1', 'V1']
NUM_ROTATIONS = 10
#rotation_angle = [random.uniform(0, 2*np.pi) for i in range(NUM_ROTATIONS)]
rotation_angle= [1.165437796416464, 3.906434868614607, 5.468126549189142, 5.144402255741231, 5.754563650541671, 4.760881857822241, 2.486110454278915, 2.204038763613354, 3.50796449906843, 4.8978570600402]
# loop through file system
classes=[]
class_count = 0
for path, subdirs, files in os.walk(mypath):
    classes.extend(subdirs)
    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if (d[0] != '.' and d not in done_species)]
    print('Computing ECT of files in ', path, '...')
    print(len(files))
    for name in files:
        input_filedir = os.path.join(path, name)
        if class_count< 50:
            leaf = np.load(input_filedir)
        
            # Create graph of leaf outline
            G = nx.Graph()
            for i in range(np.shape(leaf)[0]-1):
                G.add_edge(i, i+1)
            G.add_edge(0,np.shape(leaf)[0]-1)

            # Get the vertex positions
            pos = {}
            valuesX = leaf[:,0]
            valuesY = leaf[:,1]
            for i in range(np.shape(leaf)[0]):
                pos[i] = (valuesX[i],valuesY[i])

            # Select directions around the circle
            numCircleDirs = 32
            circledirs =  np.linspace(0, 2*np.pi, num=numCircleDirs, endpoint=False)
            # Choose number of thresholds for the ECC
            numThresh = 48

            # Compute the ECT of leaf p for numCircleDirs, numThresh
            ECT_preprocess = {}
            for i, angle in enumerate(circledirs):

                outECC = ECC(G, pos, theta=angle, r=r, numThresh = numThresh)

                ECT_preprocess[i] = (angle,outECC)

            # Making a matrix M[i,j]: (numThresh x numCircleDirs)
            M = np.empty([numThresh,numCircleDirs])
            for j in range(M.shape[1]):
                M[:,j] = ECT_preprocess[j][1]

            output_filedir1 = os.path.join(mypath_output+'ect32.48/train/'+ input_filedir[len(mypath):])
            output_filedir2 = os.path.join(mypath_output+'outline32.48/train/'+ input_filedir[len(mypath):-4])
            # NPY file to save ECT
            Path(os.path.dirname(output_filedir1)).mkdir(parents=True, exist_ok=True)
            np.save(output_filedir1, M) 
            #Plot the leaf and PNG file to save
            plt.figure(figsize=(2,3))
            nx.draw_networkx(G, pos, **options)
            plt.axis('off')
            plt.axis('equal')
            Path(os.path.dirname(output_filedir2)).mkdir(parents=True, exist_ok=True)
            plt.savefig(output_filedir2+'.png', dpi=16)
            plt.close()


            # Randomly rotate the leaf, compute the ECT, and save the matrix-- repeat 10 times

            print(rotation_angle)
            for idx in range(NUM_ROTATIONS):
                pos_rotate = rotate(pos, theta = rotation_angle[idx])
                
                # Compute the ECT of leaf p for numCircleDirs, numThresh
                ECT_preprocess = {}
                for i, angle in enumerate(circledirs):

                    outECC = ECC(G, pos_rotate, theta=angle, r=r, numThresh = numThresh)

                    ECT_preprocess[i] = (angle,outECC)

                # Making a matrix M[i,j]: (numThresh x numCircleDirs)
                M = np.empty([numThresh,numCircleDirs])
                for j in range(M.shape[1]):
                    M[:,j] = ECT_preprocess[j][1]

                savedir1 = os.path.join(mypath_output+'ect32.48/test'+str(idx), input_filedir[len(mypath):])     
                savedir2 = os.path.join(mypath_output+'outline32.48/test'+str(idx), input_filedir[len(mypath):-4])

                # NPY file to save
                Path(os.path.dirname(savedir1)).mkdir(parents=True, exist_ok=True)
                np.save(savedir1, M) 
                #Plot the leaf and PNG file to save
                plt.figure(figsize=(2,3))
                nx.draw_networkx(G, pos_rotate, **options)
                plt.axis('off')
                plt.axis('equal')
                Path(os.path.dirname(savedir2+'.png')).mkdir(parents=True, exist_ok=True)
                plt.savefig(savedir2, dpi=16)
                plt.close()

        class_count+=1
    class_count=0
    print('----------------\n completed subdirectory ', path ,'\n-------------', )