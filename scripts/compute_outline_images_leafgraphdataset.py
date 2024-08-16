import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import math

from ect_utils import ECC, bounding_box

filename = "../data/0.procrustes_landmarks.txt"
df = pd.read_csv(filename, sep='	').transpose()
labels = df[:][:5].transpose()

# Compute bounding box
## COLLECT ALL POINTS 
filename = "../data/0.procrustes_landmarks.txt"
df = pd.read_csv(filename, sep='	').transpose()
labels = df[:][:5].transpose()
global_pos_list =[]
for p in range(len(df.columns)):
    valuesX = df[p].iloc[5::2]
    valuesY = df[p].iloc[6::2]
    for j in range(15):
        global_pos_list.append((valuesX.T[j], valuesY.T[j]))

print('Loaded all data points, computing global bounding box...')
x_box,y_box = zip(*bounding_box(global_pos_list)) # build a bounding box
# bounding box size (use to get a radius for the bounding circle)
print('Completed bounding box, starting ect computations...')
dist = math.dist((x_box[0],y_box[0]),(x_box[1],y_box[1]))
r = dist/2
print('r=',r)

G = nx.Graph()
G.add_edge(0, 5)
G.add_edge(0, 6)
G.add_edge(0, 1)
G.add_edge(1, 6)
G.add_edge(1, 8)
G.add_edge(1, 2)
G.add_edge(2, 8)
G.add_edge(2, 10)
G.add_edge(2, 3)
G.add_edge(3, 10)
G.add_edge(3, 12)
G.add_edge(3, 4)
G.add_edge(4, 12)
G.add_edge(4, 14)
G.add_edge(4, 5)
G.add_edge(5, 14)
G.add_edge(6, 7)
G.add_edge(7, 8)
G.add_edge(8, 9)
G.add_edge(9, 10)
G.add_edge(10, 11)
G.add_edge(11, 12)
G.add_edge(12, 13)
G.add_edge(13, 14)

options = {
    "node_size": 0,
    "node_color": "black",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 2,
    'with_labels':False,
}

dataset = []
for p in range(len(df.columns)):
    
    # Get the vertex positions
    pos = {}
    valuesX = df[p].iloc[5::2]
    valuesY = df[p].iloc[6::2]
    for i in range(15):
        pos[i] = (valuesX[i],valuesY[i])
        
    # Select directions around the circle
    numCircleDirs = 32
    circledirs =  np.linspace(0, 2*np.pi, num=numCircleDirs, endpoint=False)
    
    # Choose number of thresholds for the ECC
    numThresh = 48
    
    # Compute the ECT of sample p for numCircleDirs, numThresh
    ECT_preprocess = {}
    for i, angle in enumerate(circledirs):

        outECC = ECC(G, pos, theta=angle, r=r, numThresh = numThresh)

        ECT_preprocess[i] = (angle,outECC)


    # Making a matrix M[i,j]: (numThresh x numCircleDirs)
    M = np.empty([numThresh,numCircleDirs])
    for j in range(M.shape[1]):
        M[:,j] = ECT_preprocess[j][1]
        
    output_filedir = '../data/leafgraph_ECT/'+str(df[p].species)+'/leaf_'+str(p)+'.npy'
    # NPY file to save
    Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)
    np.save(output_filedir, M)

    ## NOW save small image versions of each leaf graph
    #Plot the leaf
    plt.figure(figsize=(2,3))
    nx.draw_networkx(G, pos, **options)
    plt.axis('off')
    plt.axis('equal')
    
    # PNG file to save
    output_filedir = '../data/leafgraph_images_small/'+str(df[p].species)+'/leaf_'+str(p)+'.png'
    Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filedir, dpi=16)
    plt.close()
