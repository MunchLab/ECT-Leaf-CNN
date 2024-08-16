"""
   ROTATIONAL INVARIANCE TESTING
     Apply random rotations to a single input sample.
    Compute ECT of each randomly rotated instance.
    (-)Pass each ECT sample through trained CNN to compare outputs.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path

from ect_utils import ECC

input_filedir = '../data/ALLleaves/Cotton/2a_861b.npy'
NUM_ROTATIONS = 50

numCircleDirs = 32 # Select directions around the circle
circledirs =  np.linspace(0, 2*np.pi, num=numCircleDirs, endpoint=False)
numThresh = 48 # Choose number of thresholds for the ECC
r = 2.9092515639765497
#r = 1.22

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


""" G.add_edge(0, 5)
G.add_edge(0, 6)
G.add_edge(0, 1)
G.add_edge(1, 6)
G.add_edge(1, 8)
G.add_edge(1, 2)
G.add_edge(2, 8)
G.add_edge(2, 4)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(3, 7)
G.add_edge(5,7)
pos = nx.spring_layout(G) """

    



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

rotation_angle = np.linspace(0, 2*np.pi, num=NUM_ROTATIONS, endpoint=False)
print(rotation_angle)

for idx in range(NUM_ROTATIONS):
    output_filedir = os.path.join('../data/ALLleaves_ECT_Cotton_rotation/', 'rotation'+str(idx))
    #output_filedir = os.path.join('../data/ALLleaves_ECT_graph_rotation/', 'rotation'+str(idx))
    Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)

    pos_rotate = rotate(pos, theta = rotation_angle[idx])

    # Plot the graph
    options = {
        "node_size": 1,
        "node_color": "black",
        "edgecolors": "black",
        "linewidths": 0.2,
        "width": 0.4,
        'with_labels':False,
    }
    plt.figure(figsize=(5,5), dpi=60)
    nx.draw_networkx(G, pos_rotate, **options)
    ax = plt.gca()
    midpt = (0,0)
    circle = plt.Circle(midpt, 1.7, color='white', fill=False)
    ax.add_patch(circle)
    plt.axis("off")
    plt.axis('equal')
    plt.savefig(output_filedir+'.png', dpi=1000, bbox_inches='tight')
    #plt.savefig(output_filedir+'.png', dpi=1000)
    #plt.show()
    plt.close()


    # Compute the ECT of leaf p for numCircleDirs, numThresh
    ECT_preprocess = {}
    for i, angle in enumerate(circledirs):

        outECC = ECC(G, pos_rotate, theta=angle, r=r, numThresh = numThresh)

        ECT_preprocess[i] = (angle,outECC)

    # Making a matrix M[i,j]: (numThresh x numCircleDirs)
    M = np.empty([numThresh,numCircleDirs])
    for j in range(M.shape[1]):
        M[:,j] = ECT_preprocess[j][1]


    # NPY file to save
    np.save(output_filedir+'.npy', M) 
