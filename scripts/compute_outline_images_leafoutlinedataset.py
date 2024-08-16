import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path




mypath = '../data/ALLleaves/'
mypath_output = '../data/ALLleaves_images_small/'

options = {
    "node_size": 0,
    "node_color": "black",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 2,
    'with_labels':False,
}
#done_species = ['Alstroemeria', 'Arabidopsis', 'Apple','Cotton','Grass', 'Ivy', 'Passiflora','Potato', 'Viburnum', 'Grape','Brassica','Pepper','Tomato_entire', 'Tomato_asymmetry','C1','C2','C3','C4','C5', 'F1','F2','WA','WB','B3','B2','B1','Co1','Co3','Co4','Co5','Co6']
done_species = ['Grape','Viburnum','Cotton','Alstroemeria','Potato','Grass','Arabidopsis','Ivy','Passiflora','Apple', 'Brassica','Pepper','Tomato_entire','Tomato_asymmetry','chamber','field','wild','BILs']
# loop through file system
classes=[]
for path, subdirs, files in os.walk(mypath):
    classes.extend(subdirs)
    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if (d[0] != '.' and d not in done_species)]
    print('Computing image version of outlines in ', path, '...')
    print(len(files))
    for name in files:
        input_filedir = os.path.join(path, name)
        output_filedir = os.path.join(mypath_output+ input_filedir[len(mypath):-3])
        leaf = np.load(input_filedir)
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
        
        #Plot the leaf
        plt.figure(figsize=(2,3))
        nx.draw_networkx(G, pos, **options)
        plt.axis('off')
        plt.axis('equal')


        # PNG file to save
        Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_filedir, dpi=16)
        plt.close()

    print('----------------\n completed subdirectory ', path ,'\n-------------', )