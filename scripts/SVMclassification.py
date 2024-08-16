# SVM classification models 


import seaborn as sns
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
sns.set_style('whitegrid')

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default= 'leafoutline',
    help='which dataset to use for the model: mpeg7, leafgraph, or leafoutline')

args = vars(parser.parse_args())

dataset = args['dataset']

if dataset=='leafoutline':
    image_path ='../data/ALLleaves_ECT'
elif dataset=='leafgraph':
    image_path ='../data/leafgraph_ECT'
elif dataset=='mpeg7': 
    image_path ='../data/MPEG7_ECT/subset'
else:
    raise Exception(f"The dataset name {dataset=} passed is not valid.")


classes = []
for (dirpath, dirnames, filenames) in os.walk(image_path):
    classes.extend(dirnames)
    break
print(classes)


def translate_np(mat):
    # get random rotation angle, use to determine how many columns to shift
    r_angle = random.uniform(0, 2*np.pi)
    cols = np.linspace(0,2*np.pi, 32)
    col_idx = min(range(len(cols)), key=lambda i: abs(cols[i]-r_angle))
    #Translate columns of image according to the random angle
    first = mat[:,col_idx:]
    second = mat[:,0:col_idx]
    new_image = np.concatenate((first,second), axis=1)
    return new_image



all_data = []

for path, subdirs, files in os.walk(image_path):

    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if not d[0] == '.']

    for name in files:
        input_filedir = os.path.join(path, name)
        ect_normal = np.load(input_filedir)
        # Randomly rotate each leaf (done by translating matrix by some random angle rotation)
        ect_r = translate_np(ect_normal)
        leaf_vector = ect_r.flatten('F')
        splitpath = os.path.normpath(input_filedir).split(os.path.sep)
        label = list(set(splitpath).intersection(classes))[0]
        
        leaf_list = [input_filedir, label] + list(leaf_vector)

        all_data.append(leaf_list)


col_names = ['filename', 'label'] + ['t'+str(i) for i in range(32*48)]
df = pd.DataFrame(all_data, columns=col_names)#


X = df.drop(['filename', 'label'], axis=1)
y = df.label

print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")


model = LinearSVC(dual='auto', max_iter=3000)

###
## WITH CROSS VALIDATION
###

# For K-fold cross validation
k_folds = 10

scores = cross_val_score(model, X, y, cv=k_folds)

print(scores)
print("%0.2f%% accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()*100))