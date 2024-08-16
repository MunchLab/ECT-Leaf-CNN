# ECT-Leaf-CNN

# Convolutional Neural Networks for the Euler Characteristic Transform with Applications to Leaf Shape Data

**Author**: Sarah McGuire

**Date**: August 2024

## Description
The ECT is a simple to define and simple to compute topological representation which descriptively represents the topological shape of data. 
In contrast to alternative directional transform options defined in the literature, the ECT is simpler to compute, as well as more amenable to machine learning input requirements in a format well-suited for machine learning tasks. 
In this work, we propose to apply a particular choice of CNN architecture for classification of directional transform data, leveraging the inherent structure of the data on a cylinder. 

Using our proposed ECT-CNN pipeline, we apply the method for classification tasks of multiple leaf shape datasets.
Measuring leaf shape is paramount to discovering and understanding phylogenetic and evolutionary relationships of plants.
Traditional methods, however, often rely on direct measurements and limited statistical methods to quantify differences in leaf shape.
In this example application, we harness the effectiveness of ECT representations and the power of convolutional neural network models to quantify the naturally occurring widespread variation in leaf morphology.


Here we present `ECT-Leaf-CNN`, a repository of python code used to classify ECT summaries of leaf shape, as described in Chapters 4 and 5 of [this dissertation](https://www.proquest.com/openview/b5047898828a759dba5de90c460bde39/1?pq-origsite=gscholar&cbl=18750&diss=y).
Note that the tutorial example contained in `leaf-example-tutorial` utilizes the [ect](https://munchlab.github.io/ect/index.html#) python package for all ECT computation, which can be installed using `pip install ect`. However, the experiments and code in `scripts` are use a local version of code for ect computation (in `ect_sarah.py`).    

## To install

The code must be installed from source.

* download the code with:
```shell
git clone https://github.com/MunchLab/ECT-Leaf-CNN.git
```
* move to the directory:
```shell
cd ECT-Leaf-CNN
```


## Contents

- `cnn`: python files for the CNN model.
    - `dataloaders.py`: 
    - `models.py`:
    - `utils.py`:
- `leaf-example-tutorial`:
    - `Tutorial-ECT_for_example_dataset.ipynb`: (Jupyter notebook) Tutorial showing how to load in leaf shape data from `example_data` and train a CNN model for binary classification.
    - `example_data`
        - `outline_input`: directory contains the input leaf shape data in the form of npy files. Each file contains an ordered list of (x,y)-coordinates outlining the leaf shape.
        - `ect_output`: directory where the output computed ECT is written for each sample in `outline_input`.
        
    - `outputs`: Directory where outputs from the example jupyter notebooks are written.

- `scripts`: Python scripts to perform all of the data laoding and classification. _These are not cleaned up and organized_.


