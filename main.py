import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

#There are three features: Shape, Margin and Texture. As discussed in the paper(s) above.
#	For Each feature, a 64 element vector is given per sample of leaf.
#	These vectors are taken as a contigous descriptors (for shape) or histograms (for texture and margin).

# Unione dei dati sulle varie piante in un unico dataset
def create_ds(file_path,col_prefix):
    ds = pd.read_csv(files[0], header=None)
    col_names = ['species'] + [f'{col_prefix}_{i+1}' for i in range(ds.shape[1] - 1)]
    ds.columns = col_names
    return ds
    
def append_ds(dataset1, dataset2):
    merged_dataset = dataset1
    for column in dataset2:
            if column != 'species':
                merged_dataset[column] = dataset2.loc[:,column]
            
    return merged_dataset

files = ['./Dataset/data_Sha_64.txt','./Dataset/data_Mar_64.txt','./Dataset/data_Tex_64.txt']

shape_data = create_ds(files[0],'sha_att')
margin_data = create_ds(files[1],'mar_att')
texture_data = create_ds(files[2],'tex_att')

dataset = append_ds(shape_data,margin_data)
dataset = append_ds(dataset, texture_data)

print(dataset)

#creazione del dataset grafico

directory = './Dataset/data'
for path in directory:
    image_dataset =[os.path.join(__file__)]
