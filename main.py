#There are three features: Shape, Margin and Texture. As discussed in the paper(s) above.
#	For Each feature, a 64 element vector is given per sample of leaf.
#	These vectors are taken as a contigous descriptors (for shape) or histograms (for texture and margin).

import numpy as np
from sklearn.datasets import load_files
from PIL import Image
import os
import matplotlib.pyplot as plt
from custom_kNN import Custom_kNN as cNN
from DataSetSplitter import DS_Splitter
from metrics import calculateMetrics
from custom_Multi import CustomRandomForest as CRF

    
data = 'Shape'
distance = 'Chebyshev'
train_x,test_x, train_y, test_y = DS_Splitter(data)

kNN_clas = cNN(distance = distance)

Multi_clas = CRF()

kNN_clas.fit(train_x,train_y)

pred_y = kNN_clas.predict(test_x)

print("kNN custom : ",calculateMetrics(pred_y,test_y))

Multi_clas.fit(train_x,train_y)

pred_y = Multi_clas.predict(test_x)

print("Random Forest Custom : ",calculateMetrics(pred_y,test_y))