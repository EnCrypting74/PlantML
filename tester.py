
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
import numpy as np
from custom_kNN import Custom_kNN as cNN

data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
col_names = ['species'] + [f'shape_{i+1}' for i in range(data.shape[1] - 1)]
data.columns = col_names
labels = data['species']

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
data = data.drop(data.columns[0], axis = 1)

train_x, test_x, train_y, test_y = train_test_split(data, encoded_labels, random_state=0, test_size=0.25)
print('Shape training set:', train_x.shape)
print('Shape validation set:', test_x.shape)

kNN_clas = cNN()

kNN_clas.fit(train_x,train_y)

pred_y = kNN_clas.predict(test_x)

print(pred_y)