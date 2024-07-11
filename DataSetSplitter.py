
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
from Preprocessing import syntheticData

def DS_Splitter(type):
    if type == 'Shape':

        shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
        shape_col_names = ['species'] + [f'shape_{i+1}' for i in range(shape_data.shape[1] - 1)]
        shape_data.columns = shape_col_names
        shape_labels = shape_data['species']

        shape_label_encoder = LabelEncoder()
        shape_encoded_labels = shape_label_encoder.fit_transform(shape_labels)
        shape_data = shape_data.drop(shape_data.columns[0], axis = 1)

        train_x, test_x, train_y, test_y = train_test_split(shape_data, shape_encoded_labels, random_state=0, test_size=0.25)
        print('Shape training set:', train_x.shape)
        print('Shape validation set:', test_x.shape)

        return train_x,test_x, train_y, test_y

    elif type == 'Margin':

        margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header = None)
        margin_col_names = ['species'] + [f'margin_{i+1}' for i in range(margin_data.shape[1] - 1)]
        margin_data.columns = margin_col_names
        margin_labels = margin_data['species']

        margin_label_encoder = LabelEncoder()
        margin_encoded_labels = margin_label_encoder.fit_transform(margin_labels)
        margin_data = margin_data.drop(margin_data.columns[0], axis = 1)

        train_x, test_x, train_y, test_y = train_test_split(margin_data, margin_encoded_labels, random_state=0, test_size=0.25)
        print('Margin training set:', train_x.shape)
        print('Margin validation set:', test_x.shape)

        return train_x,test_x, train_y, test_y

    elif type == 'Texture':
        texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header = None)
        texture_col_names = ['species'] + [f'texture_{i+1}' for i in range(texture_data.shape[1] - 1)]
        texture_data.columns = texture_col_names
        texture_labels = texture_data['species']

        texture_label_encoder = LabelEncoder()
        texture_encoded_labels = texture_label_encoder.fit_transform(texture_labels)
        texture_data = texture_data.drop(texture_data.columns[0], axis = 1)

        train_x, test_x, train_y, test_y = train_test_split(texture_data, texture_encoded_labels, random_state=0, test_size=0.25)
        print('Texture training set:', train_x.shape)
        print('Texture validation set:', test_x.shape)
        return train_x,test_x, train_y, test_y
    
    elif type == 'Mixed':
        texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header = None)
        texture_col_names = ['species'] + [f'texture_{i+1}' for i in range(texture_data.shape[1] - 1)]
        texture_data.columns = texture_col_names
        texture_data = syntheticData(texture_data)
        texture_labels = texture_data['species']

        shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
        shape_col_names = ['species'] + [f'shape_{i+1}' for i in range(shape_data.shape[1] - 1)]
        shape_data.columns = shape_col_names
        shape_data = shape_data.drop(shape_data.columns[0], axis = 1)

        margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header = None)
        margin_col_names = ['species'] + [f'margin_{i+1}' for i in range(margin_data.shape[1] - 1)]
        margin_data.columns = margin_col_names
        margin_data = margin_data.drop(margin_data.columns[0], axis = 1)

        partial_ds = pd.concat([texture_data,shape_data], axis = 1,names=[texture_data.columns]+[shape_data.columns])
        mixed_ds = pd.concat([partial_ds,margin_data], axis = 1,names=[partial_ds.columns]+[margin_data.columns] )
        mixed_labels = mixed_ds['species']
        mixed_ds = mixed_ds.drop(mixed_ds.columns[0], axis = 1)

        mixed_label_encoder = LabelEncoder()
        mixed_encoded_labels = mixed_label_encoder.fit_transform(mixed_labels)
        mixed_ds = mixed_ds.drop(mixed_ds.columns[0], axis = 1)
        
        train_x, test_x, train_y, test_y = train_test_split(mixed_ds, mixed_encoded_labels, random_state=0, test_size=0.25)
        print('Texture training set:', train_x.shape)
        print('Texture validation set:', test_x.shape)
        return train_x,test_x, train_y, test_y

    else:

        raise TypeError("Tipo non supportato")
