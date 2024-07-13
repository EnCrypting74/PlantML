
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
from Preprocessing import syntheticData

def DS_Splitter(type = 'Total', split = 'T'):
    if type == 'Shape':

        # Creazione della porzione di dataset con i vettori di feature di shape
        shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
        shape_col_names = ['species'] + [f'shape_{i+1}' for i in range(shape_data.shape[1] - 1)]
        shape_data.columns = shape_col_names
        # Ordinamento alfabetico del dataset
        shape_data = shape_data.sort_values(by='species').reset_index(drop=True)
        shape_labels = shape_data['species']
        
        # Encoding labels 
        shape_label_encoder = LabelEncoder()
        shape_labels = shape_label_encoder.fit_transform(shape_labels)
        shape_data = shape_data.drop(shape_data.columns[0], axis = 1)

        # Split del dataset in train e test
        train_x, test_x, train_y, test_y = train_test_split(shape_data, shape_labels, random_state=0, test_size=0.25)

        return train_x, test_x, train_y, test_y

    elif type == 'Margin':

        # Creazione della porzione di dataset con i vettori di feature di margine
        margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header = None)
        margin_col_names = ['species'] + [f'margin_{i+1}' for i in range(margin_data.shape[1] - 1)]
        margin_data.columns = margin_col_names
        margin_labels = margin_data['species']

        # Encoding labels 
        margin_label_encoder = LabelEncoder()
        margin_labels = margin_label_encoder.fit_transform(margin_labels)
        margin_data = margin_data.drop(margin_data.columns[0], axis = 1)

        # Split del dataset in train e test
        train_x, test_x, train_y, test_y = train_test_split(margin_data, margin_labels, random_state=0, test_size=0.25)

        return train_x, test_x, train_y, test_y

    elif type == 'Texture':
        
        # Creazione della porzione di dataset con i vettori di feature di texture
        texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header = None)
        texture_col_names = ['species'] + [f'texture_{i+1}' for i in range(texture_data.shape[1] - 1)]
        texture_data.columns = texture_col_names
        texture_labels = texture_data['species']

        # Encoding labels 
        texture_label_encoder = LabelEncoder()
        texture_labels = texture_label_encoder.fit_transform(texture_labels)
        texture_data = texture_data.drop(texture_data.columns[0], axis = 1)

        # Split del dataset in train e test
        train_x, test_x, train_y, test_y = train_test_split(texture_data, texture_labels, random_state=0, test_size=0.25)
        return train_x, test_x, train_y, test_y
    
    elif type == 'Total':
        
        # Creazione del dataset completo con i vettori di feature di shape, margine e texture
        # Texture vectors
        texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header = None)
        texture_col_names = ['species'] + [f'texture_{i+1}' for i in range(texture_data.shape[1] - 1)]
        texture_data.columns = texture_col_names

        # Aggiunta del record di texture mancante, calcolato con la media dei valori della classe
        texture_data = syntheticData(texture_data)
        texture_labels = texture_data['species']

        # Shape vectors
        shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
        shape_col_names = ['species'] + [f'shape_{i+1}' for i in range(shape_data.shape[1] - 1)]
        shape_data.columns = shape_col_names
        # Ordinamento alfabetico delle feature
        shape_data = shape_data.sort_values(by='species').reset_index(drop=True)
        shape_data = shape_data.drop(shape_data.columns[0], axis = 1)

        # Margin vectors
        margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header = None)
        margin_col_names = ['species'] + [f'margin_{i+1}' for i in range(margin_data.shape[1] - 1)]
        margin_data.columns = margin_col_names
        margin_data = margin_data.drop(margin_data.columns[0], axis = 1)

        # Concatenazione dei vettori di features per le righe
        partial_ds = pd.concat([texture_data,shape_data], axis = 1,names=[texture_data.columns]+[shape_data.columns])
        mixed_ds = pd.concat([partial_ds,margin_data], axis = 1,names=[partial_ds.columns]+[margin_data.columns] )
        mixed_labels = mixed_ds['species']

        # Encoding delle labels
        mixed_label_encoder = LabelEncoder()
        mixed_encoded_labels = mixed_label_encoder.fit_transform(mixed_labels)

        # Nel caso ci serva il dataset non splittato
        if split == 'F':
            return mixed_ds
        
        mixed_ds = mixed_ds.drop(mixed_ds.columns[0], axis = 1)
        
        # Se non selezioniamo di aggiungere il record sintetico per la prima classe
        # droppiamo la riga corrispondente al record con shape mancante
        mixed_ds = mixed_ds.dropna()
        
        # Split del dataset in train e test
        train_x, test_x, train_y, test_y = train_test_split(mixed_ds, mixed_encoded_labels, random_state=0, test_size=0.25)
        return train_x, test_x, train_y, test_y

    else:

        raise TypeError("Tipo non supportato")
