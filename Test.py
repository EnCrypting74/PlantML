
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pandas as pd

def syntheticData(texture_data):
    prime_15_righe = texture_data.head(15)
    colonne_da_calcolare = prime_15_righe.iloc[:, 1:]

    # Calcolare la media per le altre colonne
    media_colonne = colonne_da_calcolare.mean()

    # Inserire manualmente il valore della prima colonna
    valore_prima_colonna = 'Acer Campester'  # Esempio di valore inserito manualmente
    media_colonne['species'] = valore_prima_colonna

    # Riordinare le colonne per riportare 'colonna1' al primo posto
    syntheticRecord = media_colonne[['species'] + [col for col in media_colonne.index if col != 'species']].to_frame().transpose()

    texture_data = pd.concat([syntheticRecord,texture_data],axis = 0)
    for i in range(1,len(texture_data.index)):
        texture_data.index[i] +=1

    print(texture_data.index)
    
    return texture_data

texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header = None)

texture_col_names = ['species'] + [f'texture_{i+1}' for i in range(texture_data.shape[1] - 1)]
texture_data.columns = texture_col_names
texture_labels = texture_data['species']

shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header = None)
shape_col_names = ['species'] + [f'shape_{i+1}' for i in range(shape_data.shape[1] - 1)]
shape_data.columns = shape_col_names
texture_data = syntheticData(texture_data)

shape_data = shape_data.drop(shape_data.columns[0], axis = 1)

margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header = None)
margin_col_names = ['species'] + [f'margin_{i+1}' for i in range(margin_data.shape[1] - 1)]
margin_data.columns = margin_col_names
margin_data = margin_data.drop(margin_data.columns[0], axis = 1)

print(texture_data.columns,"\n",shape_data.columns)

partial_ds = pd.concat([texture_data,shape_data], axis = 1,names=[texture_data.columns]+[shape_data.columns])
mixed_ds = pd.concat([partial_ds,margin_data], axis = 1,names=[partial_ds.columns]+[margin_data.columns] )
mixed_labels = mixed_ds['species']
mixed_label_encoder = LabelEncoder()
mixed_encoded_labels = mixed_label_encoder.fit_transform(mixed_labels)
mixed_ds = mixed_ds.drop(mixed_ds.columns[0], axis = 1)

train_x, test_x, train_y, test_y = train_test_split(mixed_ds, mixed_encoded_labels, random_state=0, test_size=0.25)
print('Texture training set:', train_x.shape)
print('Texture validation set:', test_x.shape)

from collections import Counter

def secondo_elemento_meno_frequente(arr):
    # Conta le occorrenze di ciascun elemento nell'array
    counter = Counter(arr)
    
    # Ordina gli elementi in base al loro conteggio
    ordinati_per_frequenza = sorted(counter.items(), key=lambda x: x[1])
    
    # Verifica se ci sono almeno due elementi con frequenze diverse
    if len(ordinati_per_frequenza) < 2:
        return None, None
    
    # Trova il secondo valore meno frequente
    secondo_meno_frequente = ordinati_per_frequenza[1]
    
    return secondo_meno_frequente

elemento, conteggio = secondo_elemento_meno_frequente(mixed_encoded_labels)
print(f"Il secondo elemento meno frequente è {elemento} con {conteggio} occorrenze")
