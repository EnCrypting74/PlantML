from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def syntheticData(texture_data):
    # Funzione per creare il record mancante della prima classe tramite media

    prime_15_righe = texture_data.head(15)
    colonne_da_calcolare = prime_15_righe.iloc[:, 1:]

    # Calcolare la media per le altre colonne
    media_colonne = colonne_da_calcolare.mean()

    # Inserire manualmente il valore della prima colonna
    valore_prima_colonna = 'Acer Campester'  # Esempio di valore inserito manualmente
    media_colonne['species'] = valore_prima_colonna

    # Riordinare le colonne per riportare 'colonna1' al primo posto
    syntheticRecord = media_colonne[['species'] + [col for col in media_colonne.index if col != 'species']].to_frame().transpose()

    texture_data = pd.concat([syntheticRecord,texture_data],axis = 0 ,ignore_index=True)

    return texture_data

def normalizeDataset(train_x,test_x):

    # Normalizza il dataset con una funzione Min-Max
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)

    return train_x, test_x

def aggregateFeatures(train_x, test_x):
    # Funzione per l'aggregazione delle features da 64 valori per feature ad 1 valore per feature
    aggreg_train_ds = pd.DataFrame()
    aggreg_test_ds = pd.DataFrame()
        
    for line in range(len(train_x)):
        # Somma i valori delle feature per il training
        texture_row = train_x.iloc[line, 0:64].sum()
        shape_row = train_x.iloc[line, 64:128].sum()
        margin_row = train_x.iloc[line, 128:192].sum()

        # Normalizza i risultati
        texture_row = texture_row / 64
        shape_row = shape_row / 64
        margin_row = margin_row / 64

        # Crea un array aggregato e poi un DataFrame
        aggreg_arr = np.array([texture_row, shape_row, margin_row])
        aggreg_row = pd.DataFrame([aggreg_arr], columns=['Texture', 'Shape', 'Margin'])
        aggreg_train_ds = pd.concat([aggreg_train_ds, aggreg_row], ignore_index=True)

    for line in range(len(test_x)):
        # Somma i valori delle feature per il test
        texture_row = test_x.iloc[line, 0:64].sum()
        shape_row = test_x.iloc[line, 64:128].sum()
        margin_row = test_x.iloc[line, 128:192].sum()

        # Normalizza i risultati
        texture_row = texture_row / 64
        shape_row = shape_row / 64
        margin_row = margin_row / 64

        # Crea un array aggregato e poi un DataFrame
        aggreg_arr = np.array([texture_row, shape_row, margin_row])
        aggreg_row = pd.DataFrame([aggreg_arr], columns=['Texture', 'Shape', 'Margin'])
        aggreg_test_ds = pd.concat([aggreg_test_ds, aggreg_row], ignore_index=True)

    return aggreg_train_ds, aggreg_test_ds