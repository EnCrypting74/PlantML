from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

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

def NormalizeDataset():
    
    return