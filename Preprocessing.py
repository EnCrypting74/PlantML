from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize
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

def normalizeDataset(train_x,test_x,n_type = "Standard"):
    if n_type == "Standard":
        # scala il dataset con una standardizzazione standard
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.fit_transform(test_x)
        
        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)

        return train_x, test_x
    
    elif n_type == "MinMax":
        # scala il dataset con una standardizzazione Min-Max
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.fit_transform(test_x)
        
        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)
        
        return train_x, test_x
        
    elif n_type == "Robust":
        # Rimuove la mediana e scala il dataset in base all'IQR
        scaler = RobustScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.fit_transform(test_x)
        
        train_x = pd.DataFrame(train_x)
        test_x = pd.DataFrame(test_x)
        
        return train_x, test_x
    
    elif n_type == "Normalization":
        # Normalizza il dataset
        
        train_x = pd.DataFrame(normalize(train_x, norm = 'l1'))
        test_x = pd.DataFrame(normalize(test_x, norm = 'l1'))
        
        return train_x, test_x
    
def aggregateFeatures(train_x, test_x, mode = 16):
    # Funzione per l'aggregazione delle features 
    aggreg_train_ds = pd.DataFrame()
    aggreg_test_ds = pd.DataFrame()

    if mode == 32:
        for line in range(len(train_x)):
            # Calcola la media dei valori delle feature per il training
            texture_row1 = train_x.iloc[line, 0:31].sum()/ 32
            texture_row2 = train_x.iloc[line, 32:63].sum()/ 32
            shape_row1 = train_x.iloc[line, 64:95].sum()/ 32
            shape_row2 = train_x.iloc[line, 96:127].sum()/ 32
            margin_row1 = train_x.iloc[line, 128:159].sum()/ 32
            margin_row2 = train_x.iloc[line, 160:192].sum()/ 32

            # Calcola la media dei valori delle feature per il training
            aggreg_arr = np.array([texture_row1,texture_row2, shape_row1,shape_row2, margin_row1, margin_row2])
            aggreg_row = pd.DataFrame([aggreg_arr], columns=['texture_1','texture_2','shape_1','shape_2','margin_1','margin_2'])
            aggreg_train_ds = pd.concat([aggreg_train_ds, aggreg_row], ignore_index=True)

        for line in range(len(test_x)):
            # Somma i valori delle feature per il training
            texture_row1 = test_x.iloc[line, 0:31].sum()/ 32
            texture_row2 = test_x.iloc[line, 32:63].sum()/ 32
            shape_row1 = test_x.iloc[line, 64:95].sum()/ 32
            shape_row2 = test_x.iloc[line, 96:127].sum()/ 32
            margin_row1 = test_x.iloc[line, 128:159].sum()/ 32
            margin_row2 = test_x.iloc[line, 160:192].sum()/ 32

            # Crea un array aggregato e poi un DataFrame
            aggreg_arr = np.array([texture_row1,texture_row2, shape_row1,shape_row2, margin_row1, margin_row2])
            aggreg_row = pd.DataFrame([aggreg_arr], columns=['texture_1','texture_2','shape_1','shape_2','margin_1','margin_2'])
            aggreg_test_ds = pd.concat([aggreg_test_ds, aggreg_row], ignore_index=True)
    elif mode == 16:
        for line in range(len(train_x)):
            # Calcola la media dei valori delle feature per il training
            texture_row1 = train_x.iloc[line, 0:15].sum()/ 16
            texture_row2 = train_x.iloc[line, 16:31].sum()/ 16
            texture_row3 = train_x.iloc[line, 32:47].sum()/ 16
            texture_row4 = train_x.iloc[line, 47:63].sum()/ 16
            shape_row1 = train_x.iloc[line, 64:79].sum()/ 16
            shape_row2 = train_x.iloc[line, 80:95].sum()/ 16
            shape_row3 = train_x.iloc[line, 96:110].sum()/ 16
            shape_row4 = train_x.iloc[line, 111:127].sum()/ 16
            margin_row1 = train_x.iloc[line, 128:143].sum()/ 16
            margin_row2 = train_x.iloc[line, 144:159].sum()/ 16
            margin_row3 = train_x.iloc[line, 160:175].sum()/ 16
            margin_row4 = train_x.iloc[line, 176:192].sum()/ 16

            # Crea un array aggregato e poi un DataFrame
            aggreg_arr = np.array([texture_row1,texture_row2,texture_row3,texture_row4,shape_row1,shape_row2,shape_row3,shape_row4,margin_row1,margin_row2,margin_row3, margin_row4])
            aggreg_row = pd.DataFrame([aggreg_arr], columns=['texture_1','texture_2','texture_3','texture_4','shape_1','shape_2','shape_3','shape_4','margin_1','margin_2','margin_3','margin_4'])
            aggreg_train_ds = pd.concat([aggreg_train_ds, aggreg_row], ignore_index=True)

        for line in range(len(test_x)):
            # Calcola la media dei valori delle feature per il training 
            texture_row1 = test_x.iloc[line, 0:15].sum()/ 16
            texture_row2 = test_x.iloc[line, 16:31].sum()/ 16
            texture_row3 = test_x.iloc[line, 32:47].sum()/ 16
            texture_row4 = test_x.iloc[line, 47:63].sum()/ 16
            shape_row1 = test_x.iloc[line, 64:79].sum()/ 16
            shape_row2 = test_x.iloc[line, 80:95].sum()/ 16
            shape_row3 = test_x.iloc[line, 96:110].sum()/ 16
            shape_row4 = test_x.iloc[line, 111:127].sum()/ 16
            margin_row1 = test_x.iloc[line, 128:143].sum()/ 16
            margin_row2 = test_x.iloc[line, 144:159].sum()/ 16
            margin_row3 = test_x.iloc[line, 160:175].sum()/ 16
            margin_row4 = test_x.iloc[line, 176:192].sum()/ 16

            # Crea un array aggregato e poi un DataFrame
            aggreg_arr = np.array([texture_row1,texture_row2,texture_row3,texture_row4,shape_row1,shape_row2,shape_row3,shape_row4,margin_row1,margin_row2,margin_row3, margin_row4])
            aggreg_row = pd.DataFrame([aggreg_arr], columns=['texture_1','texture_2','texture_3','texture_4','shape_1','shape_2','shape_3','shape_4','margin_1','margin_2','margin_3','margin_4'])
            aggreg_test_ds = pd.concat([aggreg_test_ds, aggreg_row], ignore_index=True)
    return aggreg_train_ds, aggreg_test_ds