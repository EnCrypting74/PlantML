import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from DataSetSplitter import DS_Splitter

def calculateMetrics(predictions, labels):
    #definizione metriche

    # Overall accuracy
    ACC = accuracy_score(labels, predictions)
    # Recall
    REC = recall_score(labels, predictions, average='macro',zero_division=np.nan) 
    # Precision  
    PRE = precision_score(labels, predictions, average='macro',zero_division=np.nan)
    # False Positive Rate
    F1 = f1_score(labels, predictions, average='macro',zero_division=np.nan) 
    
    #AUC = roc_auc_score(labels, predictions, multi_class='ovr')  da fixare
    c_matrix= confusion_matrix(labels, predictions) #illeggibile, da fixare

    return(f'Accuracy = {ACC} \nPrecision = {PRE} \nRecall = {REC} \nF1 score = {F1}\n ')

def calc_zeros():
    # Calcola il numero di zeri nel dataset
    return

def histo(tipo):
    # Visualizza istogrammi texture e margin
    if tipo == 'Margin':
        row , _, _ , _= DS_Splitter(type = 'Margin')
        h_data = row.iloc[:,1]

        fig, ax = plt.subplots(figsize = (4,2))
        ax.hist(h_data, bins=20, color='blue', edgecolor='black')
        ax.set_title("Istogramma margini primo campione")
        ax.set_xlabel("Valore")
        ax.set_ylabel("Frequenza")

        return fig,ax


    if tipo == 'Texture':
        row , _, _ , _= DS_Splitter(type = 'Texture')
        h_data = row.iloc[:,1]

        fig, ax = plt.subplots(figsize = (4,2))
        ax.hist(h_data, bins=20, color='orange', edgecolor='black')
        ax.set_title("Istogramma texture primo campione")
        ax.set_xlabel("Valore")
        ax.set_ylabel("Frequenza")

        return fig,ax

    return
        