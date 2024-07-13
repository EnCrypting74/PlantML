import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from DataSetSplitter import DS_Splitter

def calculateMetrics(predictions, labels):
    #definizione metriche

    # Overall accuracy
    ACC = accuracy_score(labels, predictions)
    # Recall
    REC = recall_score(labels, predictions, average='macro',zero_division=0) 
    # Precision  
    PRE = precision_score(labels, predictions, average='macro',zero_division=0)
    # False Positive Rate
    F1 = f1_score(labels, predictions, average='macro',zero_division=0) 
    
    #AUC = roc_auc_score(labels, predictions, multi_class='ovr')  da fixare
    c_matrix= confusion_matrix(labels, predictions) #illeggibile, da fixare

    return(f'Accuracy = {ACC} \nPrecision = {PRE} \nRecall = {REC} \nF1 score = {F1}\n ')

def calc_zeros():
    # Calcola il numero di zeri nel dataset
    texture_data = pd.read_csv("Dataset/data_Tex_64.txt", header=None)
    shape_data = pd.read_csv("Dataset/data_Sha_64.txt", header=None)
    margin_data = pd.read_csv("Dataset/data_Mar_64.txt", header=None)

    zero_counts_texture = (texture_data == 0).sum().sum()
    zero_counts_shape = (shape_data == 0).sum().sum()
    zero_counts_margin = (margin_data == 0).sum().sum()

    total_zeros = zero_counts_texture + zero_counts_shape + zero_counts_margin

    return total_zeros

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
        