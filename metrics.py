import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from DataSetSplitter import DS_Splitter
from itertools import cycle
from collections import Counter

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
    
    c_matrix= confusion_matrix(labels, predictions) #illeggibile, da fixare
    
    return(f'Accuracy = {ACC} \nPrecision = {PRE} \nRecall = {REC} \nF1 score = {F1}\n ')

def show_auc(labels, predictions):
       
    # Binarizzazione delle etichette
    y_true_bin = label_binarize(labels, classes=np.arange(100))
    y_pred_bin = label_binarize(predictions, classes=np.arange(100))

    # Calcola le curve ROC e AUC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(100):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_pred_bin[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcola la curva ROC media micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot della curva ROC per ciascuna classe
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    # Plot delle curve ROC per alcune classi (per non sovraccaricare il grafico)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green','purple'])
    for i, color in zip(range(10), colors):  # Mostra solo 10 classi
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for 15 classes')
    plt.legend(loc="lower right")

    return fig

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


def find_outliers():
    # Funzione per cercare outlier outlier nel dataset
    # Creiamo un dizionario per conservare gli outlier per ogni colonna
    data = DS_Splitter(split='F')
    outliers_dict = {}
    count = Counter()

    for column in data.select_dtypes(include=[np.number]).columns:
        # Calcolo dei quartili
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Limiti per gli outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identificazione degli outlier
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_dict[column] = len(outliers)

    return outliers_dict


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
        