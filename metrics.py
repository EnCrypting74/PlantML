import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from DataSetSplitter import DS_Splitter
from itertools import cycle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

def calc_nan():
    # Calcola il numero di zeri nel dataset
    data = DS_Splitter(split='F')
    data = data.drop(data.columns[0], axis = 1)

    nan_count = data.isna().sum().sum()

    return nan_count

def calc_zeros():
    # Calcola il numero di zeri nel dataset
    data = DS_Splitter(split='F')
    data = data.drop(data.columns[0], axis = 1)

    zero_dict = {}
    zero_counts = 0

    for column in data:
        zero_dict[column] = (data[column] == 0).sum()
        zero_counts += (data[column] == 0).sum()

    return zero_counts, zero_dict


def find_outliers():
    # Funzione per cercare outlier outlier nel dataset
    # Creiamo un dizionario per conservare gli outlier per ogni colonna
    data = DS_Splitter(split='F')
    outliers_dict = {}
    num_outliers = 0
    data = data.drop(data.columns[0], axis = 1)
    for column in data.columns:
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
        num_outliers += len(outliers)

    return num_outliers, outliers_dict


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

def scatterPlot(X, clusters):
    # Riduzione della dimensionalità con PCA per la visualizzazione
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_r[:, 0], X_r[:, 1], c=clusters, cmap='viridis', s=5)
    
    # Aggiunta di una legenda
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    ax.set_title('Scatter Plot of Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    return fig