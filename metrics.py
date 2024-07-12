import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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