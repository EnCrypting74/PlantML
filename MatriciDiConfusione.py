from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
def MatriciDiConfusione(pred_y, true_y):
    
    disp = roc_auc_score( true_y, pred_y, multi_class='ovr', average='macro')

    disp.plot()
    plt.show()
    return