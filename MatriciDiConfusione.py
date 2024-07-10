from sklearn.metrics import ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
def MatriciDiConfusione(pred_y, true_y):
    labels = [i for i in range(100)]
    disp = ConfusionMatrixDisplay.from_predictions( true_y, pred_y, display_labels= labels, normalize = 'true')

    disp.plot()
    plt.show()
    return