def calculateMetrics(predictions, labels):
    #definizione metriche
    # TP = 0
    # TN = 0
    # FP = 0
    # FN = 0
    # FPR = 0
    # TPR = 0
    # FNR = 0
    # TNR =0
    Right = 0
    Wrong = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            Right +=1
        elif labels[i] != predictions[i]:
            Wrong += 1
        
    # TPR = TP / (TP + FP)
    # FPR = FP / (TN + FP)
    # FNR = FN / (FN + TP)
    # TNR = TN / (FN + TN)

    return("Accuracy = ",Right/len(labels))