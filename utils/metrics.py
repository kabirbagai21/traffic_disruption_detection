import numpy as np
from sklearn.metrics import balanced_accuracy_score



def compute_balanced_accuracy(preds, labels):
    """
    Compute the mean balanced accuracy over all classes.
        1) Iterate over each class present and compute the balanced accuracy over all samples using balanced_accuracy_score(...)
        2) Compute the mean score over all classes
    
    When using balanced_accuracy_score(), keep adjusted=False
        
    args:
        preds: Numpy array of shape (num_samples, num_classes) with predictions.
        labels: Numpy array of shape (num_samples, num_classes) with labels.
    returns:
        mean_balanced_accuracy: A single average balanced accuracy score computed over all classes
        accuracies: A list of balanced accuracy for each class
    """
    
    accuracies = []
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    
    num_samples, num_classes = preds.shape
    mean_balanced_accuracy = 0
    """
    for i in range(num_classes):
        pred_class = preds[:,i]
        label_class = labels[:,i]

        tp = np.sum((pred_class==1) & (label_class==1))
        fn = np.sum((pred_class==0) & (label_class==1))

        acc_class = tp/(tp+fn)/num_samples

        accuracies.append(acc_class)
        mean_balanced_accuracy = mean_balanced_accuracy + acc_class

    mean_balanced_accuracy = mean_balanced_accuracy/num_classes

    # raise Exception('utils.metrics.compute_balanced_accuracy not implemented!') # delete me
    """
    for i in range(num_classes):
        pred_class = preds[:,i]
        label_class = labels[:,i]
        bal_acc_cls = balanced_accuracy_score(pred_class, label_class)
        accuracies.append(bal_acc_cls)
        mean_balanced_accuracy += bal_acc_cls
    
    mean_balanced_accuracy = mean_balanced_accuracy/num_classes
        
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    

    
    return mean_balanced_accuracy, accuracies
