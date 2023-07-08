# placeholder for core result types
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def calc_accuracy(pred_vals:np.array, real_vals:np.array)->dict:
    
    f1 = f1_score(real_vals, pred_vals,zero_division=0)
    pr = precision_score(real_vals, pred_vals,zero_division=0)
    re = recall_score(real_vals, pred_vals,zero_division=0)
    
    accuracy = {}
    accuracy["f1"] = f1
    accuracy["prec"] = pr
    accuracy["recall"] = re
    
    return(accuracy)