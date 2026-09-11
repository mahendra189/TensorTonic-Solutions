import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    # # Write code here
    tp = {}
    fp = {}
    fn = {}
    support = {}
    precision = {}
    recall = {}
    f1 = {}
    for cls in classes:
        tp[cls] = 0
        fp[cls] = 0
        fn[cls] = 0
        for a,y in zip(y_true,y_pred):
            if a == y == cls:
                tp[cls] += 1
            elif a != cls and y == cls:
                fp[cls] += 1
            elif a == cls and y != cls:
                fn[cls] += 1
        precision[cls] = (
            tp[cls] / (tp[cls] + fp[cls])
            if tp[cls] + fp[cls] > 0 else 0
        )
        
        recall[cls] = (
            tp[cls] / (tp[cls] + fn[cls])
            if tp[cls] + fn[cls] > 0 else 0
        )
        
        f1[cls] = (
            2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
            if precision[cls] + recall[cls] > 0 else 0
        )
        support[cls] = np.sum(y_true == cls)
    # Accuracy
    
    accuracy = np.mean(y_true == y_pred)
    if average == "binary":
        cls = pos_label
    
        avg_precision = precision.get(cls, 0)
        avg_recall = recall.get(cls, 0)
        avg_f1 = f1.get(cls, 0)
    elif average == "micro":
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
    
        avg_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        avg_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    
        avg_f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if avg_precision + avg_recall > 0 else 0
        )
    elif average == "macro":
        avg_precision = np.mean(list(precision.values()))
        avg_recall = np.mean(list(recall.values()))
        avg_f1 = np.mean(list(f1.values()))
    elif average == "weighted":
        total_support = sum(support.values())
    
        avg_precision = sum(
            precision[cls] * support[cls]
            for cls in classes
        ) / total_support
    
        avg_recall = sum(
            recall[cls] * support[cls]
            for cls in classes
        ) / total_support
    
        avg_f1 = sum(
            f1[cls] * support[cls]
            for cls in classes
        ) / total_support
    
    else:
        raise ValueError(
            "average must be 'micro', 'macro', or 'weighted'"
        )

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(avg_precision, 6),
        "recall": round(avg_recall, 6),
        "f1": round(avg_f1, 6),
    }
    
        
    
    