def calculate_string_set_metric(ground_truth, predicted):
    """
    Calculate precision, recall, f1 score, and accuracy for string comparison.
    
    Args:
        ground_truth (list): List of ground truth strings
        predicted (list): List of predicted strings
        
    Returns:
        dict: Dictionary containing precision, recall, f1_score, accuracy, tp, fp, fn
    """
    # Convert to sets for faster lookups
    gt_set = set(ground_truth)
    pred_set = set(predicted)
    
    # Calculate true positives (predicted strings that are in ground truth)
    tp_items = pred_set & gt_set
    tp = len(tp_items)
    
    # Calculate false positives (predicted strings not in ground truth)
    fp_items = pred_set - gt_set
    fp = len(fp_items)

    
    # Calculate false negatives (ground truth strings not in predicted)
    fn_items = gt_set - pred_set
    fn = len(fn_items)
    
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'gt_length': len(ground_truth),
        'tp_items': list(tp_items),
        'fp_items': list(fp_items),
        'fn_items': list(fn_items)
    }