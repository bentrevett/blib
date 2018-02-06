class CategoricalAccuracy:
    """
    Categorical top-k accuracy.
    Assuming integer labels with each item classified by a single class.

    Predictions is a tensor [batch_size, n_classes]
    Targets is a tensor sized [batch_size] NOT [batch_size, 1]
    """
    def __init__(self, top_k = 1):
        
        self._top_k = top_k
    
    def __call__(self, predictions, targets):
        
        assert len(targets.shape) == 1, 'Targets must be [bsz], not [bsz,1]'

        #top k indexes of predictions (or fewer if less than k of them)
        top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        #this is shape (batch_size, ..., top_k)
        correct = top_k.eq(targets.long().unsqueeze(-1)).float()

        correct_count = correct.sum()
        total_count = targets.numel() #numel = number of elements

        print(f'correct_count: {correct_count}')
        print(f'total_count: {total_count}')

        acc = correct_count / total_count

        assert acc <= 1.0, 'Accuracy larger than 1'
        assert acc >= 0, 'Accuracy less than 0'

        return acc

class F1Score:
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label):
        
        self._positive_label = positive_label
        
    def __call__(self, predictions, targets):
        
        assert len(targets.shape) == 1, 'Targets must be [bsz], not [bsz,1]'
        
        positive_label_mask = targets.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask
        
        argmax_predictions = predictions.topk(1, -1)[1].float().squeeze(-1)
        
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions != self._positive_label).float() * negative_label_mask
        true_negatives = (correct_null_predictions.float()).sum()
        
        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions == self._positive_label).float() * positive_label_mask
        true_positives = (correct_non_null_predictions).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions != self._positive_label).float() * positive_label_mask
        false_negatives = (incorrect_null_predictions).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions == self._positive_label).float() * negative_label_mask
        false_positives = (incorrect_non_null_predictions).sum()
        
        print(f'tp {true_positives}, tn {true_negatives}, fp {false_positives}, fn {false_negatives}')
        
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_score = 2. * ((precision * recall) / (precision + recall + 1e-13))
        
        return precision, recall, f1_score

    
    
        
    
    
    
    
    