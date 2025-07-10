from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix, accuracy_score
from torchmetrics.classification import MulticlassAveragePrecision, MultilabelAveragePrecision
import numpy as np
import torch

class Metric():
    def __init__(self, weather_types, severity_types):
        self.weather_types = weather_types
        self.severity_levels = severity_types
        
    def calculate(self, all_probs, all_labels, is_multilabel=True):
        if is_multilabel:
            preds = (all_probs > 0.5).astype(int)
            for idx, i in enumerate(preds):
                if sum(i) == 0:
                    preds[idx] = (all_probs[idx] >= max(all_probs[idx])).astype(int)
            mm = MultilabelAveragePrecision(num_labels=4, average="weighted", thresholds=None)
            map = mm(torch.Tensor(preds.astype(np.int64)), torch.tensor(all_labels.astype(np.int64)))
            report = classification_report(all_labels, preds, target_names=self.weather_types, zero_division=0, digits=4)
            conf_matrix = multilabel_confusion_matrix(all_labels, preds)
            accuracy = accuracy_score(all_labels, preds)
        else:
            preds = np.argmax(all_probs, axis=1)
            mm = MulticlassAveragePrecision(num_classes=3, average="weighted", thresholds=None)
            map = mm(torch.Tensor(all_probs), torch.tensor(all_labels))
            report = classification_report(all_labels, preds, target_names=self.severity_levels, zero_division=0, digits=4)
            conf_matrix = None
            accuracy = None
        metrics = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy' : accuracy,
            'map' : map
        }
        return metrics