from collections import defaultdict

import numpy as np
import torch


# +
def micro_acc(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    acc = (y_pred == y_true).mean()
    return acc

def macro_acc(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    labels = np.unique(y_true)
    acc_arr = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        mask_lab = y_true == lab
        acc_lab = (y_pred[mask_lab] == y_true[mask_lab]).mean()
        #print(acc_lab)
        #print(acc_arr)
        acc_arr[i] = acc_lab
    acc = acc_arr.sum()/len(labels)
    return acc


# -

def calculate_accuracy(y_pred, y_true):
    """
    Calculates the accuracy of the predicted labels.

    Args:
        y_pred (torch.Tensor): predicted labels
        y_true (torch.Tensor): true labels

    Returns:
        float: accuracy score
    """
    # Get the predicted class by finding the index of the maximum value along axis 1
    y_pred = torch.argmax(y_pred, axis=1)


    # Calculate the number of correctly classified examples
    correct = (y_pred == y_true).sum().item()

    # Calculate the total number of examples
    total = len(y_true)

    # Calculate the accuracy
    acc = correct / total

    return acc

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
    def return_averages(self,metric_name):
        metric = self.metrics[metric_name]
        avg=metric["avg"]
        return avg


class History:
    def __init__(self):

        self.train_loss = {"value":[],"epoch":0}
        self.train_accuracy =  {"value":[],"epoch":0}
        self.validation_loss =  {"value":[],"epoch":0}
        self.validation_accuracy =  {"value":[],"epoch":0}
        self.lr =  {"value":[],"epoch":0}
        #self.reg = {"value":[],"epoch":0}

        self.best_epoch = 0
        self.best_val = 1000# using accuracy

    def update(self,mode,loss,accuracy):

        if mode == "train":

            self.train_loss["value"].append(loss)
            self.train_loss["epoch"]+=1
            self.train_accuracy["value"].append(accuracy)
            self.train_accuracy["epoch"]+=1

        elif mode == "validation":

            self.validation_loss["value"].append(loss)
            self.validation_loss["epoch"]+=1
            self.validation_accuracy["value"].append(accuracy)
            self.validation_accuracy["epoch"]+=1

    def update_lr(self,lr):

        self.lr["value"].append(lr)
        self.lr["epoch"]+=1

    # def update_reg(self,reg):

    #     self.reg["value"].append(reg)
    #     self.reg["epoch"]+=1
