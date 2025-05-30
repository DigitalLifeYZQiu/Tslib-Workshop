import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = args.learning_rate * (0.2 ** (epoch // 2))
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
#     elif args.lradj == "cosine":
#         lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    else:
        raise ValueError('Unknown lr type: {}'.format(args.lradj))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        
def TestTimeaLRAdjust(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    else:
        raise ValueError('Unknown lr type: {}'.format(args.lradj))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class TestTimeEarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_anomaly(true, preds=None, best_pred=None, border1=None, border2=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', c='dodgerblue', linewidth=0.5)
    if best_pred is not None:
        selected = np.where(best_pred)[0]
        for i in selected:
            plt.axvline(x=i, color='purple', alpha=0.1, linewidth=0.1)
    if border1 is not None and border2 is not None:
        plt.axvspan(border1, border2, alpha=0.25, color='red')
    plt.plot(true, label='GroundTruth', c='tomato', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.savefig(name, bbox_inches='tight')
    print(f'Plot saved to {name}')


def visual_anomaly_segment(true, preds=None, best_pred=None, label=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', c='dodgerblue', linewidth=0.5)
    if best_pred is not None:
        selected = np.where(best_pred)[0]
        for i in selected:
            plt.axvline(x=i, color='purple', alpha=0.1, linewidth=0.1)
    # import pdb; pdb.set_trace()
    segments = find_segments(label)
    if segments is not None:
        for (border1, border2) in segments:
            plt.axvspan(border1, border2, alpha=0.25, color='green')
    plt.plot(true, label='GroundTruth', c='tomato', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.savefig(name, bbox_inches='tight')
    print(f'Plot saved to {name}')


def visual_anomaly_segment_MS(true, preds=None, best_pred=None, label=None, name='./pic/test.pdf', count=1):
    """
    Results visualization
    """
    if len(true.shape)==1:
        S = true.shape[0]
        C=1
        true = np.expand_dims(true, axis=1)
        preds = np.expand_dims(preds, axis=1)
    elif len(true.shape)==2:
        S, C = true.shape[-2:]
    segments = find_segments(label)
    fig, axes = plt.subplots(C, 1, figsize=(8, 2 * C), sharex=True)  # create subplots
    if C == 1:
        axes = [axes]  # If there's only one variate, ensure axes is iterable
    counter = 0
    if segments is not None:
        for border1, border2 in segments:
            name_split = name.split('.')
            name_split[-2] += f'_seg_{border1}_{border2}'
            seg_name = '.'.join(name_split)
            for i in range(C):
                left = border1 - 1000 if (border1 - 1000) > 0 else 0
                right = border2 + 1000 if (border2 + 1000) < S else S - 1
                # True anomaly span
                axes[i].axvspan(border1 - left, border2 - left, alpha=0.25, color='green')
                # true value
                axes[i].plot(true[left:right, i], label='GroundTruth', c='tomato', linewidth=0.5)
                # prediction value
                if preds is not None:
                    axes[i].plot(preds[left:right, i], label='Prediction', c='dodgerblue', linewidth=0.5)
                    axes[i].set_title(f'Variate {i + 1}')
                    axes[i].grid(True)
                if best_pred is not None:
                    selected = np.where(best_pred[left:right])[0]
                    for select in selected:
                        axes[i].axvline(x=select, color='purple', alpha=0.1, linewidth=0.5)
                
                # plt.legend(loc='upper left')
            plt.savefig(seg_name, bbox_inches='tight')
            print(f'Plot saved to {seg_name}')
            counter += 1
            if counter>count:
                return


def visual_anomaly_segment_Multi(true, preds=None, best_pred=None, label=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    S, C = true.shape[-2:]
    for i in range(C):
        name_split = name.split('.')
        name_split[-2] += f'_val_{i}'
        val_name = '.'.join(name_split)
        visual_anomaly_segment(true=true[:, i], preds=preds[:, i], best_pred=best_pred, label=label, name=val_name)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def find_segments(arr):
    """
    return the indices of the segments of 1s within a segment
    """
    arr = np.array(arr)
    diff = np.diff(np.concatenate(([0], arr, [0])))  # Add boundaries
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))


def find_segment_lengths(arr):
    """
    return the length of segment of 1s in arr
    """
    lengths = []
    current_length = 0
    
    for val in arr:
        if val == 1:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
                current_length = 0
    
    if current_length > 0:  # Add the last segment if necessary
        lengths.append(current_length)
    
    return lengths