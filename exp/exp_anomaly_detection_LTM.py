from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from utils.tools import adjust_learning_rate, visual, visual_anomaly, adjustment, find_segment_lengths, find_segments, \
    visual_anomaly_segment, visual_anomaly_segment_MS, visual_anomaly_segment_Multi
from utils.tsne import visualization,visualization_PCA
from utils.anomaly_detection_metrics import adjbestf1,f1_score
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import collections
from collections import Counter
import csv

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_LTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_LTM, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if self.args.ad_mask_type == 'random':
                    # random mask
                    B, T, N = batch_x.shape
                    """
					B = batch size
					T = seq len
					N = number of features
					"""
                    assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                    mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))
                    mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None)
                    pred = outputs.detach().cpu()
                    f_dim = -1 if self.args.features == 'MS' else 0
                    true = batch_x[:, :, f_dim:].detach().cpu()
                    mask = mask[:, :, f_dim:].detach().cpu()
                    # import pdb; pdb.set_trace()
                    loss = criterion(pred[mask == 0], true[mask == 0])
                else:
                    outputs = self.model(batch_x, None, None, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()

                    loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                
                if self.args.ad_mask_type == 'random':
                    # random mask
                    B, T, N = batch_x.shape
                    """
					B = batch size
					T = seq len
					N = number of features
					"""
                    assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                    mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    # import pdb; pdb.set_trace()
                    mask = mask.view(mask.size(0), -1, mask.size(-1))
                    mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None)
                    pred = outputs
                    f_dim = -1 if self.args.features == 'MS' else 0
                    true = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]
                    # import pdb; pdb.set_trace()
                    loss = criterion(pred[mask == 0], true[mask == 0])
                else:
                    outputs = self.model(batch_x, None, None, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.test(setting)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test and self.args.ckpt_path is None:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        input_list = []
        output_list = []
        test_labels = []
        score_list = []
        embedding_list = []
        feature_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruct the input sequence and record the loss as a sorted list
                if self.args.ad_mask_type == 'random':
                    # random mask
                    B, T, N = batch_x.shape
                    """
					B = batch size
					T = seq len
					N = number of features
					"""
                    assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                    mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))
                    mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None)
                    
                else:
                    outputs = self.model(batch_x, None, None, None)
                    # embeds = self.model.getEmbedding(batch_x)
                    # features = self.model.getFeature(batch_x)
                    
                input_list.append(batch_x.detach().cpu().numpy())
                output_list.append(outputs.detach().cpu().numpy())
                test_labels.append(batch_y.reshape(-1).detach().cpu().numpy())
                # embedding_list.append(embeds.detach().cpu().numpy())
                # feature_list.append(features.detach().cpu().numpy())
                
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score_list.append(score.detach().cpu().numpy())
                
        
        # * Evaluate metrics
        B,S,C = input_list[0].shape
        test_labels = np.concatenate(test_labels, axis=0).reshape((-1))
        input = np.concatenate(input_list, axis=0).reshape((-1, C))
        output = np.concatenate(output_list, axis=0).reshape((-1, C))
        score_list = np.concatenate(score_list, axis=0).reshape((-1))
        
        # 输出adjustment best f1及best f1在最佳阈值下的原始结果
        best_pred_adj, best_pred = adjbestf1(test_labels, score_list, 100)
        # 计算没有adjustment的结果
        gt = test_labels.astype(int)
        accuracy = accuracy_score(gt, best_pred.astype(int))
        precision, recall, f_score, support = precision_recall_fscore_support(gt, best_pred.astype(int),
                                                                              average='binary')
        gt, adj_pred = adjustment(gt, best_pred_adj)
        adjaccuracy = accuracy_score(gt, adj_pred)
        adjprecision, adjrecall, adjf_score, adjsupport = precision_recall_fscore_support(gt, adj_pred,
                                                                                          average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        print("adjAccuracy : {:0.4f}, adjPrecision : {:0.4f}, adjRecall : {:0.4f}, adjF-score : {:0.4f} ".format(
            adjaccuracy, adjprecision, adjrecall, adjf_score))
        
        # Write results to CSV file
        results = {
            "setting": [setting],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-score': f_score,
            'adjAccuracy': adjaccuracy,
            'adjPrecision': adjprecision,
            'adjRecall': adjrecall,
            'adjF-score': adjf_score
        }
        # 将非迭代的值包装在列表中
        for key in results:
            if not isinstance(results[key], collections.abc.Iterable):
                results[key] = [results[key]]
        
        csv_file = 'results.csv'
        
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))
        
        print("Results appended to", csv_file)
        
        # 将非迭代的值包装在列表中
        for key in results:
            if not isinstance(results[key], collections.abc.Iterable):
                results[key] = [results[key]]
        
        csv_file = folder_path + '/' + 'results.csv'
        
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))
        
        print("Results appended to", csv_file)
        
        # * visualization
        # file_path_border = folder_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AE_border.pdf'
        # file_path = folder_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AE_testset.pdf'
        #
        # visual_anomaly_segment_MS(input, output, best_pred, test_labels, file_path)
        #
        # if self.args.data == 'UCRA':
        #     border_start = self.find_border_number(self.args.data_path)
        #     border1, border2 = self.find_border(self.args.data_path)
        #     input_border = input[
        #                    border1 - border_start - self.args.patch_len * 10: border2 - border_start + self.args.patch_len * 10]
        #     output_border = output[
        #                     border1 - border_start - self.args.patch_len * 10:border2 - border_start + self.args.patch_len * 10]
        #     test_labels_border = test_labels[
        #                          border1 - border_start - self.args.patch_len * 10:border2 - border_start + self.args.patch_len * 10]
        #     best_pred_border = best_pred[
        #                        border1 - border_start - self.args.patch_len * 10:border2 - border_start + self.args.patch_len * 10]
        #     file_path_border = folder_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AE_border.pdf'
        #     visual_anomaly_segment(input_border, output_border, best_pred_border, test_labels_border, file_path_border)
        
        return
    
    def find_border(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border1_str = parts[-2]
        border2_str = parts[-1]
        if '.' in border2_str:
            border2_str = border2_str[:border2_str.find('.')]

        try:
            border1 = int(border1_str)
            border2 = int(border2_str)
            return border1, border2
        except ValueError:
            return None

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None