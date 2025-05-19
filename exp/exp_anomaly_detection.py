from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, visual_anomaly_segment_MS, visual_anomaly_segment
from utils.anomaly_detection_metrics import adjbestf1
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        if self.args.model == "Moment":
            moment = self.model_dict[self.args.model]
            import yaml
            from argparse import Namespace

            with open("Configs/moment.yaml", "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            config = Namespace(**config)
            config = moment.NamespaceWithDefaults.from_namespace(config)
            model = self.model_dict[self.args.model].MOMENT(config)

            model.load_state_dict(torch.load("/data/liuhaixuan/iTransformer_exp-master/moment.pth"))
            print("XXX MOMENT model loaded XXX")
        else:
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
                
                # if self.args.ad_mask_type == 'random':
                #     # random mask
                #     B, T, N = batch_x.shape
                #     """
				# 	B = batch size
				# 	T = seq len
				# 	N = number of features
				# 	"""
                #     assert T % self.args.patch_len == 0
                #     mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                #     mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                #     mask[mask <= self.args.mask_rate] = 0  # masked
                #     mask[mask > self.args.mask_rate] = 1  # remained
                #     mask = mask.view(mask.size(0), -1, mask.size(-1))
                #     mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                #     inp = batch_x.masked_fill(mask == 0, 0)
                #     outputs = self.model(inp, None, None, None)
                #     pred = outputs
                #     f_dim = -1 if self.args.features == 'MS' else 0
                #     true = batch_x[:, :, f_dim:]
                #     mask = mask[:, :, f_dim:]
                #     # loss = criterion(pred[mask == 0], true[mask == 0]).detach().cpu()
                #     loss = criterion(pred, true).detach().cpu()
                # else:
                #     outputs = self.model(batch_x, None, None, None)
                if self.args.model == 'Moment':
                    # random mask
                    B, T, N = batch_x.shape  # [B, L, M]
                    # assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(
                        self.device
                    )  # [B, N, M]
                    mask = mask.unsqueeze(2).repeat(
                        1, 1, self.args.patch_len, 1
                    )  # [B, N, P, M]
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))  # [B, L, M]
                    mask[:, : self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None, mask)
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
                    mask = mask.view(mask.size(0), -1, mask.size(-1))
                    mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None)
                    # loss = criterion(pred[mask == 0], true[mask == 0])
                    loss = criterion(outputs, batch_x)
                elif self.args.model == 'Moment':
                    # random mask
                    B, T, N = batch_x.shape  # [B, L, M]
                    # assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(
                        self.device
                    )  # [B, N, M]
                    mask = mask.unsqueeze(2).repeat(
                        1, 1, self.args.patch_len, 1
                    )  # [B, N, P, M]
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))  # [B, L, M]
                    mask[:, : self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None, mask)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, batch_x)
                else:
                    outputs = self.model(batch_x, None, None, None)

                    # outputs = self.model(batch_x, None, None, None)

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
            self.test(setting, csv_record=False)

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

    def test(self, setting, test=0, csv_record=True):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        input_list = []
        output_list = []
        test_labels_list = []
        score_list = []

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                
                # if self.args.ad_mask_type == 'random':
                #     # random mask
                #     B, T, N = batch_x.shape
                #     """
				# 	B = batch size
				# 	T = seq len
				# 	N = number of features
				# 	"""
                #     assert T % self.args.patch_len == 0
                #     mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                #     mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                #     mask[mask <= self.args.mask_rate] = 0  # masked
                #     mask[mask > self.args.mask_rate] = 1  # remained
                #     mask = mask.view(mask.size(0), -1, mask.size(-1))
                #     mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                #     inp = batch_x.masked_fill(mask == 0, 0)
                #     outputs = self.model(inp, None, None, None)
                # else:
                #     outputs = self.model(batch_x, None, None, None)
                    # criterion
                if self.args.model == 'Moment':
                    # random mask
                    B, T, N = batch_x.shape  # [B, L, M]
                    # assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(
                        self.device
                    )  # [B, N, M]
                    mask = mask.unsqueeze(2).repeat(
                        1, 1, self.args.patch_len, 1
                    )  # [B, N, P, M]
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))  # [B, L, M]
                    mask[:, : self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = self.model(inp, None, None, None, mask)
                else:
                    outputs = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # if self.args.ad_mask_type == 'random':
            #     # random mask
            #     B, T, N = batch_x.shape
            #     """
			# 	B = batch size
			# 	T = seq len
			# 	N = number of features
			# 	"""
            #     assert T % self.args.patch_len == 0
            #     mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
            #     mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
            #     mask[mask <= self.args.mask_rate] = 0  # masked
            #     mask[mask > self.args.mask_rate] = 1  # remained
            #     mask = mask.view(mask.size(0), -1, mask.size(-1))
            #     mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
            #     inp = batch_x.masked_fill(mask == 0, 0)
            #     outputs = self.model(inp, None, None, None)
            # else:
            #     # reconstruction
            #     outputs = self.model(batch_x, None, None, None)
                # criterion
            if self.args.model == 'Moment':
                # random mask
                B, T, N = batch_x.shape  # [B, L, M]
                # assert T % self.args.patch_len == 0
                mask = torch.rand((B, T // self.args.patch_len, N)).to(
                    self.device
                )  # [B, N, M]
                mask = mask.unsqueeze(2).repeat(
                    1, 1, self.args.patch_len, 1
                )  # [B, N, P, M]
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.view(mask.size(0), -1, mask.size(-1))  # [B, L, M]
                mask[:, : self.args.patch_len, :] = 1  # first patch is always observed
                inp = batch_x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, None, None, None, mask)
            else:
                outputs = self.model(batch_x, None, None, None)
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
            
            input_list.append(batch_x.detach().cpu().numpy())
            output_list.append(outputs.detach().cpu().numpy())
            # test_labels_list.append(batch_y.reshape(-1).detach().cpu().numpy())
            # score_list.append(score.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        
        # * Evaluate metrics
        B, S, C = input_list[0].shape
        # test_labels_list = np.concatenate(test_labels_list, axis=0).reshape((-1))
        input = np.concatenate(input_list, axis=0).reshape((-1, C))
        output = np.concatenate(output_list, axis=0).reshape((-1, C))
        # score_list = np.concatenate(score_list, axis=0).reshape((-1))

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # print("pred:   ", pred.shape)
        # print("gt:     ", gt.shape)
        
        pred = np.array(pred)
        gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)
        
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)

        adjaccuracy = accuracy_score(gt, pred)
        adjprecision, adjrecall, adjf_score, adjsupport = precision_recall_fscore_support(gt, pred, average='binary')
        print("AdjAccuracy : {:0.4f}, AdjPrecision : {:0.4f}, AdjRecall : {:0.4f}, AdjF-score : {:0.4f} ".format(
            adjaccuracy, adjprecision, adjrecall, adjf_score))

        # f = open("result_anomaly_detection.txt", 'a')
        # f.write(setting + "  \n")
        # f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #     accuracy, precision,
        #     recall, f_score))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        
        if csv_record:
            # Write results to CSV file
            import csv, collections
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
            
            csv_file = self.args.csv_file
            
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(results.keys())
                writer.writerows(zip(*results.values()))
            
            print("Results appended to", csv_file)
        
        
        # * visualization
        # file_path_border = folder_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AE_border.pdf'
        # file_path = folder_path + '/' + self.args.data + '_AE_testset.pdf'
        # visual_anomaly_segment(input[:,0], output[:,0], pred, gt, file_path)
        return
