from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.tsne import visualization,visualization_PCA
warnings.filterwarnings('ignore')


class Exp_Classification_Ablation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification_Ablation, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

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

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def extract_embedding(self, dataloader):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x, padding_mask, None, None)
                
                embeddings_ = outputs.detach().cpu().numpy()
                embeddings.append(embeddings_)
                labels.append(label.detach().cpu().numpy())
            
            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0).squeeze()
        
        return embeddings, labels
    
    def fit_knn(self, features: npt.NDArray, y: npt.NDArray):
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=self.args.knn_neighbors))
        pipe.fit(features, y)
        return pipe
    
    def fit_svm(self, features: npt.NDArray, y: npt.NDArray, MAX_SAMPLES: int = 10000):
        nb_classes = np.unique(y, return_counts=True)[1].shape[0]
        train_size = features.shape[0]
        
        svm = SVC(C=100000, gamma="scale")
        if train_size // nb_classes < 5 or train_size < 50:
            # print(f"Training SVM with {train_size} examples and {nb_classes} classes")
            return svm.fit(features, y)
        else:
            grid_search = GridSearchCV(
                svm,
                {
                    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                    "kernel": ["rbf"],
                    "degree": [3],
                    "gamma": ["scale"],
                    "coef0": [0],
                    "shrinking": [True],
                    "probability": [False],
                    "tol": [0.001],
                    "cache_size": [200],
                    "class_weight": [None],
                    "verbose": [False],
                    "max_iter": [10000000],
                    "decision_function_shape": ["ovr"],
                    # 'random_state': [None]
                },
                cv=5,
                n_jobs=10,
            )
            # If the training set is too large, subsample MAX_SAMPLES examples
            if train_size > MAX_SAMPLES:
                split = train_test_split(
                    features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
                )
                features = split[0]
                y = split[2]
            
            grid_search.fit(features, y)
            return grid_search.best_estimator_
    
    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        train_embeddings, train_labels = self.extract_embedding(train_loader)
        test_embeddings, test_labels = self.extract_embedding(test_loader)
        
        train_embeddings = torch.flatten(torch.tensor(train_embeddings), start_dim=1).numpy()
        test_embeddings = torch.flatten(torch.tensor(test_embeddings), start_dim=1).numpy()
        classifier = self.fit_knn(features=train_embeddings, y=train_labels)
        # Evaluate the model
        y_pred_train = classifier.predict(train_embeddings)
        y_pred_test = classifier.predict(test_embeddings)
        train_accuracy = classifier.score(train_embeddings, train_labels)
        test_accuracy = classifier.score(test_embeddings, test_labels)
        nb_classes = np.unique(test_labels, return_counts=True)[1].shape[0]
    
        visualization_PCA(
            X=test_embeddings,
            labels=np.array(test_labels),
            token_nums=nb_classes,
            path=folder_path,
            name=self.args.data + f'_{self.args.model_id}_feature_PCA.pdf'
        )
        visualization(
            X=test_embeddings,
            labels=np.array(test_labels),
            token_nums=nb_classes,
            perplexity=self.args.tsne_perplexity,
            path=folder_path,
            name=self.args.data + f'_{self.args.model_id}_feature_perplexity{self.args.tsne_perplexity}.pdf'
        )
        
        # self.model.eval()
        # with torch.no_grad():
        #     for i, (batch_x, label, padding_mask) in enumerate(train_loader):
        #         batch_x = batch_x.float().to(self.device)
        #         padding_mask = padding_mask.float().to(self.device)
        #         label = label.to(self.device)
        #     for i, (batch_x, label, padding_mask) in enumerate(test_loader):
        #         batch_x = batch_x.float().to(self.device)
        #         padding_mask = padding_mask.float().to(self.device)
        #         label = label.to(self.device)
        #
        #         outputs = self.model(batch_x, padding_mask, None, None)
        #
        #         preds.append(outputs.detach())
        #         trues.append(label)
        #
        # preds = torch.cat(preds, 0)
        # trues = torch.cat(trues, 0)
        # print('test shape:', preds.shape, trues.shape)
        #
        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        #
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('train accuracy:{}, test accuracy {}'.format(train_accuracy, test_accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('train accuracy:{}, test accuracy {}'.format(train_accuracy, test_accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
