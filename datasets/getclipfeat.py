import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class Clipfeature(Dataset):
    """
    Waterbirds dataset from waterbird_complete95_forest2water2 in Group DRO paper
    """

    def __init__(self,split,cfg):

        if cfg.dataset == 'waterbirds':
            if split == 'train':
                labels = np.loadtxt(
                    './features/waterbirds_train.csv',
                    delimiter=',', dtype=str)
                self.imfeat = torch.load(
                    './features/d=waterbirds-s=train-m=%s.pt'%cfg.load_base_model).to(cfg.device)
                self.textfeat = torch.load(
                    './features/text_features_train_waterbirds_%s.pt'%cfg.load_base_model).to(cfg.device)
                self.labels_s = (torch.nn.functional.one_hot(torch.from_numpy(np.loadtxt(
                    './features/prediction-waterbirds-s-train.csv').astype(int)), num_classes=2)) * 2 - 1
                if cfg.nolabel == True:
                    labels_sample = np.loadtxt('./features/prediction-waterbirds-y-train.csv').astype(int)
                    self.labels_y = (torch.nn.functional.one_hot(torch.from_numpy(labels_sample), num_classes=2)) * 2 - 1
                    self.labels_y_gt = (torch.nn.functional.one_hot(torch.from_numpy(labels[:, 2].astype(int)), num_classes=2))*2-1

                else:
                    self.labels_y = (torch.nn.functional.one_hot(torch.from_numpy(labels[:, 2].astype(int)), num_classes=2))*2-1
                    self.labels_y_gt = self.labels_y
                    labels_sample = labels[:, 2].astype(int)

                ##### sampling ######
                data_y0 = np.where(labels_sample == 1)[0]
                data_y1 = np.where(labels_sample == 0)[0]
                data_y0 = np.random.permutation(data_y0)[:int(len(data_y0) * cfg.sample_ratio1)]
                data_y1 = np.random.permutation(data_y1)[:int(len(data_y1) * cfg.sample_ratio2)]
                data = np.hstack((data_y0, data_y1))
                data.sort()
                self.imfeat = self.imfeat[data]
                self.textfeat = self.textfeat[data]
                self.labels_s = self.labels_s[data]
                self.labels_y = self.labels_y[data]

            elif split=='test':
                self.imfeat = torch.load(
                    './features/d=waterbirds-s=test-m=%s.pt'%cfg.load_base_model).to(cfg.device)
                self.textfeat = torch.load(
                    './features/text_features_test_waterbirds_%s.pt' % cfg.load_base_model).to(cfg.device)
                labels= np.loadtxt(
                    './features/waterbirds_test.csv',
                    delimiter=',', dtype=str)
                self.labels_s = (torch.nn.functional.one_hot(torch.from_numpy(labels[:, 4].astype(int)), num_classes=2))*2-1
                self.labels_y = (torch.nn.functional.one_hot(torch.from_numpy(labels[:, 2].astype(int)), num_classes=2))*2-1
                self.labels_y_gt = self.labels_y

        elif cfg.dataset=='celebA':
            if split == 'train':
                labels = np.loadtxt('./features/celeba_train.csv', delimiter=',', dtype=str)

                self.imfeat = torch.load('./features/d=celebA-s=train-m=%s.pt'%cfg.load_base_model).to(cfg.device)

                self.textfeat = torch.load('./features/text_features_train_celeba_blond_gender_%s.pt' % cfg.load_base_model).to(cfg.device)
                
                self.labels_s = (torch.nn.functional.one_hot(torch.from_numpy(np.loadtxt('./features/prediction-celeba-s-train.csv').astype(int)), num_classes=2)) * 2 - 1
                
                if cfg.nolabel == True:
                    labels_sample = np.loadtxt('./features/prediction-celeba-y-train.csv').astype(int)
                    self.labels_y = (torch.nn.functional.one_hot(torch.from_numpy(labels_sample), num_classes=2)) * 2 - 1
                    
                    self.labels_y_gt = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 10].astype(int) + 1) / 2)).astype(int)), num_classes=2)
                    self.labels_y_gt[self.labels_y_gt == 0] = -1

                else:
                    self.labels_y = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 10].astype(int) + 1) / 2)).astype(int)), num_classes=2)
                    self.labels_y[self.labels_y == 0] = -1
                    self.labels_y_gt = self.labels_y
                    labels_sample = (labels[:, 10].astype(int) + 1)/2

                ##### sampling ######
                data_y0 = np.where(labels_sample == 1)[0]
                data_y1 = np.where(labels_sample == 0)[0]
                # balance sampling
                balance_sample = min(len(data_y0), len(data_y1))
                data_y0 = np.random.permutation(data_y0)[:int(balance_sample * cfg.sample_ratio1)]
                data_y1 = np.random.permutation(data_y1)[:int(balance_sample * cfg.sample_ratio2)]
                data = np.hstack((data_y0, data_y1))
                data.sort()
                self.imfeat = self.imfeat[data]
                self.textfeat = self.textfeat[data]
                self.labels_s = self.labels_s[data]
                self.labels_y = self.labels_y[data]


            elif split == 'test':
                self.imfeat = torch.load(
                    './features/d=celebA-s=test-m=%s.pt'%cfg.load_base_model).to(
                    cfg.device)
                self.textfeat = torch.load(
                    './features/text_features_test_celeba_blond_gender_%s.pt' % cfg.load_base_model).to(cfg.device)
                labels = np.loadtxt(
                    './features/celeba_test.csv',
                    delimiter=',', dtype=str)
                self.labels_s = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 21].astype(int)+1)/2)).astype(int)), num_classes=2)
                self.labels_y = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 10].astype(int)+1)/2)).astype(int)), num_classes=2)
                # import pdb; pdb.set_trace()
                self.labels_y[self.labels_y == 0] = -1
                self.labels_s[self.labels_s == 0] = -1
                self.labels_y_gt = self.labels_y

        elif cfg.dataset=='celebA_highcheek':
            if split == 'train':
                labels = np.loadtxt(
                    './features/celeba_train.csv',
                    delimiter=',', dtype=str)
                self.imfeat = torch.load(
                    './features/d=celebA-s=train-m=%s_highcheekbone.pt'%cfg.load_base_model).to(
                    cfg.device)
                self.textfeat = torch.load(
                    './features/text_features_train_celeba_highcheekbones_gender_%s.pt' % cfg.load_base_model).to(cfg.device)
                self.labels_s = torch.nn.functional.one_hot(
                    torch.from_numpy((np.abs((labels[:, 21].astype(int) + 1) / 2 -1)).astype(int)), num_classes=2)
                self.labels_y = torch.nn.functional.one_hot(
                    torch.from_numpy((np.abs((labels[:, 20].astype(int) + 1) / 2 -1)).astype(int)), num_classes=2)
                self.labels_y[self.labels_y == 0] = -1
                self.labels_s[self.labels_s == 0] = -1
                self.labels_y_gt = self.labels_y

                ##### sampling ######
                data_y0 = np.where(labels[:, 20] == '1')[0]
                data_y1 = np.where(labels[:, 20] == '-1')[0]
                data_y0 = np.random.permutation(data_y0)[:int(len(data_y0)*cfg.sample_ratio1)]
                data_y1 = np.random.permutation(data_y1)[:int(len(data_y1)*cfg.sample_ratio2)]
                data = np.hstack((data_y0, data_y1))
                data.sort()
                self.imfeat = self.imfeat[data]
                self.textfeat = self.textfeat[data]
                self.labels_s = self.labels_s[data]
                self.labels_y = self.labels_y[data]


            elif split == 'test':
                self.imfeat = torch.load(
                    './features/d=celebA-s=test-m=%s_highcheekbone.pt'%cfg.load_base_model).to(
                    cfg.device)
                self.textfeat = torch.load(
                    './features/text_features_test_celeba_highcheekbones_gender_%s.pt' % cfg.load_base_model).to(cfg.device)
                labels = np.loadtxt(
                    './features/celeba_test.csv',
                    delimiter=',', dtype=str)
                self.labels_s = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 21].astype(int)+1)/2 -1)).astype(int)), num_classes=2)
                self.labels_y = torch.nn.functional.one_hot(torch.from_numpy((np.abs((labels[:, 20].astype(int)+1)/2 -1)).astype(int)), num_classes=2)
                self.labels_y[self.labels_y == 0] = -1
                self.labels_s[self.labels_s == 0] = -1
                self.labels_y_gt = self.labels_y

            
        elif cfg.dataset=='CFD':
            features = torch.load('./features/d=CFD-s=all-m=clip_ViTL14.pt')
            attrs_file = './features/CFD_attributes.csv'
            target_attr    = "Attractive"
            sensitive_attr    = "GenderSelf"

            df = pd.read_csv(attrs_file)
            labels_y = self.load_attributes(df, target_attr)
            labels_s = self.load_attributes(df, sensitive_attr)

            labels_y_onehot = torch.nn.functional.one_hot(labels_y.long(), num_classes=labels_y.max()+1)
            labels_y_onehot[labels_y_onehot==0] = -1
            labels_s_onehot = torch.nn.functional.one_hot(labels_s.long(), num_classes=labels_s.max()+1)
            labels_s_onehot[labels_s_onehot==0] = -1

            train_split = 0.6
            
            randperm_idx    = torch.randperm(len(features))
            train_idx  = randperm_idx[: int(train_split * len(features))]
            test_idx   = randperm_idx[int(train_split   * len(features)):]
            
            imfeat_train, y_train, y_train_onehot = features[train_idx], labels_y[train_idx], labels_y_onehot[train_idx]
            imfeat_test,  y_test,  s_test,  y_test_onehot,  s_test_onehot  = features[test_idx],  labels_y[test_idx],  labels_s[test_idx],  labels_y_onehot[test_idx],  labels_s_onehot[test_idx]
            
            

            # load the text embeddings
            text_embeddings = torch.load('./features/d=CFD_text_features-m=clip_ViTL14.pt')    
            textfeat_train = text_embeddings[y_train.long()]
            textfeat_test  = text_embeddings[y_test.long()]


            if split == 'train':
                self.imfeat = imfeat_train.to(cfg.device)
                self.textfeat = textfeat_train.to(cfg.device)
                self.labels_y = y_train_onehot.to(cfg.device)
                # self.labels_s = s_train_onehot.to(cfg.device)
                self.labels_y_gt = y_train_onehot.to(cfg.device)

                spurious_prompt_embeddings = torch.load('./features/d=CFD_text_features_s-m=clip_ViTL14.pt')
                s_train = self.get_predictions(imfeat_train, spurious_prompt_embeddings, temperature=100.)
                labels_s_onehot_train_predicted = torch.nn.functional.one_hot(torch.from_numpy(s_train), num_classes=labels_s.max() + 1)
                labels_s_onehot_train_predicted[labels_s_onehot_train_predicted == 0] = -1
                self.labels_s = labels_s_onehot_train_predicted.to(cfg.device)

            elif split == 'test':
                self.imfeat = imfeat_test.to(cfg.device)
                self.textfeat = textfeat_test.to(cfg.device)
                self.labels_y = y_test_onehot.to(cfg.device)
                self.labels_s = s_test_onehot.to(cfg.device)
                self.labels_y_gt = y_test_onehot.to(cfg.device)

        else:
            raise RuntimeError(f"No matched dataset (waterbirds / celeba / CFD)")

    @staticmethod
    def get_predictions(image_embeddings,
                             text_embeddings,
                             temperature=100.):
        with torch.no_grad():
            _image_embeddings = (image_embeddings / 
                                image_embeddings.norm(dim=-1, keepdim=True))
            
            _text_embeddings = (text_embeddings /
                                text_embeddings.norm(dim=-1, keepdim=True))

            cross = _image_embeddings @ _text_embeddings.T
            text_probs = (temperature * cross).softmax(dim=-1)
            _, predicted = torch.max(text_probs.data, 1)
            
        return predicted.cpu().numpy()


    @staticmethod
    def load_attributes(df, attr_name):
        attr = df[attr_name].copy()
        if attr_name in ["GenderSelf", "EthnicitySelf"]:
            for i, cls in enumerate(df[attr_name].unique()):
                attr[attr == cls] = i
                
            attributes = torch.tensor(attr.tolist())
        
        else: # continuous attributes
            attr -= attr.mean()
            attributes = torch.from_numpy(attr.values > 0).int()
        return attributes

    def __len__(self):
        return len(self.imfeat)

    def __getitem__(self, idx):
        imfeat=self.imfeat[idx]
        textfeat=self.textfeat[idx]
        labels_y_gt=self.labels_y_gt[idx]
        labels_y=self.labels_y[idx]
        labels_s=self.labels_s[idx]


        return imfeat, textfeat, labels_y ,labels_s, labels_y_gt