import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



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

        else:
            raise RuntimeError(f"no matched dataset (waterbirds / celeba)")



    def __len__(self):
        return len(self.imfeat)

    def __getitem__(self, idx):
        imfeat=self.imfeat[idx]
        textfeat=self.textfeat[idx]
        labels_y_gt=self.labels_y_gt[idx]
        labels_y=self.labels_y[idx]
        labels_s=self.labels_s[idx]


        return imfeat, textfeat, labels_y ,labels_s, labels_y_gt