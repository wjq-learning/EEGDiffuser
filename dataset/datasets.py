import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        # if mode=='train':
        #     length = len(self.keys)
        #     self.keys = self.keys[:int(length * 0.3)]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        data = data.reshape(32, 2000)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class SynthesisDataset(Dataset):
    def __init__(
            self,
            data_dir,
    ):
        super(SynthesisDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # length = len(self.keys)
        # self.keys = self.keys[int(length*0.2):int(length*0.3)]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        data = data.reshape(32, 2000)
        return data / 100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

def custom_collate_fn(batch):
    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return to_tensor(x_data), to_tensor(y_label).long()

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        syn_set = SynthesisDataset(self.params.synthetic_datasets)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        print(len(syn_set))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
            'syn': DataLoader(
                syn_set,
                batch_size=self.params.batch_size,
                collate_fn=syn_set.collate,
                shuffle=True,
            ),
            # 'train_with_sys': DataLoader(
            #     ConcatDataset([train_set, syn_set]),
            #     # syn_set,
            #     batch_size=self.params.batch_size,
            #     collate_fn=custom_collate_fn,
            #     shuffle=True,
            # )
        }
        return data_loader
