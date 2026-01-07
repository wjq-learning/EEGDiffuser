import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
import random
from diff_trainer import Trainer

from dataset.datasets import LoadDataset

from models.eegdiffuser import EEGDiffuser


def main():
    parser = argparse.ArgumentParser(description='Big model down stream')
    parser.add_argument('--seed', type=int, default=8888, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data3/datasets/Faced/processed_filter',
                        help='datasets_dir')
    # parser.add_argument('--classifier_path', type=str,
    #                     default='/data3/wjq/models_weights/DiT/ClassifierMI/epoch13_acc_0.51997_kappa_0.35995_f1_0.49838.pth',
    #                     help='classifier_path')
    parser.add_argument('--model_dir', type=str, default='/data3/wjq/models_weights/DiT/DiTFaced', help='model_dir')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--synthetic_datasets', type=str, default='/data3/datasets/Faced/Synthesis_1train_cfg40_processed_filter',
                        help='synthetic_datasets')
    parser.add_argument('--synthetic_data_dir', type=str, default='/data3/datasets/Faced/Synthesis_1train_cfg40_processed_filter',
                        help='synthetic_data_dir')
    parser.add_argument('--synthetic_ratio', type=int, default=1, help='synthetic_ratio')
    parser.add_argument('--cfg_scale', type=float, default=4.0, help='cfg_scale')



    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = EEGDiffuser()
    t = Trainer(params, data_loader, model)
    # evaluation_best = t.train()
    # print(evaluation_best)
    t.sample()
    # t.synthetic_data()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
