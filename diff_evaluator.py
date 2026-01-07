import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from classifier.eegconformer import Conformer
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

class Evaluator:
    def __init__(self, params, data_loader, diffusion):
        self.params = params
        self.data_loader = data_loader
        self.diffusion = diffusion

    def get_metrics(self, model):
        model.eval()

        losses = []
        for i, (x, y) in tqdm(enumerate(self.data_loader), mininterval=1):
            x = x.cuda()
            y = y.cuda()

            gen = torch.Generator()
            gen.manual_seed(i)
            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), generator=gen).cuda()
            model_kwargs = dict(y=y)
            loss_dict = self.diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            losses.append(loss.data.cpu().numpy())

        avg_val_loss = np.mean(losses)
        return avg_val_loss