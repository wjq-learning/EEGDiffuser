import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from diff_evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import umap
from sklearn.decomposition import PCA
from utils.util import VLBLoss, draw
from diffusion import create_diffusion
from collections import OrderedDict
from copy import deepcopy
import lmdb
import pickle

class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.model = model.cuda()
        self.criterion = VLBLoss().cuda()

        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                               weight_decay=self.params.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                             weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

        self.diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        self.evaluator = Evaluator(params, self.data_loader['val'], self.diffusion)

    def train(self):
        ema = deepcopy(self.model).cuda()  # Create an EMA of the model for use after training
        requires_grad(ema, False)

        update_ema(ema, self.model, decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode

        print(f"Training for {self.params.epochs} epochs...")
        best_avg_loss = 1000000
        best_epoch = 0
        for epoch in range(self.params.epochs):
            print(f"Beginning epoch {epoch}...")
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],)).cuda()
                model_kwargs = dict(y=y)
                loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema(ema, self.model)
                self.optimizer_scheduler.step()

                losses.append(loss.data.cpu().numpy())

            avg_training_loss = np.mean(losses)
            optim_state = self.optimizer.state_dict()
            with torch.no_grad():
                avg_val_loss = self.evaluator.get_metrics(model=self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, Validation Loss: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        avg_training_loss,
                        avg_val_loss,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if best_avg_loss > avg_val_loss:
                    best_epoch = epoch + 1
                    best_avg_loss = avg_val_loss
                    model_path = self.params.model_dir + "/epoch{}_avgloss_{:.5f}.pth".format(epoch + 1, avg_val_loss)
                    torch.save(self.model.state_dict(), model_path)
                    print("model save in " + model_path)

                # if (epoch + 1) % 100 == 0:
                #     model_path = self.params.model_dir + "/epoch{}_avgloss_{:.5f}.pth".format(epoch + 1, avg_val_loss)
                #     torch.save(self.model.state_dict(), model_path)
                #     print("model save in " + model_path)

            if epoch + 1 == self.params.epochs:
                print("{} epoch get the best avgloss {:.5f}".format(best_epoch, best_avg_loss))
                print("the model is save in " + model_path)
        evaluation_best = best_avg_loss
        return evaluation_best

    def sample(self):
        CHANNEL_LIST = [
            'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
            'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4',
            'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
            'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4',
            'Oz', 'O1', 'O2', 'A2', 'A1'
        ]
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        map_location = torch.device(f'cuda:{self.params.cuda}')
        self.model.load_state_dict(
            torch.load(
                '/data3/wjq/models_weights/DiT/DiTFaced/epoch4649_avgloss_0.01280.pth',
                map_location=map_location
            )
        )
        self.model.eval()
        device = next(self.model.parameters()).device
        # Labels to condition the model with (feel free to change):
        class_labels = [1, 1, 1, 4, 4, 4]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 32, 2000).cuda()
        y = torch.tensor(class_labels).cuda()

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([self.params.num_of_classes] * n).cuda()
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=self.params.cfg_scale)

        samples = diffusion.p_sample_loop(
            self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device,
        ).cuda()
        # print(samples.shape)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        for sample in samples:
            sample = sample.cpu().numpy() * 100
            # print(sample)
            draw(sample, CHANNEL_LIST)

    def synthetic_data(self):
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        map_location = torch.device(f'cuda:{self.params.cuda}')
        self.model.load_state_dict(
            torch.load(
                '/data3/wjq/models_weights/DiT/DiTFaced/epoch4985_avgloss_0.01200.pth',
                map_location=map_location
            )
        )
        self.model.eval()
        device = next(self.model.parameters()).device

        db = lmdb.open(self.params.synthetic_data_dir, map_size=66125001720)
        test_n = 0
        keys = []
        for epoch in range(self.params.synthetic_ratio):
            print(f'Epoch:{epoch}')
            for x, y in self.data_loader['train']:
                y = y.cuda()
                # Create sampling noise:
                n = y.shape[0]
                z = torch.randn(n, 32, 2000).cuda()

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([self.params.num_of_classes] * n).cuda()
                # print(y_null)
                y_with_null = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y_with_null, cfg_scale=self.params.cfg_scale)

                samples = diffusion.p_sample_loop(
                    self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                ).cuda()
                # print(samples.shape)
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                for sample, label in zip(samples, y):
                    sample = sample.contiguous().view(32, 10, 200).cpu().numpy() * 100
                    label = label.cpu().numpy()
                    data_dict = {
                        'sample': sample, 'label': label
                    }
                    txn = db.begin(write=True)
                    txn.put(key=str(test_n).encode(), value=pickle.dumps(data_dict))
                    txn.commit()
                    keys.append(str(test_n))
                    test_n += 1

        txn = db.begin(write=True)
        txn.put(key='__keys__'.encode(), value=pickle.dumps(keys))
        txn.commit()
        db.close()
        print('End!')




# -7.517609
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# extractnum /data/wjq/EEGDiT/DiTMI/logs/log09 --pattern "Training Loss: {loss}" --output loss.png