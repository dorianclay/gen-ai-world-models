"""
Trainer for trajectory diffusion models.
Ported from https://github.com/jannerm/diffuser — rendering removed (no mujoco dependency).
"""
import os
import copy

import numpy as np
import torch
import wandb


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        out = [_to_device(v, device) for v in x]
        return type(x)(*out) if hasattr(x, '_fields') else type(x)(out)
    return x


def batch_to_device(batch, device='cuda'):
    """Move a Batch namedtuple (with dict conditions) to the target device."""
    return _to_device(batch, device)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_param, cur_param in zip(ema_model.parameters(), current_model.parameters()):
            ema_param.data = self._update(ema_param.data, cur_param.data)

    def _update(self, old, new):
        return old * self.beta + (1 - self.beta) * new if old is not None else new


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training loop for GaussianDiffusion over trajectories.

    Matches the interface of the original diffuser Trainer but without
    any rendering or MuJoCo calls.
    """

    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        save_freq=20000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        device='cuda',
    ):
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.label_freq = label_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.device = device

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ))

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        os.makedirs(results_folder, exist_ok=True)

        self._reset_ema()
        self.step = 0

    def reset_dataloader(self):
        """Rebuild the dataloader after the dataset has been refreshed."""
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ))

    def _reset_ema(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def _step_ema(self):
        if self.step < self.step_start_ema:
            self._reset_ema()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_train_steps):
        self.model.to(self.device)
        self.ema_model.to(self.device)

        epoch_losses = []
        for _ in range(n_train_steps):
            total_loss = 0.0
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_losses.append(total_loss)

            if self.step % self.update_ema_every == 0:
                self._step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join(
                    f'{k}: {v.item() if hasattr(v, "item") else v:.4f}'
                    for k, v in infos.items()
                )
                print(
                    f'step {self.step:>8d} | loss {total_loss:.4f} | {infos_str}',
                    flush=True,
                )
                wandb.log({'loss': total_loss, **{k: float(v) for k, v in infos.items()}},
                          step=self.step)

            self.step += 1

        return float(np.mean(epoch_losses))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, epoch):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ Trainer ] Saved checkpoint to {savepath}', flush=True)

    def load(self, epoch):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        self._load_from_path(loadpath)

    def load_from_path(self, path):
        self._load_from_path(path)

    def _load_from_path(self, path):
        data = torch.load(path, weights_only=True)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        print(f'[ Trainer ] Loaded checkpoint from {path}', flush=True)
