"""
Learning Rate Scheduler Model Generator

Generates new_nn.py files in the standard output directory
(out/nngpt/llm/epoch/A0/synth_nn/) for evaluation by NNEval.

Each generated model is a ResNet18 adapted for CIFAR-10 with a
different learning rate scheduling strategy. All hyperparameters
are provided via the prm dict, consistent with the project approach.

Usage:
    python -m ab.gpt.brute.lr.schedulers
    python -m ab.gpt.NNEval
"""
import os
import shutil
from pathlib import Path

from ab.gpt.util.Const import epoch_dir, synth_dir, new_nn_file

OUTPUT_DIR = synth_dir(epoch_dir(0))
MODEL_PREFIX = "A0_"

# ---------------------------------------------------------------------------
# Model Template
# ---------------------------------------------------------------------------
MODEL_TEMPLATE = '''import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

def supported_hyperparameters():
    return {supported_hyperparameters_set}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 32, 32), out_shape=(10,), prm=None, device='cpu'):
        super(Net, self).__init__()
        if prm is None:
            prm = {{'lr': 0.1, 'momentum': 0.9, 'dropout': 0.2}}
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.model = models.resnet18(weights=None)
        num_classes = out_shape[0] if isinstance(out_shape, tuple) else out_shape
        self.model.fc = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.1),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )
        optimizer = self.optimizer
        prm.setdefault('epoch_max', prm.get('epoch', 90))
{scheduler_code}
        self.scheduler = scheduler

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

def get_model_and_optimizer(in_shape=(1, 3, 32, 32), out_shape=(10,), max_epoch=90, learning_rate=0.1, dropout=0.2, momentum=0.9, device='cpu', **kwargs):
    prm = {{'lr': learning_rate, 'momentum': momentum, 'dropout': dropout, 'epoch_max': max_epoch, 'epoch': max_epoch}}
    prm.update(kwargs)
    model = Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
    model.train_setup(prm)
    return model, model.optimizer, model.scheduler
'''

# ---------------------------------------------------------------------------
# Scheduler Strategies
# ---------------------------------------------------------------------------
STRATEGIES = [
    {
        "name": "StepLR_default",
        "hyperparams": {"lr", "momentum", "dropout", "step_size", "gamma"},
        "code": "        scheduler = lr_scheduler.StepLR(optimizer, step_size=prm.get('step_size', 30), gamma=prm.get('gamma', 0.1))"
    },
    {
        "name": "MultiStepLR_milestones",
        "hyperparams": {"lr", "momentum", "dropout", "milestone0", "milestone1", "milestone2", "gamma"},
        "code": "        milestones = sorted(set([max(1, int(prm['epoch_max'] * prm.get('milestone0', 0.3))), max(1, int(prm['epoch_max'] * prm.get('milestone1', 0.6))), max(1, int(prm['epoch_max'] * prm.get('milestone2', 0.8)))]))\n        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=prm.get('gamma', 0.1))"
    },
    {
        "name": "ExponentialLR_default",
        "hyperparams": {"lr", "momentum", "dropout", "gamma"},
        "code": "        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=prm.get('gamma', 0.95))"
    },
    {
        "name": "PolynomialLR_quadratic",
        "hyperparams": {"lr", "momentum", "dropout", "power"},
        "code": "        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=prm['epoch_max'], power=prm.get('power', 1.0))"
    },
    {
        "name": "PolynomialLR_cubic",
        "hyperparams": {"lr", "momentum", "dropout", "power"},
        "code": "        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=prm['epoch_max'], power=prm.get('power', 2.0))"
    },
    {
        "name": "CosineAnnealingLR_default",
        "hyperparams": {"lr", "momentum", "dropout", "eta_min"},
        "code": "        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm['epoch_max'], eta_min=prm.get('eta_min', 0.0))"
    },
    {
        "name": "CosineAnnealingLR_with_min_lr",
        "hyperparams": {"lr", "momentum", "dropout", "min_lr"},
        "code": "        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm['epoch_max'], eta_min=prm.get('lr', 0.1) * prm.get('min_lr', 0.01))"
    },
    {
        "name": "CosineAnnealingWarmRestarts_T0_small",
        "hyperparams": {"lr", "momentum", "dropout", "T_0", "T_mult", "eta_min"},
        "code": "        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(prm.get('T_0', 10)), T_mult=prm.get('T_mult', 2), eta_min=prm.get('eta_min', 0.0))"
    },
    {
        "name": "CosineAnnealingWarmRestarts_T0_large",
        "hyperparams": {"lr", "momentum", "dropout", "T_0", "T_mult", "eta_min"},
        "code": "        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(prm.get('T_0', 20)), T_mult=prm.get('T_mult', 1), eta_min=prm.get('eta_min', 0.0))"
    },
    {
        "name": "CyclicLR_triangular",
        "hyperparams": {"lr", "momentum", "dropout", "min_lr", "lr_step"},
        "code": "        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=max(1, int(prm['epoch_max'] * prm.get('lr_step', 0.1))), mode='triangular')"
    },
    {
        "name": "CyclicLR_triangular2",
        "hyperparams": {"lr", "momentum", "dropout", "min_lr", "lr_step"},
        "code": "        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=max(1, int(prm['epoch_max'] * prm.get('lr_step', 0.1))), mode='triangular2')"
    },
    {
        "name": "CyclicLR_exp_range",
        "hyperparams": {"lr", "momentum", "dropout", "min_lr", "lr_step", "cycle_momentum"},
        "code": "        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=max(1, int(prm['epoch_max'] * prm.get('lr_step', 0.1))), mode='exp_range', gamma=prm.get('cycle_momentum', 0.85))"
    },
    {
        "name": "OneCycleLR_default",
        "hyperparams": {"lr", "momentum", "dropout", "pct_start", "div_factor"},
        "code": "        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=prm.get('lr', 0.1), total_steps=max(2, prm['epoch_max']), pct_start=prm.get('pct_start', 0.3), anneal_strategy='cos', div_factor=prm.get('div_factor', 25.0))"
    },
    {
        "name": "OneCycleLR_aggressive",
        "hyperparams": {"lr", "momentum", "dropout", "pct_start", "div_factor"},
        "code": "        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=prm.get('lr', 0.1), total_steps=max(2, prm['epoch_max']), pct_start=prm.get('pct_start', 0.1), anneal_strategy='linear', div_factor=prm.get('div_factor', 10.0))"
    },
    {
        "name": "LinearLR_warmup",
        "hyperparams": {"lr", "momentum", "dropout", "warmup_epochs", "start_factor"},
        "code": "        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=prm.get('start_factor', 0.1), total_iters=max(1, int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))))"
    },
    {
        "name": "ConstantLR_warmup",
        "hyperparams": {"lr", "momentum", "dropout", "warmup_epochs", "factor"},
        "code": "        scheduler = lr_scheduler.ConstantLR(optimizer, factor=prm.get('factor', 0.5), total_iters=max(1, int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))))"
    },
    {
        "name": "LambdaLR_linear_decay",
        "hyperparams": {"lr", "momentum", "dropout"},
        "code": "        epoch_max = prm['epoch_max']\n        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.1, (1.0 - epoch / epoch_max)))"
    },
    {
        "name": "LambdaLR_power_decay",
        "hyperparams": {"lr", "momentum", "dropout", "power"},
        "code": "        epoch_max = prm['epoch_max']\n        power = prm.get('power', 2.0)\n        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1.0 - epoch / epoch_max) ** power)"
    },
    {
        "name": "MultiplicativeLR_exponential",
        "hyperparams": {"lr", "momentum", "dropout", "gamma"},
        "code": "        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: prm.get('gamma', 0.95))"
    },
    {
        "name": "ChainedScheduler_warmup_cosine",
        "hyperparams": {"lr", "momentum", "dropout", "warmup_epochs", "eta_min"},
        "code": "        warmup = lr_scheduler.LinearLR(optimizer, start_factor=prm.get('start_factor', 0.1), total_iters=max(1, int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))))\n        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(prm['epoch_max'] * (1 - prm.get('warmup_epochs', 0.1)))), eta_min=prm.get('eta_min', 0.0))\n        scheduler = lr_scheduler.ChainedScheduler([warmup, cosine])"
    },
    {
        "name": "SequentialLR_warmup_step",
        "hyperparams": {"lr", "momentum", "dropout", "warmup_epochs", "gamma"},
        "code": "        warmup_iters = max(1, int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1)))\n        warmup = lr_scheduler.LinearLR(optimizer, start_factor=prm.get('start_factor', 0.1), total_iters=warmup_iters)\n        decay = lr_scheduler.StepLR(optimizer, step_size=max(1, int(prm['epoch_max'] * 0.3)), gamma=prm.get('gamma', 0.1))\n        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_iters])"
    },
    {
        "name": "LambdaLR_exponential_decay",
        "hyperparams": {"lr", "momentum", "dropout", "gamma"},
        "code": "        gamma = prm.get('gamma', 0.95)\n        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: gamma ** epoch)"
    },
    {
        "name": "LambdaLR_cosine_like",
        "hyperparams": {"lr", "momentum", "dropout"},
        "code": "        import math\n        epoch_max = prm['epoch_max']\n        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / epoch_max)))"
    },
    {
        "name": "StepLR_long_decay",
        "hyperparams": {"lr", "momentum", "dropout", "gamma"},
        "code": "        scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, int(prm['epoch_max'] * 0.5)), gamma=prm.get('gamma', 0.1))"
    },
    {
        "name": "CosineAnnealingLR_high_min",
        "hyperparams": {"lr", "momentum", "dropout", "min_lr"},
        "code": "        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm['epoch_max'], eta_min=prm.get('lr', 0.1) * prm.get('min_lr', 0.1))"
    },
]


def generate():
    """Generate new_nn.py files in the standard output directory."""
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Clean __pycache__ which can confuse NNEval
    pycache = os.path.join(str(OUTPUT_DIR), "__pycache__")
    if os.path.exists(pycache):
        shutil.rmtree(pycache)

    generated = []
    for i, strategy in enumerate(STRATEGIES):
        module_name = f"{MODEL_PREFIX}{i + 1:03d}"
        model_dir = os.path.join(str(OUTPUT_DIR), module_name)
        os.makedirs(model_dir, exist_ok=True)

        filepath = os.path.join(model_dir, new_nn_file)
        hyperparams_str = repr(sorted(strategy["hyperparams"]))
        content = MODEL_TEMPLATE.format(
            supported_hyperparameters_set=hyperparams_str,
            scheduler_code=strategy["code"]
        )

        with open(filepath, "w") as f:
            f.write(content)

        generated.append(module_name)
        print(f"  Generated: {module_name} ({strategy['name']})")

    print(f"\nGenerated {len(generated)} models in {OUTPUT_DIR}")
    print("Run validation with: python -m ab.gpt.NNEval")
    return generated


if __name__ == "__main__":
    generate()
