# modules/optuna_optimizer.py
import optuna
from modules.trainer_2 import Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR # Added import for LR scheduler


class OptunaTuner:
    def __init__(self, model_fn, train_dataset, val_dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

    def objective(self, trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 2e-3)
        # Expanded batch_size search space to include larger values
        batch_size = trial.suggest_categorical("batch_size", [516, 1024, 2300])
        epochs = trial.suggest_int("epochs", 20, 100, 10)

        model = self.model_fn().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, device=self.device)

        # Added pin_memory=True for DataLoaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Initialize Learning Rate Scheduler within the trial
        # T_max is usually the total number of epochs for the current trial
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Pass the scheduler to the trainer
        # Pass profiler=None to avoid profiler overhead during Optuna tuning trials
        trainer.train(train_loader, epochs=epochs, optimizer=optimizer, silent=True, profiler=None, scheduler=scheduler)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        return acc

    def run(self, n_trials=10, seed=42):
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        return study