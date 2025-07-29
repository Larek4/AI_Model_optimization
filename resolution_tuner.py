"""import optuna
import torch
from torch.utils.data import DataLoader
from modules.trainer_2 import Trainer
from modules.dataset import IntelImageClassificationDataset

class ResolutionTuner:
    def __init__(self, model_fn, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_fn = model_fn
        self.device = device

    def get_datasets(self, resolution):
        dataset = IntelImageClassificationDataset(resize=(resolution, resolution))
        return dataset.train_dataset, dataset.test_dataset

    def objective(self, trial):
        resolution = trial.suggest_categorical("resolution",[96, 112, 128, 144, 160, 176, 192])
        print(f"Trial {trial.number} - Resolution: {resolution}")

        # Load datasets dynamically based on resolution
        train_dataset, val_dataset = self.get_datasets(resolution)
        #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        # Build and train model
        model = self.model_fn().to(self.device)
        trainer = Trainer(model, device=self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.8593886085709894e-05)
        trainer.train(train_loader, epochs=10, optimizer=optimizer, silent=True)

        # Evaluate on validation set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Trial {trial.number} - Accuracy: {acc:.4f}")

        return acc

    def run(self, n_trials=7):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials)
        return study
"""
# modules/resolution_tuner.py
import optuna
import torch
from torch.utils.data import DataLoader
from modules.trainer_2 import Trainer
from modules.dataset import IntelImageClassificationDataset
from torch.optim.lr_scheduler import CosineAnnealingLR # Added import for LR scheduler


class ResolutionTuner:
    def __init__(self, model_fn, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_fn = model_fn
        self.device = device

    def get_datasets(self, resolution):
        dataset = IntelImageClassificationDataset(resize=(resolution, resolution))
        return dataset.train_dataset, dataset.test_dataset

    def objective(self, trial):
        resolution = trial.suggest_categorical("resolution",[144, 160, 176, 192])
        print(f"Trial {trial.number} - Resolution: {resolution}")

        train_dataset, val_dataset = self.get_datasets(resolution)
        # Added pin_memory=True for DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

        model = self.model_fn().to(self.device)
        trainer = Trainer(model, device=self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004) # Keep the learning rate here as it was

        # For ResolutionTuner, assuming a fixed scheduler for the internal trials
        epochs_for_tuning = 10 # Example fixed epochs for resolution tuning trials
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_for_tuning) # Added scheduler

        # Pass the scheduler to the trainer
        trainer.train(train_loader, epochs=epochs_for_tuning, optimizer=optimizer, silent=True, profiler=None, scheduler=scheduler) # Pass profiler=None

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Trial {trial.number} - Accuracy: {acc:.4f}")

        return acc

    def run(self, n_trials=7):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials)
        return study