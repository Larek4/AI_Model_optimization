from typing import Mapping, Optional, Iterator, Any, Dict, List
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd.profiler import record_function
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler  # Import base class for scheduler type hinting


class Trainer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 *,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(),
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None

    def train(self,
              dataloader: DataLoader,
              *,
              epochs: int = 100,
              optimizer: torch.optim.Optimizer,
              silent: bool = False,
              profiler=None,
              scheduler: Optional[_LRScheduler] = None) -> List[Dict[str, float]]:  # Changed return type
        history = []
        for epoch_data in self.train_iter(dataloader, epochs=epochs, optimizer=optimizer, silent=silent,
                                          profiler=profiler, scheduler=scheduler):
            history.append(epoch_data['metrics'])
        return history  # Return the complete history

    def train_iter(self,
                   dataloader: DataLoader,
                   *,
                   epochs: int = 100,
                   optimizer: torch.optim.Optimizer,
                   silent: bool = False,
                   profiler=None,
                   scheduler: Optional[_LRScheduler] = None) -> Iterator[Dict[str, Any]]:  # Changed yield type
        model = self.model.to(self.device)
        self._optimizer_to(optimizer, self.device)

        for epoch in range(epochs):
            model.train()
            running_correct, running_total, running_loss = 0, 0, 0.0

            with record_function(f"epoch_{epoch + 1}"):
                for i, (data, target) in enumerate(dataloader):
                    if profiler:
                        profiler.start_batch_timer()

                    optimizer.zero_grad()
                    x, y = data.to(self.device), target.to(self.device)

                    with autocast(enabled=self.device.type == 'cuda', device_type='cuda'):
                        preds = model(x)
                        loss = self.loss_fn(preds, y)

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    if profiler:
                        profiler.step()
                        profiler.log_batch_stats(i)

                    running_loss += loss.item()
                    predicted = preds.argmax(dim=1)
                    running_correct += (predicted == y).sum().item()
                    running_total += y.size(0)

            acc = running_correct / running_total
            avg_loss = running_loss / len(dataloader)

            if scheduler:  # Step the scheduler at the end of the epoch
                scheduler.step()

            if not silent:
                print(f"✅ Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

            # Yield model and metrics
            yield {'model': model, 'metrics': {'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': acc}}

    def _optimizer_to(self, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for param in optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam.grad is not None:
                            subparam.grad.data = subparam.grad.data.to(device)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        super().load_state_dict(state_dict, strict, assign)