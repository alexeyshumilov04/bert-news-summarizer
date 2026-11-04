"""Training loop with TensorBoard logging and best-checkpoint saving."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import plot_losses


def _train_step(model, input_ids, attention_mask, labels, optimizer, criterion) -> float:
    model.train()
    optimizer.zero_grad()
    decoder_in = labels[:, :-1]
    decoder_target = labels[:, 1:]
    outputs = model(input_ids, attention_mask, decoder_in)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), decoder_target.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _val_step(model, input_ids, attention_mask, labels, criterion) -> float:
    model.eval()
    decoder_in = labels[:, :-1]
    decoder_target = labels[:, 1:]
    outputs = model(input_ids, attention_mask, decoder_in)
    return criterion(outputs.reshape(-1, outputs.size(-1)), decoder_target.reshape(-1)).item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    log_dir: str,
    checkpoint_path: str,
    plot_path: str | None = None,
) -> tuple[list[float], list[float]]:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    step = 0

    for epoch in range(num_epochs):
        epoch_train = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = _train_step(model, input_ids, attention_mask, labels, optimizer, criterion)
            epoch_train += loss * input_ids.size(0)
            writer.add_scalar("loss/train_step", loss, step)
            step += 1
        epoch_train /= len(train_loader.dataset)
        train_losses.append(epoch_train)

        epoch_val = 0.0
        for batch in tqdm(val_loader, desc=f"epoch {epoch + 1} [val]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            epoch_val += _val_step(model, input_ids, attention_mask, labels, criterion) * input_ids.size(0)
        epoch_val /= len(val_loader.dataset)
        val_losses.append(epoch_val)

        writer.add_scalars("loss/epoch", {"train": epoch_train, "val": epoch_val}, epoch)
        print(f"epoch {epoch + 1}: train={epoch_train:.4f} val={epoch_val:.4f}")

        if epoch_val < best_val:
            best_val = epoch_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": epoch_val,
                },
                checkpoint_path,
            )

    if plot_path:
        plot_losses(train_losses, val_losses, plot_path)

    writer.close()
    return train_losses, val_losses