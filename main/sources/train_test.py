import sys
from pathlib import Path

# Add the parent directory of `dc1` to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Fix potential extra dimension issue
        if x.dim() == 5:  # Check if there's an extra dim
            x = x.squeeze(1)
        if x.dim() == 5:
            x = x.squeeze(1)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Fix potential extra dimension issue
            if x.dim() == 5:
                x = x.squeeze(1)
            if x.dim() == 5:
                x = x.squeeze(1)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
    return losses
