import torch
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional


@torch.inference_mode
def get_classification_report(model: nn.Module, 
                              device: torch.device,
                              dataloaders: dict[str, data.DataLoader],
                              labels: list[str]
                            ) -> None:
    all_preds = []
    all_labels = []

    model.eval()
    model.to(device)

    for inputs, targets in tqdm(dataloaders['val'], desc="Getting a classification report"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    print("\nClassification Report (VAL):")
    print(classification_report(all_labels, all_preds, target_names=labels[:-1]))


def plot_results(batch: tuple[torch.Tensor], 
                 preds: torch.Tensor, 
                 labels: list[str],
                 grid_size: int = 4,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None
                ) -> None:
    inps, targets = batch

    _, axs = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(20, 20))

    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(grid_size):
        for j in range(grid_size):
            axs[i][j].set_title(f'{labels[preds[i * grid_size + j]]}', color="green" if preds[i * grid_size + j] == targets[ i * grid_size + j] else "red")
            inp = inps[i * grid_size + j].permute(1, 2, 0)
            inp = std * inp + mean
            axs[i][j].imshow(inp)
            axs[i][j].get_xaxis().set_ticks([])
            axs[i][j].get_yaxis().set_ticks([])

    # plt.tight_layout()
    plt.show()