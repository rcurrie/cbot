#!/usr/bin/env python3
"""Train node classification on Wikipedia dataset with PyTorch Geometric Temporal."""

import logging
from pathlib import Path

import click
import numpy as np
import torch
from torch.nn import functional
from torch_geometric_temporal import temporal_signal_split
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.nn.recurrent import DCRNN

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class RecurrentGCN(torch.nn.Module):
    """Recurrent Graph Convolutional Network for node classification."""

    def __init__(self, node_features: int, num_classes: int) -> None:
        """Initialize the RecurrentGCN model.

        Args:
            node_features: Number of input node features
            num_classes: Number of output classes

        """
        super().__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass.

        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            edge_weight: Edge weights

        Returns:
            Class predictions for nodes

        """
        h = self.recurrent(x, edge_index, edge_weight)
        h = functional.relu(h)
        return self.linear(h)


def prepare_data() -> tuple[object, object, object]:
    """Load and split the Wikipedia dataset.

    Returns:
        Tuple of (dataset, train_dataset, test_dataset)

    """
    logger.info("Loading Wikipedia dataset...")
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset(lags=14)

    logger.info("Dataset loaded with %d snapshots", dataset.snapshot_count)
    logger.info("Number of nodes: %d", dataset[0].x.shape[0])
    logger.info("Number of features: %d", dataset[0].x.shape[1])
    logger.info("Target shape: %s", dataset[0].y.shape)

    # Split dataset 80/20
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    return dataset, train_dataset, test_dataset


def train_epoch(
    model: RecurrentGCN,
    optimizer: torch.optim.Optimizer,
    train_dataset: object,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The model to train
        optimizer: Optimizer for training
        train_dataset: Training dataset
        device: Device to run on

    Returns:
        Average training loss

    """
    model.train()
    losses = []

    for _time, snapshot in enumerate(train_dataset):
        # Move data to device
        x = snapshot.x.to(device)
        y = snapshot.y.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)

        # Forward pass
        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)

        # Compute loss (reshape y to match output shape)
        y = y.reshape(-1, 1) if y.dim() == 1 else y
        loss = functional.mse_loss(out, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def evaluate(
    model: RecurrentGCN,
    test_dataset: object,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on test data.

    Args:
        model: The model to evaluate
        test_dataset: Test dataset
        device: Device to run on

    Returns:
        Tuple of (test_loss, accuracy)

    """
    model.eval()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for _time, snapshot in enumerate(test_dataset):
            # Move data to device
            x = snapshot.x.to(device)
            y = snapshot.y.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)

            # Forward pass
            out = model(x, edge_index, edge_weight)

            # Compute loss
            loss = functional.mse_loss(out, y)
            losses.append(loss.item())

            # For regression, compute MAE instead of accuracy
            y = y.reshape(-1, 1) if y.dim() == 1 else y
            mae = torch.abs(out - y).mean().item()
            correct += mae  # Track total MAE
            total += 1

    mean_mae = correct / total if total > 0 else 0.0
    return np.mean(losses), mean_mae


@click.command()
@click.option(
    "--epochs",
    default=50,
    help="Number of training epochs",
    type=int,
)
@click.option(
    "--lr",
    default=0.01,
    help="Learning rate",
    type=float,
)
@click.option(
    "--output-dir",
    default="./data",
    help="Directory to save model and results",
    type=click.Path(),
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training",
    type=str,
)
def main(epochs: int, lr: float, output_dir: str, device: str) -> None:
    """Train a node classification model on Wikipedia dataset."""
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    logger.info("Using device: %s", device)

    # Load data
    dataset, train_dataset, test_dataset = prepare_data()

    # Initialize model
    node_features = dataset[0].x.shape[1]
    # For regression tasks, output dimension is same as target dimension
    num_classes = 1  # Wikipedia dataset is a regression task

    model = RecurrentGCN(
        node_features=node_features,
        num_classes=num_classes,
    ).to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Starting training for %d epochs...", epochs)

    # Training loop
    best_test_loss = float("inf")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, optimizer, train_dataset, device_obj)

        # Evaluate
        test_loss, mae = evaluate(model, test_dataset, device_obj)

        # Log progress
        if epoch % 10 == 0:
            logger.info(
                "Epoch %03d | Train Loss: %.4f | Test Loss: %.4f | MAE: %.4f",
                epoch,
                train_loss,
                test_loss,
                mae,
            )

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_path = output_path / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            logger.info("Saved best model to %s", model_path)

    logger.info("Training completed!")
    logger.info("Best test loss: %.4f", best_test_loss)

    # Save final model
    final_model_path = output_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info("Saved final model to %s", final_model_path)


if __name__ == "__main__":
    main()
