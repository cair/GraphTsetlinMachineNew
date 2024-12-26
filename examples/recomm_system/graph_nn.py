import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import prepare_dataset
from tmu.tools import BenchmarkTimer
import os
import pandas as pd

def main(args):
    results = []
    data = prepare_dataset.aug_amazon_products(noise_ratio = args.dataset_noise_ratio)
    x, y = prepare_dataset.construct_x_y(data)
    X_train, X_test, Y_train, Y_test = prepare_dataset.train_test_split(x,y)
    # Graph Construction
    num_users = len(data['user_id'].unique())
    num_items = len(data['product_id'].unique())
    num_categories = len(data['category'].unique())
    num_nodes = num_users + num_items + num_categories
    # Build edge list
    edge_list = []
    # User ↔ Item edges
    for user, item in zip(X_train[:, 0], X_train[:, 1]):
        edge_list.append((user, num_users + item))  # User to Item
        edge_list.append((num_users + item, user))  # Item to User
    # Item ↔ Category edges
    for item, category in zip(X_train[:, 1], X_train[:, 2]):
        edge_list.append((num_users + item, num_users + num_items + category))  # Item to Category
        edge_list.append((num_users + num_items + category, num_users + item))  # Category to Item
    # Create edge index for PyTorch Geometric
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    # Node features
    node_features = torch.rand((num_nodes, 64), dtype=torch.float)
    # PyTorch Geometric Data object
    graph_data = Data(x=node_features, edge_index=edge_index)
    # Step 2: Define GCN Model
    class GCN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x
    # Initialize Model
    model = GCN(input_dim=64, hidden_dim=128, output_dim=64)
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Convert train/test data to tensors
    train_edges = torch.tensor(
        [(user, num_users + item) for user, item in zip(X_train[:, 0], X_train[:, 1])],
        dtype=torch.long
    ).t()
    train_labels = torch.tensor(Y_train, dtype=torch.float)
    test_edges = torch.tensor(
        [(user, num_users + item) for user, item in zip(X_test[:, 0], X_test[:, 1])],
        dtype=torch.long
    ).t()
    test_labels = torch.tensor(Y_test, dtype=torch.float)
    # Training Loop with Accuracy Logging
    benchmark_total = BenchmarkTimer(logger=None, text="Epochs Time")
    with benchmark_total:
        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                # Training Phase
                model.train()
                optimizer.zero_grad()
                out = model(graph_data.x, graph_data.edge_index)
                # User-item embeddings
                user_embeddings = out[train_edges[0]]
                item_embeddings = out[train_edges[1]]
                predicted_ratings = (user_embeddings * item_embeddings).sum(dim=1)
                # Compute loss
                loss = F.mse_loss(predicted_ratings, train_labels)
                loss.backward()
                optimizer.step()
            train_time = benchmark1.elapsed()
            # Testing Phase
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                model.eval()
                with torch.no_grad():
                    out = model(graph_data.x, graph_data.edge_index)
                    test_user_embeddings = out[test_edges[0]]
                    test_item_embeddings = out[test_edges[1]]
                    test_predicted_ratings = (test_user_embeddings * test_item_embeddings).sum(dim=1)
                    # Compute accuracy
                    accuracy = ((test_predicted_ratings.round() == test_labels).float().mean().item()) * 100
            test_time = benchmark2.elapsed()
    total_time = benchmark_total.elapsed()
    # Append results for each epoch
    results.append({
        "Algorithm": "Graph NN",
        "Noise_Ratio": args.dataset_noise_ratio,
        "T": 0,
        "s": 0,
        "Max_Included_Literals": 0,
        "Epochs": args.epochs,
        "Platform": args.platform,
        "Total_Time": total_time,
        "Accuracy": accuracy,
    })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_file = "experiment_results.csv"
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', index=False, header=False)
    else:
        results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="CPU", type=str, choices=["CPU", "CUDA"])
    parser.add_argument("--epochs", default=20000, type=int)
    parser.add_argument("--dataset_noise_ratio", default=0.01, type=float)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    main(default_args())