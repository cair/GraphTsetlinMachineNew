from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import argparse
import numpy as np
import prepare_dataset
import pandas as pd
from tmu.tools import BenchmarkTimer
import os

def main(args):  
    np.random.seed(42)
    results = []
    data = prepare_dataset.aug_amazon_products(noise_ratio = args.dataset_noise_ratio)
    x, y = prepare_dataset.construct_x_y(data)
    X_train, X_test, Y_train, Y_test = prepare_dataset.train_test_split(x,y)
    users = data['user_id'].unique()
    items = data['product_id'].unique()
    categories = data['category'].unique()
    # Initialize Graphs with symbols for GTM
    number_of_nodes = 3
    symbols = []
    symbols = ["U_" + str(u) for u in users] + ["I_" + str(i) for i in items] + ["C_" + str(c) for c in categories] 
    print("Symbols: ",len(symbols))

    # Train data
    graphs_train = Graphs(
        X_train.shape[0],
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing = args.double_hashing
    )
    for graph_id in range(X_train.shape[0]):
        graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)
    graphs_train.prepare_node_configuration()
    for graph_id in range(X_train.shape[0]):
        for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
            number_of_edges = 2 if node_id > 0 and node_id < graphs_train.number_of_graph_nodes[graph_id]-1 else 1
            if node_id == 0:
                graphs_train.add_graph_node(graph_id, "User", number_of_edges)
            elif node_id == 1:
                graphs_train.add_graph_node(graph_id, "Item", number_of_edges)
            else:
                graphs_train.add_graph_node(graph_id, "Category", number_of_edges)
    graphs_train.prepare_edge_configuration()
    for graph_id in range(X_train.shape[0]):
        for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
            if node_id == 0:
                graphs_train.add_graph_node_edge(graph_id, "User", "Item", "UserItem")
                
            if node_id == 1:
                graphs_train.add_graph_node_edge(graph_id, "Item", "Category", "ItemCategory")
                graphs_train.add_graph_node_edge(graph_id, "Item", "User", "ItemUser")
                
            if node_id == 2:
                graphs_train.add_graph_node_edge(graph_id, "Category", "Item", "CatrgoryItem")

        graphs_train.add_graph_node_property(graph_id, "User", "U_" + str(X_train[graph_id][0]))
        graphs_train.add_graph_node_property(graph_id, "Item", "I_" + str(X_train[graph_id][1]))
        graphs_train.add_graph_node_property(graph_id, "Category", "C_" + str(X_train[graph_id][2]))
    graphs_train.encode()
    print("Training data produced")

    # Test data
    graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
    for graph_id in range(X_test.shape[0]):
        graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)
    graphs_test.prepare_node_configuration()
    for graph_id in range(X_test.shape[0]):
        for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
            number_of_edges = 2 if node_id > 0 and node_id < graphs_test.number_of_graph_nodes[graph_id]-1 else 1
            if node_id == 0:
                graphs_test.add_graph_node(graph_id, "User", number_of_edges)
            elif node_id == 1:
                graphs_test.add_graph_node(graph_id, "Item", number_of_edges)
            else:
                graphs_test.add_graph_node(graph_id, "Category", number_of_edges)
    graphs_test.prepare_edge_configuration()
    for graph_id in range(X_test.shape[0]):
        for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
            if node_id == 0:
                graphs_test.add_graph_node_edge(graph_id, "User", "Item", "UserItem")
                
            if node_id == 1:
                graphs_test.add_graph_node_edge(graph_id, "Item", "Category", "ItemCategory")
                graphs_test.add_graph_node_edge(graph_id, "Item", "User", "ItemUser")
                
            if node_id == 2:
                graphs_test.add_graph_node_edge(graph_id, "Category", "Item", "CatrgoryItem")

        graphs_test.add_graph_node_property(graph_id, "User", "U_" + str(X_test[graph_id][0]))
        graphs_test.add_graph_node_property(graph_id, "Item", "I_" + str(X_test[graph_id][1]))
        graphs_test.add_graph_node_property(graph_id, "Category", "C_" + str(X_test[graph_id][2]))
    graphs_test.encode()
    print("Testing data produced")

    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits = args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing = args.double_hashing
    )

    benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
    with benchmark_total:
        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
            train_time = benchmark1.elapsed()
            
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                accuracy = 100*(tm.predict(graphs_test) == Y_test).mean()
            test_time = benchmark2.elapsed()
    total_time = benchmark_total.elapsed()
    # result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
    results.append({
        "Algorithm": "GraphTM",
        "Noise_Ratio": args.dataset_noise_ratio,
        "T": args.T,
        "s": args.s,
        "Max_Included_Literals": args.max_included_literals,
        "Epochs": args.epochs,
        "Platform": "CUDA",
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
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=4096, type=int)
    parser.add_argument("--hypervector-bits", default=256, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--max-included-literals", default=23, type=int)
    parser.add_argument("--dataset_noise_ratio", default=0.01, type=float)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    main(default_args())