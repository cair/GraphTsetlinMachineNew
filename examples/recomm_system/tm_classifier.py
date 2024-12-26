import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import prepare_dataset
import pandas as pd
import os

def main(args):
    results = []
    data = prepare_dataset.aug_amazon_products(noise_ratio = args.dataset_noise_ratio)
    x, y = prepare_dataset.construct_x_y(data)
    X_train, X_test, Y_train, Y_test = prepare_dataset.one_hot_encoding(x,y)
    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
    )
        
    benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
    with benchmark_total:
        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)
            train_time = benchmark1.elapsed()
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
            test_time = benchmark2.elapsed()
    total_time = benchmark_total.elapsed()
        
    # Append results for each epoch
    results.append({
        "Algorithm": "TMClassifier",
        "Noise_Ratio": args.dataset_noise_ratio,
        "T": args.T,
        "s": args.s,
        "Max_Included_Literals": args.max_included_literals,
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
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="CPU_sparse", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--dataset_noise_ratio", default=0.01, type=float)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    main(default_args())