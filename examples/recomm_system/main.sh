echo `date`, Setup the environment ...
set -e  # exit if error

models="graph_tm tm_classifier graph_nn"
dataset_noise_ratios="0.005 0.01 0.02 0.05 0.1 0.2"

for N in $dataset_noise_ratios; do
    echo `date`, Running Graph NN ...
    python3 graph_nn.py --dataset_noise_ratio $N

    echo `date`, Running Graph Tsetlin Machine ...
    python3 graph_tm.py --dataset_noise_ratio $N
    
    echo `date`, Running Tsetlin Machine Classifier ...
    python3 tm_classifier.py --dataset_noise_ratio $N
done