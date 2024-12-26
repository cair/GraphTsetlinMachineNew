from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from skimage.util import view_as_windows
from keras.datasets import mnist
from numba import jit

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

patch_size = 10
dim = 28 - patch_size + 1

number_of_nodes = dim * dim

symbols = []

# Column and row symbols
for i in range(dim):
    symbols.append("C:%d" % (i))
    symbols.append("R:%d" % (i))

# Patch pixel symbols
for i in range(patch_size*patch_size):
    symbols.append(i)

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
        graphs_train.add_graph_node(graph_id, node_id, 0)
graphs_train.prepare_edge_configuration()
for graph_id in range(X_train.shape[0]):
    if graph_id % 1000 == 0:
        print(graph_id, X_train.shape[0])
     
    windows = view_as_windows(X_train[graph_id,:,:], (patch_size, patch_size))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q*dim + r

                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_train.add_graph_node_property(graph_id, node_id, k)

                graphs_train.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
                graphs_train.add_graph_node_property(graph_id, node_id, "R:%d" % (r))
graphs_train.encode()
print("Training data produced")

graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)
graphs_test.prepare_node_configuration()
for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        graphs_test.add_graph_node(graph_id, node_id, 0)
graphs_test.prepare_edge_configuration()
for graph_id in range(X_test.shape[0]):
    if graph_id % 1000 == 0:
        print(graph_id, X_test.shape[0])
     
    windows = view_as_windows(X_test[graph_id,:,:], (10, 10))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q*dim + r

                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_test.add_graph_node_property(graph_id, node_id, k)

                graphs_test.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
                graphs_test.add_graph_node_property(graph_id, node_id, "R:%d" % (r))
graphs_test.encode()
print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,number_of_state_bits = args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    double_hashing = args.double_hashing
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)
