from torch_geometric.graphgym.golden_model_train import create_golden_vec


if __name__ == '__main__':
    method = 'golden'
    poolings = ['add', 'mean', 'max']
    for pooling in poolings:
        create_golden_vec(pooling=pooling, repeat=10, method=method)
