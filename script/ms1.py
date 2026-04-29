from torch_geometric.datasets import EllipticBitcoinTemporalDataset

dataset = EllipticBitcoinTemporalDataset(
    root="data/EllipticBitcoin",
    t=1
)

data = dataset[0]
print(data)
print(data.x.shape)
print(data.edge_index.shape)
print(data.y.shape)
