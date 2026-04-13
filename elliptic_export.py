import pandas as pd
from torch_geometric.datasets import EllipticBitcoinTemporalDataset


def main() -> None:
    timestep = 1
    dataset = EllipticBitcoinTemporalDataset(root="data/EllipticBitcoin", t=timestep)
    data = dataset[0]

    x = data.x
    output_path = f"elliptic_t{timestep}_features.csv"

    print(f"Timestep: {timestep}")
    print(f"Feature matrix shape: {tuple(x.shape)}")

    df = pd.DataFrame(x.numpy())
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()
