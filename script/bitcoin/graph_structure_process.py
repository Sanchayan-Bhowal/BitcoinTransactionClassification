from collections import defaultdict
from torch_geometric.datasets import EllipticBitcoinDataset


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.graph = {}

    def find(self, s):
        if self.parent[s] != s:
            self.parent[s] = self.find(self.parent[s])

        return self.parent[s]

    def union(self, s1, s2):
        root1 = self.find(s1)
        root2 = self.find(s2)
        
        if s1 not in self.graph:
            self.graph[s1] = set()
        if s2 not in self.graph:
            self.graph[s2] = set()

        self.graph[s1].add(s2)
        self.graph[s2].add(s1)

        if root1 < root2:
            self.parent[root2] = root1

        else:
            self.parent[root1] = root2
    
    def num_component(self):
        parents = set()

        for i in range(len(self.parent)):
            parents.add(self.find(i))

        return len(parents)

    def get_components(self):
        components = {}

        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []

            components[root].append(i)

        return components
    
if __name__ == "__main__":
    dataset = EllipticBitcoinDataset(root='/tmp/EllipticBitcoinDataset')
    data = dataset[0]
    edges = data.edge_index.numpy()
    uf = UnionFind(len(data.x))
    
    for i in range(edges.shape[1]):
        uf.union(edges[0][i], edges[1][i])
    
    print("Number of components in the graph: ",  uf.num_component())
    
    print("-"*48)
    
    for _, component in uf.get_components().items():
        print(len(component))
    
    print("-"*48)
    
    max_deg = 0
    min_deg = float("inf")
    all_degs = []
    all_degs_cnt = defaultdict(int)

    for i in uf.graph:
        max_deg = max(max_deg, len(uf.graph[i]))
        min_deg = min(min_deg, len(uf.graph[i]))
        all_degs.append(len(uf.graph[i]))
        all_degs_cnt[len(uf.graph[i])] += 1

    print(f"maximal degree: {max_deg}, minimal degree: {min_deg}")
    print("Degree counts: ", sorted(all_degs_cnt.items())[:10])