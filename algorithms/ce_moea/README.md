# CE-MOEA

Standalone implementation of Continuous Encoding MOEA from:

Sun, J.; Zheng, W.; Zhang, Q.; Xu, Z. "Graph Neural Network Encoding for Community Detection in Attribute Networks." arXiv:2006.03996v2.

## Included

- Graph neural network encoding: sigmoid, softmax, argmax, then locus-style decoding into communities.
- Objective evaluation for attributed networks: `F(x) = (-Q(x), fs(x))` or `F(x) = (-Q(x), fm(x))`.
- NSGA-II ranking, crowding distance, binary tournament selection, and environmental selection.
- Differential Evolution operator followed by polynomial mutation and bound repair.
- No changes to the existing `src/oade_nsga2` package.

## Usage

```python
import numpy as np

from algorithms.ce_moea import AttributedNetwork, AttributedNetworkProblem, CEMOEA

edges = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 3)]
attributes = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])

network = AttributedNetwork.from_edges(6, edges, attributes, undirected=True)
problem = AttributedNetworkProblem(network, attribute_mode="single")

solver = CEMOEA(problem, pop_size=100, n_gen=200, F_DE=0.7, CR=0.5, p_m=0.02, eta_m=20, seed=42)
PS, PF = solver.run()
partitions = solver.decoded_partitions()
```

For multi-attribute data, pass an attribute matrix with shape `(n_nodes, n_attributes)`. The default `multi_attribute_mode="distance"` minimizes `1 - cosine` so smaller values mean better homogeneity. Use `multi_attribute_mode="paper_cosine"` if you want the literal cosine summation printed in the paper.

