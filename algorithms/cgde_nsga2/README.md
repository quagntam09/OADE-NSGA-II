# CGDE-NSGA-II

Standalone implementation of Cross-Generation Differential Evolution NSGA-II.

The algorithm keeps standard NSGA-II environmental selection, but generates
offspring using a cross-generation DE/current-to-pbest/1 operator:

```text
v = x_i + F * (x_pbest - x_i) + F * (x_r1,current - x_r2,archive)
```

The archive stores solutions from previous generations, so the DE difference
vector explicitly mixes current-generation and previous-generation information.

## Usage

```python
from pymoo.problems import get_problem

from algorithms.cgde_nsga2 import CGDE_NSGAII, ProblemAdapter

problem = ProblemAdapter(get_problem("zdt1", n_var=30))
solver = CGDE_NSGAII(problem, pop_size=100, n_gen=200, F=0.5, CR=0.9, seed=42)
X, F = solver.run()
```

