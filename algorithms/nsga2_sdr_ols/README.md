# NSGA-II/SDR-OLS

Standalone implementation of NSGA-II/SDR-OLS based on the attached paper:

Zhang, Y.; Wang, G.; Wang, H. "NSGA-II/SDR-OLS: A Novel Large-Scale Many-Objective Optimization Method Using Opposition-Based Learning and Local Search." Mathematics 2023, 11, 1911.

## What is included

- SDR non-dominated sorting with adaptive niche angle theta.
- OBL initialization using `x_opposite = xl + xu - x`.
- Gaussian local search using `omega+` and `omega-`.
- Basic GA variation implemented as binary tournament selection, SBX crossover, and polynomial mutation.
- No dependency on the existing `src/oade_nsga2` implementation.

## Usage

```python
from pymoo.problems import get_problem

from algorithms.nsga2_sdr_ols import NSGAIISDROLS, ProblemAdapter

problem = ProblemAdapter(get_problem("zdt1", n_var=30))
solver = NSGAIISDROLS(
    problem,
    pop_size=100,
    n_gen=200,
    mu=0.0,
    sigma=0.1,
    seed=42,
    local_search_max_neighbors=200,
)

population = solver.run()
F = solver.result_F()
X = solver.result_X()
```

`local_search_max_neighbors=None` follows the paper literally by visiting all variables and generating two neighbors for each visited variable. For large-scale problems this can be very expensive, so a cap is exposed for practical runs.

