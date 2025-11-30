## Algorithms

This project implements/analyzes Quasi-Newton methods for unconstrained non-linear optimization.

**BFGS (Broyden–Fletcher–Goldfarb–Shanno)**
BFGS is an iterative method for solving unconstrained non-linear optimization problems. As a Quasi-Newton method, it approximates the Hessian matrix of the objective function using gradient evaluations.
- **Mechanism:** It maintains a full dense approximation of the inverse Hessian matrix, updating it at each step based on the Secant equation.
- **Performance:** It offers superlinear convergence and is generally robust.
- **Memory Cost:** $O(n^2)$, where $n$ is the number of variables. This makes it suitable for small to medium-sized problems.

**L-BFGS (Limited-memory BFGS)**
L-BFGS is an optimization algorithm in the family of Quasi-Newton methods that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) using a limited amount of computer memory.
- **Mechanism:** Instead of storing the full $n \times n$ inverse Hessian matrix, it stores only a few vectors that represent the approximation implicitly. It maintains a history of the past $m$ updates (where $m$ is typically small, e.g., 5-20).
- **Performance:** Similar convergence properties to BFGS but slightly less robust on very ill-conditioned problems.
- **Memory Cost:** $O(mn)$, making it ideal for large-scale problems with thousands or millions of variables.

### Organization of the code

The directory structure is as follows:

```
./Project
  ├── build/
  ├── lib/
  ├── paper/
  └── src/ 
```

- `build` contains the output of CMAKE
- `lib` contains the external libraries
- `paper`containes the academic literature provided
- `src` contains all the C++ code to be compiled

### Compiling

To compile we use CMAKE, from the `Project` directory run:

```zsh
mkdir build
cd build
cmake ..
make
```
#### Run
To run the project from the `build` directory:
```zsh
./main.o
```