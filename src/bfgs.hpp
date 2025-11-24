#pragma once

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include "common.hpp"

class BFGS {
public:
  // V vector, M matrix
  template <typename V, typename W, typename M>
  V solve(V x, M B, VecFun<V,W> f, GradFun<V> &Gradient, int maxiter, double tol) {
    Eigen::ConjugateGradient<M> solver;

    for (iters = 0; iters < maxiter && Gradient(x).norm() > tol; ++iters) {
      //TODO set solver parameters
      solver.compute(B);
      if (solver.info() != Eigen::Success) {
        std::cerr << "SOLVER ERROR, ABORTING" << std::endl;
        // TODO throw error
      }

      V p = solver.solve(-Gradient(x));
      double alpha = 1;
      
      for (int i = 0; i < armijo_max_iter; ++i) {
        if (f(x + alpha * p) <=
            f(x) + c1 * alpha * Gradient(x).transpose() * p) 
          break;
        alpha*= rho;
      }

      V s = alpha * p;
      V x_next = x + s;

      V y = Gradient(x_next) - Gradient(x);

      M b_prod = B * s;
      B = B + (y * y.transpose()) / (y.transpose() * s) -
          (b_prod * b_prod.transpose()) / (s.transpose() * B * s);
      x = x_next;
    }

    return x;
  }

  int getIters() {
    return iters;
  }

private:
  double c1 = 1e-4;
  double c2 = 0.9;
  double rho = 0.5;
  double armijo_max_iter = 20;
  int iters = 0;
};

