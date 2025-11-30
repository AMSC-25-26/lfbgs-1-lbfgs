#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <eigen3/Eigen/Eigen>
#include <limits>

template <typename V, typename M>
class BFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  // V vector, M matrix
  V solve(V x, M B, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    Eigen::ConjugateGradient<M> solver;

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol;
         ++_iters) {
      solver.compute(B);
      check(solver.info() == Eigen::Success, "conjugate gradient solver error");

      V p = solver.solve(-Gradient(x));
      double alpha = 1.0;

      alpha = this->line_search(x, p, f, Gradient);

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

private:
};
