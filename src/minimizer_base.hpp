#pragma once

#include "common.hpp"
#include <eigen3/Eigen/Eigen>

template <typename V, typename M>
class MinimizerBase {
public:
  virtual ~MinimizerBase() = default;

  int iterations() const noexcept {
    return _iters;
  }

  double tolerance() const noexcept {
    return _tol;
  }

  void setMaxIterations(int max_iters) noexcept {
    _max_iters = max_iters;
  }

  void setTolerance(double tol) noexcept { _tol = tol; }

  virtual V solve(V x, M b, VecFun<V, double> &f, GradFun<V> &Gradient) = 0;

protected:
  unsigned int _max_iters = 1000;
  unsigned int _iters = 0;
  double _tol = 1.e-10;
  double armijo_max_iter = 20;
  double max_line_iters = 50;
  size_t m = 5;
  double alpha_wolfe = 1e-3;
  double c1 = 1e-4;
  double c2 = 0.9;
  double rho = 0.5;

  double line_search(V x, V p, VecFun<V, double> &f, GradFun<V> &Gradient) {
    
    double grad_f_old = Gradient(x).dot(p);

    double inf = std::numeric_limits<double>::infinity();
    double alpha_min = 0.0;
    double alpha_max = inf;

    double alpha = 1.0;

    for (int i = 0; i < max_line_iters; ++i) {
      V x_new = x + alpha * p;
      double f_new = f(x_new);
      double grad_f_new = Gradient(x_new).dot(p);

      for (int armijo_iter = 0; armijo_iter < armijo_max_iter; ++armijo_iter) {
        if (f_new > f(x) + c1 * alpha * grad_f_old) {
          alpha_max = alpha;
          alpha = rho * (alpha_min + alpha_max);
        } else {
          break;
        }
      }

      if (grad_f_new >= c2 * grad_f_old) {
        if (alpha_max < inf) {
          alpha_min = alpha;
          alpha_max = rho * (alpha_min + alpha_max);
        } else {
          alpha_min = alpha;
          alpha = 5 * alpha;
        }
      }
    }
    return alpha;
  }
};
