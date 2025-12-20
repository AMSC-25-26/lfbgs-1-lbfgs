#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <eigen3/Eigen/Eigen>

/**
 * @brief Newton minimizer (full Newton) for unconstrained optimization.
 *
 * At each iteration solves:
 *      H(x_k) p_k = -∇f(x_k)
 * then performs a line search along p_k.
 *
 * @tparam V Vector type (e.g. Eigen::VectorXd).
 * @tparam M Matrix type (e.g. Eigen::MatrixXd).
 */
template <typename V, typename M>
class Newton : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_hessFun;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  /**
   * @brief Run Newton's method with line search.
   *
   * @param x Initial guess (passed by value).
   * @param f Objective function.
   * @param Gradient Gradient function.
   * @return Approximate minimizer.
   */
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    Eigen::LDLT<M> ldlt;

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol; ++_iters) {
      M H = _hessFun(x);
      V g = Gradient(x);

      check(H.rows() == H.cols(), "Hessian must be square");
      check(H.rows() == g.size(), "Hessian/gradient size mismatch");

      ldlt.compute(H);
      check(ldlt.info() == Eigen::Success, "LDLT factorization failed");

      V p = ldlt.solve(-g);
      check(ldlt.info() == Eigen::Success, "LDLT solve failed");

      if (p.dot(g) >= 0.0)
        p = -g;

      double alpha = this->line_search(x, p, f, Gradient);

      x = x + alpha * p;
    }

    return x;
  }

private:
};
