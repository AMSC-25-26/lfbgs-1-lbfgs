#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <eigen3/Eigen/Eigen>

template <typename V, typename M>
class LBFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
  using Base::alpha_wolfe;
  using Base::m;

public:
  // V vector, M matrix

  // TODO: being able to handle a hessian initial guess (now B is not used)
  V solve(V x, M b, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    std::vector<V> s_list;
    std::vector<V> y_list;
    std::vector<double> rho_list;

    V grad = Gradient(x);
    V p = -grad;
    V x_new = x;

    V s = x; // initializatione a caso giusto per la dimensione
    V y = grad;

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol;
         ++_iters) {

      grad = Gradient(x);
      p = compute_direction(grad, s_list, y_list, rho_list);

      alpha_wolfe = this->line_search(x, p, f, Gradient);

      x_new = x + alpha_wolfe * p;
      V s = x_new - x;
      V y = Gradient(x_new) - grad;

      x = x_new;

      double rho = 1.0 / y.dot(s);
      s_list.push_back(s);
      y_list.push_back(y);
      rho_list.push_back(rho);

      if (s_list.size() > m) {
        s_list.erase(s_list.begin());
        y_list.erase(y_list.begin());
        rho_list.erase(rho_list.begin());
      }
    }

    return x;
  }

  V compute_direction(V grad, std::vector<V> &s_list, std::vector<V> &y_list,
                      std::vector<double> &rho_list) {
    if (s_list.size() == 0) {
      return -grad;
    }

    V z = V::Zero(grad.size()); // random initialization
    V q = grad;
    std::vector<double> alpha_list(s_list.size());

    double gamma = 1.0;

    for (int i = s_list.size() - 1; i >= 0; --i) {
      alpha_list[i] = rho_list[i] * s_list[i].dot(q);
      q -= alpha_list[i] * y_list[i];
    }

    gamma = s_list.back().dot(y_list.back()) / y_list.back().dot(y_list.back());

    M H0 = gamma * M::Identity(grad.size(), grad.size());

    z = H0 * q;

    for (size_t i = 0; i < s_list.size(); ++i) {
      double beta = rho_list[i] * y_list[i].dot(z);
      z += s_list[i] * (alpha_list[i] - beta);
    }

    z = -z;
    return z;
  }

private:
};
