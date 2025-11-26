#include "bfgs.hpp"
#include <eigen3/Eigen/Eigen>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
  VecFun<Vec, double> f = [](Vec v) {
    double x1 = v(0);
    double x2 = v(1);
    return 100 * std::pow((x2 - x1 * x2), 2.0) +
      std::pow((1 - x1), 2.0);
  };

  GradFun<Vec> grad = [](Vec v) {
    double x1 = v(0);
    double x2 = v(1);

    Vec res = Vec::Zero(v.size());
    //corrected the derivative I guess
    //res(0) = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1); 
    res(0) = -2 * (1 - x1) * (100 * x2 * x2 + 1);
    //res(1) = 200 * (x2 - x1 * x1);
    res(1) = 200 * x2 * (x1 - 1)  * (x1 - 1);
    return res;
  };

  Vec v(2);
  v(0) = -1.2;
  v(1) = 1.0;
  double tol = 1.e-6;
  int maxiter = 4000;

  Mat m(2, 2);
  m.setIdentity();

  auto solver = BFGS();
  Vec res = solver.solve(v, m, f, grad, maxiter, tol);

  std::cout << res << std::endl;
  std::cout << "iterations: " << solver.getIters() << std::endl;
  return 0;
}
