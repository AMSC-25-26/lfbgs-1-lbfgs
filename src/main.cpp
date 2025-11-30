#include "bfgs.hpp"
#include "lbfgs.hpp"
#include "minimizer_base.hpp"
#include <eigen3/Eigen/Eigen>
#include <memory>

// Trying to automate derivative computation
#include <autodiff/reverse/var.hpp>       //we could also use /forward/ but reverse should be more efficient for gradients like O(1) vs O(n) (sempre se fossi attento a quella lezione di naml)
#include <autodiff/reverse/var/eigen.hpp> //reverse is also easier to write

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
  VecFun<Vec, double> f = [](Vec v) {
    int n = v.size();

    double result = 10 * n;
    for (int i = 0; i < n; ++i) {
      result += v(i) * v(i) - 10 * std::cos(2 * M_PI * v(i));
    }
    return result;
  };

  auto f_var = [](Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> x) {
    int n = x.size();
    autodiff::var result = 10 * n;
    for (int i = 0; i < n; ++i) {
      result += x(i) * x(i) - 10 * cos(2 * M_PI * x(i));
    }
    return result;
  };

  GradFun<Vec> grad = [f_var](Vec v) {
    Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> x(v.size());
    for (int i = 0; i < v.size(); ++i)
      x(i) = v(i);
  https: // github.com/autodiff/autodiff.git
    autodiff::var y = f_var(x);
    Vec dy_dv = autodiff::gradient(y, x);

    return dy_dv;
  };

  int n = 15;
  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = 0.25;

  Mat m(n, n);
  m.setIdentity();

  using minimizerPtr = std::unique_ptr<MinimizerBase<Vec, Mat>>;

  minimizerPtr bfgs = std::make_unique<BFGS<Vec, Mat>>();
  minimizerPtr lbfgs = std::make_unique<LBFGS<Vec, Mat>>();

  auto assert_solver = [&](minimizerPtr &solver, std::string name) {
    solver->setMaxIterations(4000);
    //    solver->setTolerance(1.e-12);

    Vec result = solver->solve(v, m, f, grad);
    std::cout << "========" << name << "========" << std::endl;
    std::cout << "computed result: " << std::endl
              << result << std::endl
              << std::endl;

    std::cout << "\nFunction value: " << f(result) << std::endl;
    std::cout << "iterations: " << solver->iterations() << std::endl;
    std::cout << "tolerance: " << solver->tolerance() << std::endl;
    std::cout << "error norm: " << grad(result).norm() << std::endl;
    std::cout << std::endl;
  };

  assert_solver(bfgs, "BFGS");
  assert_solver(lbfgs, "LBFGS");

  return 0;
}
