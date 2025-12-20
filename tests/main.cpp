#include "test.hpp"
#include <iostream>

#include <eigen3/unsupported/Eigen/IterativeSolvers>

#include "../src/bfgs.hpp"
#include "../src/common.hpp"
#include "../src/lbfgs.hpp"
#include "../src/newton.hpp"

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

using minimizerPtr = std::shared_ptr<MinimizerBase<Vec, Mat>>;

void test_rastrigin(minimizerPtr &solver) {
  
  VecFun<Vec, double> f = [](Vec v) {
    double val = 0.0;
    double A = 10.0;
    int n = v.size();
    for (int i = 0; i < n; ++i) {
      val += (v(i) * v(i)) - (A * std::cos(2.0 * M_PI * v(i)));
    }
    return A * n + val;
  };

  GradFun<Vec> grad = [](Vec v) {
    int n = v.size();
    Vec g(n);
    double A = 10.0;
    for (int i = 0; i < n; ++i) {
      g(i) = 2.0 * v(i) + 2.0 * M_PI * A * std::sin(2.0 * M_PI * v(i));
    }
    return g;
  };

  HessFun<Vec, Mat> hess = [](const Vec &v) {
    const int n = static_cast<int>(v.size());
    Mat H = Mat::Zero(n, n);
    const double A = 10.0;

    for (int i = 0; i < n; ++i) {
      H(i, i) = 2.0 + 4.0 * M_PI * M_PI * A * std::cos(2.0 * M_PI * v(i));
    }
    return H;
  };

  int n = 500;

  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = (i % 2 == 0) ? 4.0 : -4.0;

  Mat m(n, n);
  m.setIdentity();

  solver->setMaxIterations(5000);
  solver->setTolerance(1.e-9);
  solver->setInitialHessian(m);
  solver->setHessian(hess);

  Vec result = solver->solve(v, f, grad);

  check((grad(result).norm() <= 1.e-8), "should converge on rastrigin function");

  Vec expected_min(n);
  expected_min.setZero();
  // check(((result - expected_min).norm() <= 1.e-6), "solution should be close to the global minimum [0, 0, ...]");
}

void test_rosenbrock(minimizerPtr &solver) {

  VecFun<Vec, double> f = [](Vec v) {
    double val = 0.0;
    int n = v.size();
    for (int i = 0; i < n - 1; ++i) {
      double term1 = v(i + 1) - v(i) * v(i);
      double term2 = 1.0 - v(i);
      val += 100.0 * term1 * term1 + term2 * term2;
    }
    return val;
  };

  GradFun<Vec> grad = [](Vec v) {
    int n = v.size();
    Vec g(n);
    g.setZero();

    if (n > 1) {
      g(0) = -2.0 * (1.0 - v(0)) - 400.0 * v(0) * (v(1) - v(0) * v(0));
    } else {
      g(0) = -2.0 * (1.0 - v(0));
    }

    for (int i = 1; i < n - 1; ++i) {
      double term1 = -2.0 * (1.0 - v(i));
      double term2 = -400.0 * v(i) * (v(i + 1) - v(i) * v(i));
      double term3 = 200.0 * (v(i) - v(i - 1) * v(i - 1));
      g(i) = term1 + term2 + term3;
    }

    if (n > 1) {
      g(n - 1) = 200.0 * (v(n - 1) - v(n - 2) * v(n - 2));
    }

    return g;
  };

  HessFun<Vec, Mat> hess = [](Vec v) {
    const int n = v.size();
    Mat H(n, n);
    H.setZero();
    if (n == 1) {
      H(0, 0) = 2.0;
      return H;
    }

    H(0, 0) = 2.0 - 400.0 * (v(1) - 3.0 * v(0) * v(0));
    H(0, 1) = -400.0 * v(0);
    H(1, 0) = H(0, 1);
    for (int i = 1; i < n - 1; ++i) {
      if (i < n - 1)
        H(i, i) = 202.0 - 400.0 * (v(i + 1) - 3.0 * v(i) * v(i));

      if (i + 1 < n) {
        H(i, i + 1) = -400.0 * v(i);
        H(i + 1, i) = H(i, i + 1);
      }
    }
    H(n - 1, n - 1) = 200.0;

    return H;
  };

  int n = 4;

  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = (i % 2 == 0) ? -1.2 : 1.0;

  Mat m(n, n);
  m.setIdentity();

  solver->setMaxIterations(4000);
  solver->setTolerance(1.e-12);
  solver->setInitialHessian(m);
  solver->setHessian(hess);

  Vec result = solver->solve(v, f, grad);

  check((grad(result).norm() <= 1.e-10), "should converge on rosenbrock function");

  Vec expected_min(n);
  expected_min.setOnes();
  check(((result - expected_min).norm() <= 1.e-8), "solution should be close to the global minimum [1, 1, ...]");
}

void test_ackley(minimizerPtr &solver) {

  VecFun<Vec, double> f = [](Vec v) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int n = v.size();

    for (int i = 0; i < n; ++i) {
      sum1 += v(i) * v(i);
      sum2 += std::cos(2.0 * M_PI * v(i));
    }

    return -20.0 * std::exp(-0.2 * std::sqrt(sum1 / n)) -
           std::exp(sum2 / n) + 20.0 + std::exp(1.0);
  };

  GradFun<Vec> grad = [](Vec v) {
    int n = v.size();
    Vec g(n);

    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
      sum1 += v(i) * v(i);
      sum2 += std::cos(2.0 * M_PI * v(i));
    }

    double term_exp_cos = std::exp(sum2 / n);
    double term_exp_sqrt = std::exp(-0.2 * std::sqrt(sum1 / n));

    for (int i = 0; i < n; ++i) {
      double grad_sqrt = (v(i) / (n * std::sqrt(sum1 / n)));
      double grad_exp1 = 4.0 * term_exp_sqrt * grad_sqrt;
      double grad_exp2 = (2.0 * M_PI / n) * term_exp_cos * std::sin(2.0 * M_PI * v(i));

      g(i) = grad_exp1 + grad_exp2;
    }
    return g;
  };

  HessFun<Vec, Mat> hess = [](Vec v) {
    const int n = v.size();
    Mat H(n, n);
    H.setZero();
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
      sum1 += v(i) * v(i);
      sum2 += std::cos(2.0 * M_PI * v(i));
    }
    const double eps = 1e-12;
    const double s = sum1 / n;
    const double r = std::sqrt(s + eps);
    const double E1 = std::exp(-0.2 * r);
    const double E2 = std::exp(sum2 / n);
    const double twoPi = 2.0 * M_PI;
    const double coef_dE1 = -0.2 * E1;
    const double coef_grad2 = twoPi / n;
    const double coef_dE2 = -(twoPi / n) * E2;
    std::vector<double> s2(n), c2(n);
    for (int i = 0; i < n; ++i) {
      s2[i] = std::sin(twoPi * v(i));
      c2[i] = std::cos(twoPi * v(i));
    }
    for (int i = 0; i < n; ++i) {
      const double ai = v(i) / (n * r);
      for (int j = 0; j < n; ++j) {
        const double aj = v(j) / (n * r);
        const double dE1_dxj = coef_dE1 * aj;
        double dai_dxj = 0.0;
        if (i == j)
          dai_dxj += (1.0 / n) * (1.0 / r);
        dai_dxj += -(1.0 / n) * (v(i) * v(j)) / (n * r * r * r);
        const double Hess1 = 4.0 * (dE1_dxj * ai + E1 * dai_dxj);
        const double dE2_dxj = coef_dE2 * s2[j];
        double Hess2 = coef_grad2 * (dE2_dxj * s2[i]);
        if (i == j)
          Hess2 += coef_grad2 * (E2 * twoPi * c2[i]);
        H(i, j) = Hess1 + Hess2;
      }
    }
    return H;
  };

  int n = 3;

  Vec v(n);
  v << 10.0, -5.0, 1.0;

  Mat m(n, n);
  m.setIdentity();

  solver->setMaxIterations(4000);
  solver->setTolerance(1.e-10);
  solver->setInitialHessian(m);
  solver->setHessian(hess);

  Vec result = solver->solve(v, f, grad);

  check((grad(result).norm() <= 1.e-9), "should converge on ackley function");
}

int main() {
  minimizerPtr bfgs = std::make_shared<BFGS<Vec, Mat>>();
  minimizerPtr lbfgs = std::make_shared<LBFGS<Vec, Mat>>();
  minimizerPtr newton = std::make_shared<Newton<Vec, Mat>>();

  using GMRES_Solver = Eigen::GMRES<Mat>;
  GMRES_Solver solver = GMRES_Solver();
  solver.setTolerance(1.e-12);
  solver.setMaxIterations(10000);
  auto bfgs_gmres = std::make_shared<BFGS<Vec, Mat, GMRES_Solver>>((solver));

  auto suite = Tests::TestSuite<Vec, Mat>();

  suite.addImplementation(bfgs, "BFGS");
  suite.addImplementation(lbfgs, "LBFGS");
  suite.addImplementation(bfgs_gmres, "BFGS + GMRES");
  suite.addImplementation(newton, "Newton");

  suite.addTest("rosenbrock function", test_rosenbrock);
  suite.addTest("ackley function", test_ackley);
  suite.addTest("rastrigin function", test_rastrigin);

  suite.runTests();
}
