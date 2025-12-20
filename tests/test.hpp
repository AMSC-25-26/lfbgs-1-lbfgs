#include "../src/minimizer_base.hpp"
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace Tests {

/**
 * @brief Generic test suite for minimizer implementations.
 *
 * This class allows you to:
 *  - register multiple minimizer implementations (e.g. BFGS, L-BFGS, ...)
 *  - register multiple tests (each test being a function using a minimizer)
 *  - run all tests on all implementations, collecting timing and iteration info.
 *
 * @tparam V Vector type used by the minimizers (e.g. Eigen::VectorXd).
 * @tparam M Matrix type used by the minimizers (e.g. Eigen::MatrixXd).
 */
template <typename V, typename M>
class TestSuite {
  /// Shared pointer to a generic minimizer implementation.
  using minimizerPtr = std::shared_ptr<MinimizerBase<V, M>>;
  /// Type of a test function: takes a minimizer instance by reference.
  using testFunction = std::function<void(minimizerPtr &)>;

public:
  /**
   * @brief Construct an empty test suite.
   *
   * Initializes the internal containers used to store implementations
   * and tests.
   */
  TestSuite() {
    impls = std::map<std::string, minimizerPtr>();
    tests = std::vector<std::pair<std::string, testFunction>>();
  }

  /**
   * @brief Register a minimizer implementation in the suite.
   *
   * The implementation is associated with a human-readable name and will
   * be used for all registered tests when @ref runTests is called.
   *
   * @param ptr  Shared pointer to a minimizer instance.
   * @param name Identifier for this implementation (used in output).
   */
  void addImplementation(minimizerPtr ptr, std::string name) {
    impls[name] = ptr;
  }

  /**
   * @brief Register a test in the suite.
   *
   * A test is a callable object that receives a reference to a minimizer
   * implementation and typically:
   *  - sets up the optimization problem,
   *  - calls solve(...),
   *  - checks/prints results.
   *
   * @param name Descriptive name of the test (used in output).
   * @param fun  Test function to be executed on each implementation.
   */
  void addTest(std::string name, testFunction fun) {
    tests.push_back(std::make_pair(name, fun));
  }

  /**
   * @brief Run all registered tests on all registered implementations.
   *
   * For each test, this method:
   *  - prints a header with the test name,
   *  - runs the test on every registered implementation,
   *  - measures wall-clock time using std::chrono::steady_clock,
   *  - prints elapsed time, number of iterations, and tolerance used
   *    by the minimizer.
   */
  void runTests() {
    for (std::pair<std::string, testFunction> &test : tests) {
      std::cout << "======================"
                   "RUNNING TEST:"
                << test.first << "======================" << std::endl;

      for (auto &impl : impls) {
        std::cout << "  implementation: " << impl.first << std::endl;

        auto before = std::chrono::steady_clock::now();

        // Execute the test on the current implementation
        test.second(impl.second);

        auto after = std::chrono::steady_clock::now();
        auto delta_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after - before)
                .count();

        std::cout << "\t time elapsed: " << delta_us << " us" << std::endl;
        std::cout << "\t iterations:   " << impl.second->iterations()
                  << std::endl;
        std::cout << "\t tolerance:    " << impl.second->tolerance()
                  << std::endl;
      }
    }
  }

private:
  /// Map from implementation name to minimizer instance.
  std::map<std::string, minimizerPtr> impls;

  /// List of (test name, test function) pairs.
  std::vector<std::pair<std::string, testFunction>> tests;
};

} // namespace Tests
