#include <eigen3/Eigen/Eigen>
#include <functional>
#include <iostream>

#ifndef NDEBUG
  #define check(condition, message)                                   \
    do {                                                              \
      if (!condition) {                                               \
        std::cerr << "[FAILED ASSERTION]" << std::endl;               \
        std::cerr << "  Condition: " << #condition << std::endl;      \
        std::cerr << "  Message: " << (message) << std::endl;         \
        std::cerr << "  File: " << __FILE__ << ", Line: " << __LINE__ \
                  << std::endl;                                       \
        std::cerr << "  Aborting..." << std::endl;                    \
        std::abort();                                                 \
      }                                                               \
    } while (0)

#else
  #define check(condition, message) ((void)0)
#endif

template <typename T>
using GradFun = std::function<T(T)>;

template <typename T, typename W>
using VecFun = std::function<W(T)>;

template <typename V, typename M>
using HessFun = std::function<M(V)>;