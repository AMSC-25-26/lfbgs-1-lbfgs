#include <eigen3/Eigen/Eigen>
#include <functional>

template<typename T>
using GradFun = std::function<T(T)>;

template <typename T, typename W>
using VecFun = std::function<W(T)>;
