// Autograd engine. STUDENT FILE.
#include <unordered_set>
#include <utility>
#include <vector>

#include "tinydl/tensor.hpp"

namespace tdl {

// Reverse-mode automatic differentiation.
//
// 1. This tensor must hold exactly one element; its gradient is seeded with 1
//    (added to whatever gradient it already has).
// 2. Find every tensor this one depends on by following grad_fn()->inputs, and
//    order them topologically: a tensor must come after every tensor computed
//    from it. Use an explicit stack, not recursion: the hidden tests and the
//    LSTM (unrolled over 128 time steps) build graphs over 100,000 nodes deep,
//    which overflow the call stack even when the public tests still pass.
// 3. Visit the tensors from this one backwards. For each tensor that has a
//    grad_fn and a gradient, call grad_fn->backward(gradient). By then all of
//    its gradient has been accumulated, because everything computed from it
//    has already been visited.
// Leaf tensors keep accumulating across calls; Optimizer::zero_grad() resets them.
void Tensor::backward() {
    todo("Tensor::backward");  // TODO: implement
}

}  // namespace tdl
