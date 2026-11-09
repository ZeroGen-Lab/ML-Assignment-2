// tinydl ops. Provided by the course staff. Do NOT modify.
// Every op computes its forward result and, when gradients are needed,
// records a Node whose backward() propagates gradients to its inputs.
// Implement them in src/ops.cpp (add() is given as a complete example).
#pragma once

#include <vector>

#include "tinydl/tensor.hpp"

namespace tdl {

// ---- element-wise (a and b must have the same shape) ----
Tensor add(const Tensor& a, const Tensor& b);  // a + b   (given as an example)
Tensor mul(const Tensor& a, const Tensor& b);  // a * b

// ---- activations (element-wise, any shape) ----
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);

// ---- linear algebra / shapes ----
Tensor matmul(const Tensor& a, const Tensor& b);    // [N, K] x [K, M] -> [N, M]
Tensor add_bias(const Tensor& x, const Tensor& b);  // [N, M] + [M] (b added to every row)
Tensor reshape(const Tensor& x, const std::vector<int>& shape);  // same numel, row-major order
Tensor slice_cols(const Tensor& x, int start, int len);  // [N, M] -> columns [start, start+len)

// ---- loss ----
// Mean over the batch of -log softmax(logits)[i, labels[i]].
// logits: [N, C]; labels: N class indices in [0, C). Result shape [1].
// Compute it in a numerically stable way (subtract the row max).
Tensor cross_entropy(const Tensor& logits, const std::vector<int>& labels);

// ---- convolution / pooling (NCHW layout) ----
// x: [N, C, H, W], w: [O, C, K, K], b: [O]
// out: [N, O, Ho, Wo], Ho = (H + 2*pad - K) / stride + 1 (same for Wo).
// Zero padding. Hint: im2col turns each image into a matrix so the
// convolution becomes one matrix product.
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int pad);
// x: [N, C, H, W] -> [N, C, Ho, Wo], Ho = (H - k) / stride + 1. No padding.
Tensor max_pool2d(const Tensor& x, int k, int stride);

}  // namespace tdl
