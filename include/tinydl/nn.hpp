// tinydl layers. Provided by the course staff. Do NOT modify.
// Constructors (parameter shapes and initialisation) are given in src/nn.cpp;
// you implement the forward() methods by composing ops.
#pragma once

#include <utility>
#include <vector>

#include "tinydl/ops.hpp"
#include "tinydl/tensor.hpp"

namespace tdl {

class Module {
public:
    virtual ~Module() = default;
    virtual std::vector<Tensor> parameters() { return {}; }
};

// y = x W + b.  x: [N, in] -> [N, out].  W: [in, out], b: [out].
class Linear : public Module {
public:
    Linear(int in_features, int out_features, Rng& rng);
    Tensor forward(const Tensor& x);
    std::vector<Tensor> parameters() override { return {W, b}; }
    Tensor W, b;
};

// x: [N, C, H, W] -> [N, out_channels, Ho, Wo].  W: [out, in, k, k], b: [out].
class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, int kernel, int stride, int pad, Rng& rng);
    Tensor forward(const Tensor& x);
    std::vector<Tensor> parameters() override { return {W, b}; }
    Tensor W, b;
    int stride, pad;
};

class MaxPool2d : public Module {
public:
    MaxPool2d(int kernel, int stride) : kernel(kernel), stride(stride) {}
    Tensor forward(const Tensor& x);
    int kernel, stride;
};

// One LSTM time step.
//   gates = x Wih + h Whh + b          ([N, 4H], columns in order i, f, g, o)
//   i = sigmoid(gates[:, 0H:1H])  f = sigmoid(gates[:, 1H:2H])
//   g = tanh   (gates[:, 2H:3H])  o = sigmoid(gates[:, 3H:4H])
//   c' = f * c + i * g            h' = o * tanh(c')
// x: [N, in], h and c: [N, H]. Returns {h', c'}.
class LSTMCell : public Module {
public:
    LSTMCell(int input_size, int hidden_size, Rng& rng);
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& h, const Tensor& c);
    std::vector<Tensor> parameters() override { return {Wih, Whh, b}; }
    Tensor Wih, Whh, b;  // [in, 4H], [H, 4H], [4H]
    int hidden_size;
};

}  // namespace tdl
