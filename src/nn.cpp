// Layers. STUDENT FILE: the constructors are given; implement forward().
#include "tinydl/nn.hpp"

#include <cmath>
#include <vector>

namespace tdl {

namespace {
Tensor uniform_param(const std::vector<int>& shape, float bound, Rng& rng) {
    std::vector<float> v(shape_numel(shape));
    for (float& x : v) x = rng.uniform(-bound, bound);
    return Tensor::from_vector(std::move(v), shape, /*requires_grad=*/true);
}
}  // namespace

// ---------------------------------------------------------------- Linear
Linear::Linear(int in_features, int out_features, Rng& rng) {
    W = uniform_param({in_features, out_features}, 1.f / std::sqrt(static_cast<float>(in_features)), rng);
    b = Tensor::zeros({out_features}, /*requires_grad=*/true);
}

Tensor Linear::forward(const Tensor& x) {
    todo("Linear::forward");  // TODO: implement
}

// ---------------------------------------------------------------- Conv2d
Conv2d::Conv2d(int in_channels, int out_channels, int kernel, int stride, int pad, Rng& rng)
    : stride(stride), pad(pad) {
    const float fan_in = static_cast<float>(in_channels * kernel * kernel);
    W = uniform_param({out_channels, in_channels, kernel, kernel}, std::sqrt(6.f / fan_in), rng);  // He uniform
    b = Tensor::zeros({out_channels}, /*requires_grad=*/true);
}

Tensor Conv2d::forward(const Tensor& x) {
    todo("Conv2d::forward");  // TODO: implement
}

// ---------------------------------------------------------------- MaxPool2d
Tensor MaxPool2d::forward(const Tensor& x) {
    todo("MaxPool2d::forward");  // TODO: implement
}

// ---------------------------------------------------------------- LSTMCell
LSTMCell::LSTMCell(int input_size, int hidden_size, Rng& rng) : hidden_size(hidden_size) {
    const float bound = 1.f / std::sqrt(static_cast<float>(hidden_size));
    Wih = uniform_param({input_size, 4 * hidden_size}, bound, rng);
    Whh = uniform_param({hidden_size, 4 * hidden_size}, bound, rng);
    std::vector<float> bias(4 * hidden_size, 0.f);
    for (int j = hidden_size; j < 2 * hidden_size; ++j) bias[j] = 1.f;  // forget gate starts open
    b = Tensor::from_vector(std::move(bias), {4 * hidden_size}, /*requires_grad=*/true);
}

std::pair<Tensor, Tensor> LSTMCell::forward(const Tensor& x, const Tensor& h, const Tensor& c) {
    todo("LSTMCell::forward");  // TODO: implement
}

}  // namespace tdl
