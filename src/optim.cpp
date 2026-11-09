// Optimisers. STUDENT FILE (clip_grad_norm is given).
#include "tinydl/optim.hpp"

#include <cmath>
#include <vector>

namespace tdl {

void Optimizer::zero_grad() {
    todo("Optimizer::zero_grad");  // TODO: implement
}

SGD::SGD(std::vector<Tensor> params, float lr, float momentum)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum) {
    todo("SGD::SGD");  // TODO: implement
}

void SGD::step() {
    todo("SGD::step");  // TODO: implement
}

float clip_grad_norm(std::vector<Tensor>& params, float max_norm) {
    double sq = 0.0;
    for (const Tensor& p : params) {
        const float* g = p.grad();
        if (!g) continue;
        for (int64_t i = 0; i < p.numel(); ++i) sq += static_cast<double>(g[i]) * g[i];
    }
    const float norm = static_cast<float>(std::sqrt(sq));
    if (norm > max_norm && norm > 0.f) {
        const float s = max_norm / norm;
        for (Tensor& p : params) {
            if (!p.has_grad()) continue;
            float* g = p.grad();
            for (int64_t i = 0; i < p.numel(); ++i) g[i] *= s;
        }
    }
    return norm;
}

}  // namespace tdl
