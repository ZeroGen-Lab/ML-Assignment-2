// tinydl optimisers. Provided by the course staff. Do NOT modify.
#pragma once

#include <vector>

#include "tinydl/tensor.hpp"

namespace tdl {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;
    void zero_grad();         // set every parameter gradient to zero
    virtual void step() = 0;  // update parameters from their gradients

protected:
    std::vector<Tensor> params_;
};

// Parameters with no gradient yet (never used in the graph) are skipped by step().

// v = momentum * v + g;  p -= lr * v   (v starts at zero)
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, float lr, float momentum = 0.f);
    void step() override;

private:
    float lr_, momentum_;
    std::vector<std::vector<float>> velocity_;
};

// Scales all gradients so their global L2 norm is at most max_norm.
// Returns the norm before clipping. Provided by the course staff.
float clip_grad_norm(std::vector<Tensor>& params, float max_norm);

}  // namespace tdl
