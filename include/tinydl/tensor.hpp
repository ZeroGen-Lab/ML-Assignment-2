// tinydl: a tiny deep learning framework for Assignment 2.
// This header is provided by the course staff. Do NOT modify it: the grader
// compiles your .cpp files against its own copy of every header.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdl {

struct Node;

// Storage shared by every Tensor handle that refers to the same tensor.
struct TensorImpl {
    std::vector<float> data;
    std::vector<float> grad;  // empty until a gradient is first accumulated
    std::vector<int> shape;
    bool requires_grad = false;
    std::shared_ptr<Node> grad_fn;  // the op that produced this tensor (null for leaves)
    // Frees long graphs iteratively (a recursive chain of shared_ptr releases
    // would overflow the stack for graphs with ~100k nodes).
    ~TensorImpl();
};

// A Tensor is a cheap handle (shared pointer) to a TensorImpl.
// Copying a Tensor does NOT copy the data.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> p) : impl(std::move(p)) {}

    static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false);
    static Tensor full(const std::vector<int>& shape, float value, bool requires_grad = false);
    static Tensor from_vector(std::vector<float> data, const std::vector<int>& shape,
                              bool requires_grad = false);

    bool defined() const { return impl != nullptr; }
    const std::vector<int>& shape() const;
    int ndim() const;
    int dim(int i) const;  // negative i counts from the end
    int64_t numel() const;

    float* data();
    const float* data() const;
    // Gradient buffer. grad() allocates it (filled with zeros) on first use.
    float* grad();
    const float* grad() const;  // nullptr if no gradient yet
    bool has_grad() const;
    void zero_grad();

    bool requires_grad() const;
    void set_requires_grad(bool value);
    const std::shared_ptr<Node>& grad_fn() const;

    float item() const;  // value of a one-element tensor

    // Reverse-mode automatic differentiation from this (one-element) tensor.
    // Implemented by the student in src/autograd.cpp.
    void backward();

    std::shared_ptr<TensorImpl> impl;
};

// One recorded operation in the computation graph.
struct Node {
    std::vector<Tensor> inputs;  // tensors the op read (graph edges)
    virtual ~Node() = default;
    virtual const char* name() const = 0;
    // grad_out holds d(loss)/d(output), one value per output element.
    // Implementations add d(loss)/d(inputs[i]) into inputs[i] with accumulate_grad.
    virtual void backward(const std::vector<float>& grad_out) = 0;
};

// ---- helpers for writing ops ----

// Builds the result of an op. If gradients are enabled and any of
// node->inputs requires grad, the result requires grad and records node as
// its grad_fn; otherwise the node is dropped. node->inputs must be filled first.
Tensor make_output(std::vector<float> data, std::vector<int> shape, std::shared_ptr<Node> node);

// grad(t) += g (t.numel() values). Does nothing if t does not require grad.
void accumulate_grad(Tensor& t, const float* g);

// Gradient recording switch (used for evaluation).
bool grad_enabled();
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

std::string shape_str(const std::vector<int>& shape);
int64_t shape_numel(const std::vector<int>& shape);

// Throws std::runtime_error(msg) when cond is false.
inline void check(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Marks code the student still has to write.
[[noreturn]] void todo(const char* what);

// Deterministic random numbers (same sequence on every platform/compiler).
class Rng {
public:
    explicit Rng(uint64_t seed = 42);
    uint64_t next_u64();
    float uniform();                    // [0, 1)
    float uniform(float lo, float hi);  // [lo, hi)
    float normal();                     // mean 0, std 1
    int randint(int n);                 // [0, n)

private:
    uint64_t state_;
    bool has_spare_ = false;
    float spare_ = 0.f;
};

}  // namespace tdl
