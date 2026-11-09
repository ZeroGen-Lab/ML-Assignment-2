// Ops: forward computation + backward Node for each op. STUDENT FILE.
//
// Pattern (see add() below, which is complete):
//   * the op function checks shapes, computes the output values, creates a
//     Node, stores the inputs in node->inputs, and returns
//     make_output(values, shape, node);
//   * Node::backward(grad_out) receives d(loss)/d(output) and adds
//     d(loss)/d(input) into each input with accumulate_grad(inputs[i], ...).
// You may add members to the Node structs and helper functions in this file.
// Do NOT store the output Tensor inside its own Node (that creates a reference
// cycle and leaks memory); copy the values you need into a std::vector instead.
#include "tinydl/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace tdl {

namespace {
void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    check(a.defined() && b.defined(), std::string(op) + ": undefined tensor");
    check(a.shape() == b.shape(),
          std::string(op) + ": shape mismatch " + shape_str(a.shape()) + " vs " + shape_str(b.shape()));
}
}  // namespace

// =====================================================================
// add: complete example
// =====================================================================
struct AddBackward : Node {
    const char* name() const override { return "AddBackward"; }
    void backward(const std::vector<float>& g) override {
        // d(a + b)/da = 1 and d(a + b)/db = 1, so both inputs receive grad_out.
        accumulate_grad(inputs[0], g.data());
        accumulate_grad(inputs[1], g.data());
    }
};

Tensor add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b, "add");
    const int64_t n = a.numel();
    std::vector<float> out(n);
    const float* pa = a.data();
    const float* pb = b.data();
    for (int64_t i = 0; i < n; ++i) out[i] = pa[i] + pb[i];
    auto node = std::make_shared<AddBackward>();
    node->inputs = {a, b};
    return make_output(std::move(out), a.shape(), node);
}

// =====================================================================
// mul
// =====================================================================
struct MulBackward : Node {
    const char* name() const override { return "MulBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("MulBackward::backward");  // TODO: implement
    }
};

Tensor mul(const Tensor& a, const Tensor& b) {
    todo("mul");  // TODO: implement
}

// =====================================================================
// activations
// =====================================================================
struct ReluBackward : Node {
    const char* name() const override { return "ReluBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("ReluBackward::backward");  // TODO: implement
    }
};

Tensor relu(const Tensor& x) {
    todo("relu");  // TODO: implement
}

struct SigmoidBackward : Node {
    std::vector<float> y;  // forward output values
    const char* name() const override { return "SigmoidBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("SigmoidBackward::backward");  // TODO: implement
    }
};

Tensor sigmoid(const Tensor& x) {
    todo("sigmoid");  // TODO: implement
}

struct TanhBackward : Node {
    std::vector<float> y;  // forward output values
    const char* name() const override { return "TanhBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("TanhBackward::backward");  // TODO: implement
    }
};

Tensor tanh(const Tensor& x) {
    todo("tanh");  // TODO: implement
}

// =====================================================================
// matmul, add_bias, reshape, slice_cols
// =====================================================================
struct MatMulBackward : Node {
    const char* name() const override { return "MatMulBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("MatMulBackward::backward");  // TODO: implement
    }
};

Tensor matmul(const Tensor& a, const Tensor& b) {
    todo("matmul");  // TODO: implement
}

struct AddBiasBackward : Node {
    const char* name() const override { return "AddBiasBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("AddBiasBackward::backward");  // TODO: implement
    }
};

Tensor add_bias(const Tensor& x, const Tensor& b) {
    todo("add_bias");  // TODO: implement
}

struct ReshapeBackward : Node {
    const char* name() const override { return "ReshapeBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("ReshapeBackward::backward");  // TODO: implement
    }
};

Tensor reshape(const Tensor& x, const std::vector<int>& shape) {
    todo("reshape");  // TODO: implement
}

struct SliceColsBackward : Node {
    int start = 0, len = 0;
    const char* name() const override { return "SliceColsBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("SliceColsBackward::backward");  // TODO: implement
    }
};

Tensor slice_cols(const Tensor& x, int start, int len) {
    todo("slice_cols");  // TODO: implement
}

// =====================================================================
// cross entropy
// =====================================================================
struct CrossEntropyBackward : Node {
    std::vector<float> probs;  // softmax(logits), [N, C]
    std::vector<int> labels;
    const char* name() const override { return "CrossEntropyBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("CrossEntropyBackward::backward");  // TODO: implement
    }
};

Tensor cross_entropy(const Tensor& logits, const std::vector<int>& labels) {
    todo("cross_entropy");  // TODO: implement
}

// =====================================================================
// conv2d (im2col) and max_pool2d
// =====================================================================
// TODO (conv helpers): helper functions you need can go here.

struct Conv2dBackward : Node {
    int stride = 1, pad = 0;
    const char* name() const override { return "Conv2dBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("Conv2dBackward::backward");  // TODO: implement
    }
};

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int pad) {
    todo("conv2d");  // TODO: implement
}

struct MaxPool2dBackward : Node {
    std::vector<int64_t> argmax;  // for each output element, flat index of the max in x
    const char* name() const override { return "MaxPool2dBackward"; }
    void backward(const std::vector<float>& g) override {
        todo("MaxPool2dBackward::backward");  // TODO: implement
    }
};

Tensor max_pool2d(const Tensor& x, int k, int stride) {
    todo("max_pool2d");  // TODO: implement
}

}  // namespace tdl
