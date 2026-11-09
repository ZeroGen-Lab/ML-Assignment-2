// Test helpers for tinydl: gradient checking and a tiny test runner.
// Provided by the course staff. Do NOT modify.
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tinydl/nn.hpp"
#include "tinydl/ops.hpp"
#include "tinydl/optim.hpp"
#include "tinydl/tensor.hpp"

using namespace tdl;

namespace tdl_test {

inline Rng g_rng(20261027);

inline Tensor randn(const std::vector<int>& shape, bool rg = true, float s = 1.f) {
    std::vector<float> v(shape_numel(shape));
    for (float& x : v) x = s * g_rng.normal();
    return Tensor::from_vector(std::move(v), shape, rg);
}

// Distinct values at least 0.07 apart and away from 0, shuffled, so relu and
// max_pool are differentiable at every point (no ties within finite-difference eps).
inline Tensor rand_away_from_zero(const std::vector<int>& shape) {
    const int64_t n = shape_numel(shape);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = (static_cast<float>(i) - n / 2) * 0.07f + 0.035f;
    for (int64_t i = n - 1; i > 0; --i) std::swap(v[i], v[g_rng.randint(static_cast<int>(i + 1))]);
    return Tensor::from_vector(std::move(v), shape, true);
}

inline void expect(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Test-only op: loss = sum(y * R) for a constant R. Defined here so the tests
// do not depend on your mul().
struct WeightedSumBackward : Node {
    std::vector<float> r;
    const char* name() const override { return "WeightedSumBackward"; }
    void backward(const std::vector<float>& g) override {
        std::vector<float> gy(r.size());
        for (size_t i = 0; i < r.size(); ++i) gy[i] = g[0] * r[i];
        accumulate_grad(inputs[0], gy.data());
    }
};

inline Tensor weighted_sum(const Tensor& y, const std::vector<float>& r) {
    expect(static_cast<int64_t>(r.size()) == y.numel(), "weighted_sum: size mismatch");
    double s = 0.0;
    for (int64_t i = 0; i < y.numel(); ++i) s += static_cast<double>(y.data()[i]) * r[i];
    auto node = std::make_shared<WeightedSumBackward>();
    node->inputs = {y};
    node->r = r;
    return make_output({static_cast<float>(s)}, {1}, node);
}

// Compares analytic gradients of every input with central finite differences
// of loss = sum(f(inputs) * R), R fixed and random.
//   single_op = true : f must be ONE op; its Node::backward(R) is called
//                      directly, so the check does not need Tensor::backward().
//   single_op = false: loss.backward() runs the whole graph (needs autograd).
inline void gradcheck(const std::function<Tensor(std::vector<Tensor>&)>& f, std::vector<Tensor> inputs,
               bool single_op = true, float eps = 1e-2f, float tol = 2e-2f, int max_checks = 40) {
    Tensor probe;
    {
        NoGradGuard ng;
        probe = f(inputs);
    }
    std::vector<float> R(probe.numel());
    for (float& r : R) r = g_rng.normal();
    auto loss_value = [&]() {
        NoGradGuard ng;
        Tensor y = f(inputs);
        double s = 0.0;
        for (int64_t i = 0; i < y.numel(); ++i) s += static_cast<double>(y.data()[i]) * R[i];
        return static_cast<float>(s);
    };
    for (Tensor& t : inputs) t.zero_grad();
    Tensor y = f(inputs);
    expect(y.shape() == probe.shape(), "output shape changed between calls");
    if (single_op) {
        expect(y.grad_fn() != nullptr, "output has no grad_fn (did you return make_output(...)?)");
        y.grad_fn()->backward(R);
    } else {
        weighted_sum(y, R).backward();
    }

    for (size_t k = 0; k < inputs.size(); ++k) {
        Tensor& t = inputs[k];
        if (!t.requires_grad()) continue;
        expect(t.has_grad(), "input " + std::to_string(k) + " got no gradient");
        const float* ga = static_cast<const Tensor&>(t).grad();
        const int64_t n = t.numel();
        const int64_t step = n > max_checks ? n / max_checks : 1;
        for (int64_t i = 0; i < n; i += step) {
            const float old = t.data()[i];
            t.data()[i] = old + eps;
            const float lp = loss_value();
            t.data()[i] = old - eps;
            const float lm = loss_value();
            t.data()[i] = old;
            const float num = (lp - lm) / (2 * eps);
            const float err = std::fabs(num - ga[i]) / std::max(1.f, std::fabs(num) + std::fabs(ga[i]));
            if (err > tol) {
                char buf[200];
                std::snprintf(buf, sizeof buf, "input %zu elem %lld: analytic %.5f numeric %.5f", k,
                              static_cast<long long>(i), ga[i], num);
                throw std::runtime_error(buf);
            }
        }
    }
}

inline void expect_close(float a, float b, float tol, const std::string& what) {
    if (std::fabs(a - b) > tol * std::max(1.f, std::fabs(b))) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s: got %.6f expected %.6f", what.c_str(), a, b);
        throw std::runtime_error(buf);
    }
}

struct Test {
    const char* group;
    const char* name;
    std::function<void()> fn;
};

inline int run_tests(const std::vector<Test>& tests, int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // keep finished results if a later test crashes
    std::string only = argc > 1 ? argv[1] : "";
    if (only == "--list") {
        for (const Test& t : tests) std::printf("%s/%s\n", t.group, t.name);
        return 0;
    }
    int passed = 0, total = 0;
    std::string failed;
    for (const Test& t : tests) {
        if (!only.empty() && only != t.group && only != t.name) continue;
        ++total;
        try {
            t.fn();
            ++passed;
            std::printf("PASS %s/%s\n", t.group, t.name);
        } catch (const std::exception& e) {
            std::printf("FAIL %s/%s: %s\n", t.group, t.name, e.what());
            failed += std::string(failed.empty() ? "" : ",") + "\"" + t.group + "/" + t.name + "\"";
        }
    }
    std::printf("{\"passed\":%d,\"total\":%d,\"failed\":[%s]}\n", passed, total, failed.c_str());
    return passed == total ? 0 : 1;
}

}  // namespace tdl_test
