// Public gradient checks and unit tests for tinydl. Provided by the course staff.
// Each test prints PASS/FAIL; the last line is a JSON summary.
#include "check.hpp"

using namespace tdl;
using namespace tdl_test;

namespace {

std::vector<Test> tests() {
    using V = std::vector<Tensor>&;
    return {
        // ---------------------------------------------------------------- ops
        {"ops", "add", [] { gradcheck([](V v) { return add(v[0], v[1]); }, {randn({3, 4}), randn({3, 4})}); }},
        {"ops", "mul", [] { gradcheck([](V v) { return mul(v[0], v[1]); }, {randn({2, 5}), randn({2, 5})}); }},
        {"ops", "relu", [] { gradcheck([](V v) { return relu(v[0]); }, {rand_away_from_zero({4, 6})}); }},
        {"ops", "sigmoid", [] { gradcheck([](V v) { return sigmoid(v[0]); }, {randn({4, 6}, true, 3.f)}); }},
        {"ops", "tanh", [] { gradcheck([](V v) { return tdl::tanh(v[0]); }, {randn({4, 6})}); }},
        {"ops", "matmul", [] { gradcheck([](V v) { return matmul(v[0], v[1]); }, {randn({3, 5}), randn({5, 4})}); }},
        {"ops", "add_bias", [] { gradcheck([](V v) { return add_bias(v[0], v[1]); }, {randn({4, 3}), randn({3})}); }},
        {"ops", "reshape", [] { gradcheck([](V v) { return reshape(v[0], {6, 2}); }, {randn({3, 4})}); }},
        {"ops", "slice_cols", [] { gradcheck([](V v) { return slice_cols(v[0], 2, 3); }, {randn({4, 7})}); }},
        {"ops", "cross_entropy", [] {
             std::vector<int> y = {1, 0, 3, 3, 2};
             gradcheck([y](V v) { return cross_entropy(v[0], y); }, {randn({5, 4}, true, 2.f)});
         }},
        {"ops", "cross_entropy_value", [] {
             Tensor x = Tensor::from_vector({1.f, 2.f, 3.f, 1000.f, 0.f, -1000.f}, {2, 3});
             float l = cross_entropy(x, {2, 0}).item();
             // row0: log(e1+e2+e3) - 3 ; row1: ~0 (stable even with huge logits)
             float r0 = std::log(std::exp(1.f) + std::exp(2.f) + std::exp(3.f)) - 3.f;
             expect(std::isfinite(l), "loss is not finite (subtract the row max)");
             expect_close(l, r0 / 2.f, 1e-4f, "cross_entropy");
         }},
        {"ops", "conv2d", [] {
             gradcheck([](V v) { return conv2d(v[0], v[1], v[2], 1, 1); },
                       {randn({2, 3, 5, 5}), randn({4, 3, 3, 3}, true, 0.5f), randn({4})});
         }},
        {"ops", "conv2d_stride2", [] {
             gradcheck([](V v) { return conv2d(v[0], v[1], v[2], 2, 0); },
                       {randn({1, 2, 7, 6}), randn({3, 2, 3, 3}, true, 0.5f), randn({3})});
         }},
        {"ops", "conv2d_value", [] {
             Tensor x = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
             Tensor w = Tensor::from_vector({1, 0, 0, -1}, {1, 1, 2, 2});
             Tensor b = Tensor::from_vector({0.5f}, {1});
             Tensor y = conv2d(x, w, b, 1, 0);
             expect(y.shape() == std::vector<int>({1, 1, 2, 2}), "shape " + shape_str(y.shape()));
             for (int i = 0; i < 4; ++i) expect_close(y.data()[i], -3.5f, 1e-5f, "conv value");
         }},
        {"ops", "max_pool2d", [] { gradcheck([](V v) { return max_pool2d(v[0], 2, 2); }, {rand_away_from_zero({2, 3, 4, 6})}); }},
        {"ops", "max_pool2d_overlap", [] { gradcheck([](V v) { return max_pool2d(v[0], 3, 1); }, {rand_away_from_zero({1, 2, 5, 5})}); }},
        // ----------------------------------------------------------- autograd
        {"autograd", "shared_input", [] {
             // y = x*x + x: x is used three times, gradient must accumulate.
             gradcheck([](V v) { return add(mul(v[0], v[0]), v[0]); }, {randn({4})}, false);
         }},
        {"autograd", "diamond", [] {
             gradcheck([](V v) {
                 Tensor a = tdl::tanh(v[0]);
                 Tensor b = sigmoid(a), c = add(a, a);
                 return mul(b, c);
             }, {randn({3, 3})}, false);
         }},
        {"autograd", "deep_chain", [] {
             // 20000 nodes: a recursive topological sort overflows the stack.
             Tensor x = Tensor::full({1}, 0.5f, true);
             Tensor y = x;
             for (int i = 0; i < 20000; ++i) y = relu(y);
             y.backward();
             expect_close(x.grad()[0], 1.f, 1e-5f, "dy/dx");
         }},
        {"autograd", "accumulate", [] {
             Tensor x = Tensor::full({2}, 3.f, true);
             weighted_sum(x, {1.f, 1.f}).backward();
             weighted_sum(x, {2.f, 2.f}).backward();
             expect_close(x.grad()[0], 3.f, 1e-5f, "grad after two backward calls");
         }},
        {"autograd", "no_grad", [] {
             Tensor x = Tensor::full({2}, 1.f, true);
             Tensor y;
             {
                 NoGradGuard ng;
                 y = mul(x, x);
             }
             expect(!y.requires_grad() && !y.grad_fn(), "NoGradGuard output should not track gradients");
         }},
        {"autograd", "constant_input", [] {
             // b does not require grad: it must not receive a gradient.
             Tensor a = randn({3}), b = randn({3}, false);
             weighted_sum(mul(a, b), {1.f, 1.f, 1.f}).backward();
             expect(!b.has_grad(), "constant got a gradient");
             for (int i = 0; i < 3; ++i) expect_close(a.grad()[i], b.data()[i], 1e-5f, "da");
         }},
        // ----------------------------------------------------------------- nn
        {"nn", "linear", [] {
             Rng r(1);
             auto L = std::make_shared<Linear>(5, 3, r);
             gradcheck([L](V v) { L->W = v[1]; L->b = v[2]; return L->forward(v[0]); },
                       {randn({4, 5}), L->W, L->b}, false);
         }},
        {"nn", "conv_relu_pool", [] {
             Rng r(2);
             auto C = std::make_shared<Conv2d>(2, 3, 3, 1, 1, r);
             auto P = std::make_shared<MaxPool2d>(2, 2);
             gradcheck([C, P](V v) { C->W = v[1]; C->b = v[2]; return P->forward(relu(C->forward(v[0]))); },
                       {randn({2, 2, 4, 4}), C->W, C->b}, false, 1e-3f, 3e-2f);
         }},
        {"nn", "lstm_cell", [] {
             Rng r(3);
             auto cell = std::make_shared<LSTMCell>(3, 4, r);
             gradcheck([cell](V v) {
                 cell->Wih = v[3]; cell->Whh = v[4]; cell->b = v[5];
                 auto hc = cell->forward(v[0], v[1], v[2]);
                 return add(hc.first, hc.second);
             }, {randn({2, 3}), randn({2, 4}), randn({2, 4}), cell->Wih, cell->Whh, cell->b}, false);
         }},
        {"nn", "lstm_value", [] {
             Rng r(4);
             LSTMCell cell(1, 1, r);
             cell.Wih = Tensor::from_vector({0, 0, 0, 0}, {1, 4}, true);
             cell.Whh = Tensor::from_vector({0, 0, 0, 0}, {1, 4}, true);
             cell.b = Tensor::from_vector({0.f, 100.f, 0.f, 0.f}, {4}, true);  // f=1, i=o=0.5, g=0
             auto hc = cell.forward(Tensor::zeros({1, 1}), Tensor::zeros({1, 1}), Tensor::full({1, 1}, 2.f));
             expect_close(hc.second.item(), 2.f, 1e-4f, "c'");
             expect_close(hc.first.item(), 0.5f * std::tanh(2.f), 1e-4f, "h'");
         }},
        // -------------------------------------------------------------- optim
        {"optim", "zero_grad", [] {
             Tensor p = Tensor::full({3}, 1.f, true);
             p.grad()[0] = p.grad()[1] = p.grad()[2] = 5.f;
             SGD opt({p}, 0.1f);
             opt.zero_grad();
             for (int i = 0; i < 3; ++i) expect(p.grad()[i] == 0.f, "gradient not cleared");
         }},
        {"optim", "sgd_momentum", [] {
             Tensor p = Tensor::from_vector({1.f, -2.f}, {2}, true);
             SGD opt({p}, 0.1f, 0.9f);
             for (int s = 0; s < 2; ++s) {
                 opt.zero_grad();
                 for (int i = 0; i < 2; ++i) p.grad()[i] = 2.f * p.data()[i];  // g = d(p^2)/dp
                 opt.step();
             }
             // hand-computed: v1 = 2p0 ; p1 = p0 - .1 v1 ; v2 = .9 v1 + 2 p1 ; p2 = p1 - .1 v2
             float p0 = 1.f, v1 = 2.f * p0, p1 = p0 - 0.1f * v1, v2 = 0.9f * v1 + 2.f * p1, p2 = p1 - 0.1f * v2;
             expect_close(p.data()[0], p2, 1e-5f, "p[0]");
             expect_close(p.data()[1], -2.f * p2, 1e-5f, "p[1]");
         }},
        {"optim", "sgd_plain", [] {
             Tensor p = Tensor::from_vector({2.f, -1.f}, {2}, true);
             SGD opt({p}, 0.5f);
             p.grad()[0] = 1.f;
             p.grad()[1] = -4.f;
             opt.step();
             expect_close(p.data()[0], 1.5f, 1e-6f, "p[0]");
             expect_close(p.data()[1], 1.f, 1e-6f, "p[1]");
         }},
        {"optim", "skip_unused", [] {
             Tensor used = Tensor::full({1}, 1.f, true), unused = Tensor::full({1}, 1.f, true);
             SGD opt({used, unused}, 0.1f, 0.9f);
             used.grad()[0] = 1.f;
             opt.step();
             expect(unused.data()[0] == 1.f, "parameter without gradient was changed");
         }},
    };
}

}  // namespace

int main(int argc, char** argv) { return run_tests(tests(), argc, argv); }
