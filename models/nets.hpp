// The three networks and their optimiser settings, shared by the training
// programs and the consistency check. Provided by the course staff. Do NOT modify.
#pragma once

#include <vector>

#include "tinydl/nn.hpp"
#include "tinydl/ops.hpp"
#include "tinydl/optim.hpp"

namespace tdl {

inline void collect(std::vector<Tensor>& out, Module& m) {
    for (const Tensor& p : m.parameters()) out.push_back(p);
}

// Fashion-MNIST: x [N, 1, 28, 28] or [N, 784] -> fc(784,256)-relu -> fc(256,128)-relu -> fc(128,10)
struct MlpNet {
    static constexpr float lr = 0.05f, momentum = 0.9f, clip = 0.f;  // clip 0: no clipping
    Linear fc1, fc2, fc3;
    explicit MlpNet(Rng& rng) : fc1(784, 256, rng), fc2(256, 128, rng), fc3(128, 10, rng) {}
    Tensor forward(const Tensor& x) {
        Tensor h = reshape(x, {x.dim(0), 784});
        h = relu(fc1.forward(h));
        h = relu(fc2.forward(h));
        return fc3.forward(h);
    }
    std::vector<Tensor> parameters() {
        std::vector<Tensor> p;
        for (Module* m : std::vector<Module*>{&fc1, &fc2, &fc3}) collect(p, *m);
        return p;
    }
    std::vector<const char*> names() const { return {"fc1.W", "fc1.b", "fc2.W", "fc2.b", "fc3.W", "fc3.b"}; }
};

// CIFAR-10: x [N, 3, 32, 32]
//   conv(3->16,3x3,p1)-relu-pool2 -> conv(16->32,3x3,p1)-relu-pool2 -> fc(2048,128)-relu -> fc(128,10)
struct CnnNet {
    static constexpr float lr = 0.01f, momentum = 0.9f, clip = 0.f;
    Conv2d c1, c2;
    MaxPool2d pool;
    Linear fc1, fc2;
    explicit CnnNet(Rng& rng)
        : c1(3, 16, 3, 1, 1, rng), c2(16, 32, 3, 1, 1, rng), pool(2, 2), fc1(32 * 8 * 8, 128, rng), fc2(128, 10, rng) {}
    Tensor forward(const Tensor& x) {
        Tensor h = pool.forward(relu(c1.forward(x)));
        h = pool.forward(relu(c2.forward(h)));
        h = reshape(h, {x.dim(0), 32 * 8 * 8});
        h = relu(fc1.forward(h));
        return fc2.forward(h);
    }
    std::vector<Tensor> parameters() {
        std::vector<Tensor> p;
        for (Module* m : std::vector<Module*>{&c1, &c2, &fc1, &fc2}) collect(p, *m);
        return p;
    }
    std::vector<const char*> names() const {
        return {"conv1.W", "conv1.b", "conv2.W", "conv2.b", "fc1.W", "fc1.b", "fc2.W", "fc2.b"};
    }
};

// UCI HAR: a sequence of T tensors [N, 9] -> LSTMCell(9, 64) unrolled over time,
// last hidden state -> fc(64, 6). Gradients are clipped to norm 5 before each step.
struct LstmNet {
    static constexpr float lr = 0.1f, momentum = 0.9f, clip = 5.f;
    static constexpr int input = 9, hidden = 64, classes = 6;
    LSTMCell cell;
    Linear head;
    explicit LstmNet(Rng& rng) : cell(input, hidden, rng), head(hidden, classes, rng) {}
    Tensor forward(const std::vector<Tensor>& xs) {
        const int b = xs[0].dim(0);
        Tensor h = Tensor::zeros({b, hidden}), c = Tensor::zeros({b, hidden});
        for (const Tensor& x : xs) {
            auto hc = cell.forward(x, h, c);
            h = hc.first;
            c = hc.second;
        }
        return head.forward(h);
    }
    std::vector<Tensor> parameters() {
        std::vector<Tensor> p;
        collect(p, cell);
        collect(p, head);
        return p;
    }
    std::vector<const char*> names() const { return {"lstm.Wih", "lstm.Whh", "lstm.b", "fc.W", "fc.b"}; }
};

}  // namespace tdl
