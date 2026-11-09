// Consistency check: with a fixed seed, build one of the three networks, feed it
// random inputs and run a few steps of forward / backward / SGD. Prints a JSON
// trace (losses, first-step gradients, final parameters) that
// tools/check_consistency.py compares with the reference implementation.
// Provided by the course staff. Do NOT modify.
//
//   ./build/consistency mlp|cnn|lstm SEED
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "nets.hpp"

using namespace tdl;

namespace {

constexpr int kSteps = 5, kSamples = 8;

Tensor randn(Rng& rng, const std::vector<int>& shape) {
    std::vector<float> v(shape_numel(shape));
    for (float& x : v) x = rng.normal();
    return Tensor::from_vector(std::move(v), shape);
}

std::vector<int> labels(Rng& rng, int n, int classes) {
    std::vector<int> y(n);
    for (int& v : y) v = rng.randint(classes);
    return y;
}

void num(double v) {
    if (std::isfinite(v))
        std::printf("%.9g", v);
    else
        std::printf("null");
}

// L2 norm plus a few entries at fixed positions. Missing buffers print as null.
void summary(const float* p, int64_t n, const std::vector<int64_t>& at) {
    std::printf("{\"n\":%lld,\"norm\":", static_cast<long long>(n));
    if (!p) {
        std::printf("null,\"at\":null}");
        return;
    }
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += static_cast<double>(p[i]) * p[i];
    num(std::sqrt(s));
    std::printf(",\"at\":[");
    for (size_t k = 0; k < at.size(); ++k) {
        if (k) std::printf(",");
        num(p[at[k]]);
    }
    std::printf("]}");
}

template <class Net, class Batch>
int run(const char* model, uint64_t seed, Batch make_batch) {
    Rng rng(seed);
    Net net(rng);
    std::vector<Tensor> params = net.parameters();
    std::vector<const char*> names = net.names();
    SGD opt(params, Net::lr, Net::momentum);

    Rng pick(seed ^ 0x9e3779b97f4a7c15ULL);
    std::vector<std::vector<int64_t>> at(params.size());
    for (size_t i = 0; i < params.size(); ++i)
        for (int k = 0; k < kSamples; ++k) at[i].push_back(pick.randint(static_cast<int>(params[i].numel())));

    std::vector<double> losses;
    std::vector<std::vector<float>> grad0(params.size());
    std::vector<bool> has0(params.size());
    for (int step = 0; step < kSteps; ++step) {
        Tensor loss = make_batch(net, rng);
        losses.push_back(loss.item());
        opt.zero_grad();
        loss.backward();
        if (step == 0)
            for (size_t i = 0; i < params.size(); ++i) {
                has0[i] = params[i].has_grad();
                if (has0[i]) grad0[i].assign(params[i].grad(), params[i].grad() + params[i].numel());
            }
        if (Net::clip > 0) clip_grad_norm(params, Net::clip);
        opt.step();
    }

    std::printf("{\"model\":\"%s\",\"seed\":%llu,\"loss\":[", model, static_cast<unsigned long long>(seed));
    for (size_t k = 0; k < losses.size(); ++k) {
        if (k) std::printf(",");
        num(losses[k]);
    }
    std::printf("],\"grad\":{");
    for (size_t i = 0; i < params.size(); ++i) {
        std::printf("%s\"%s\":", i ? "," : "", names[i]);
        summary(has0[i] ? grad0[i].data() : nullptr, params[i].numel(), at[i]);
    }
    std::printf("},\"param\":{");
    for (size_t i = 0; i < params.size(); ++i) {
        std::printf("%s\"%s\":", i ? "," : "", names[i]);
        summary(params[i].data(), params[i].numel(), at[i]);
    }
    std::printf("}}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s mlp|cnn|lstm SEED\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const uint64_t seed = std::stoull(argv[2]);
    if (model == "mlp")
        return run<MlpNet>("mlp", seed, [](MlpNet& n, Rng& r) {
            Tensor x = randn(r, {8, 784});
            return cross_entropy(n.forward(x), labels(r, 8, 10));
        });
    if (model == "cnn")
        return run<CnnNet>("cnn", seed, [](CnnNet& n, Rng& r) {
            Tensor x = randn(r, {4, 3, 32, 32});
            return cross_entropy(n.forward(x), labels(r, 4, 10));
        });
    if (model == "lstm")
        return run<LstmNet>("lstm", seed, [](LstmNet& n, Rng& r) {
            std::vector<Tensor> xs;
            for (int t = 0; t < 16; ++t) xs.push_back(randn(r, {8, LstmNet::input}));
            return cross_entropy(n.forward(xs), labels(r, 8, LstmNet::classes));
        });
    std::fprintf(stderr, "unknown model %s\n", model.c_str());
    return 2;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
