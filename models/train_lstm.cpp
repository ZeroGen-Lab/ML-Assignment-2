// UCI HAR (smartphone accelerometer/gyroscope, 128 steps x 9 channels, 6 activities).
// LSTMCell(9, 64) unrolled over time, last hidden state -> Linear(64, 6). SGD with momentum + gradient clipping.
// Provided by the course staff.
#include <algorithm>
#include <cstdio>
#include <exception>

#include "data.hpp"
#include "nets.hpp"

using namespace tdl;

namespace {
// [B, T, F] batch -> T tensors of shape [B, F]
std::vector<Tensor> time_major(const Dataset& d, const std::vector<int>& order, int start, int b) {
    const int T = d.sample_shape[0], F = d.sample_shape[1];
    std::vector<Tensor> xs;
    for (int t = 0; t < T; ++t) {
        std::vector<float> v(static_cast<size_t>(b) * F);
        for (int i = 0; i < b; ++i)
            for (int f = 0; f < F; ++f)
                v[static_cast<size_t>(i) * F + f] = d.x[(static_cast<size_t>(order[start + i]) * T + t) * F + f];
        xs.push_back(Tensor::from_vector(std::move(v), {b, F}));
    }
    return xs;
}
}  // namespace

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    const int epochs = args.epochs > 0 ? args.epochs : 4, bs = 32;
    Dataset train = load_dataset(args.data, "har_train", args.limit);
    Dataset test = load_dataset(args.data, "har_test");

    Rng rng(args.seed);
    LstmNet model(rng);
    auto net = [&](const std::vector<Tensor>& xs) { return model.forward(xs); };
    std::vector<Tensor> params = model.parameters();
    SGD opt(params, LstmNet::lr, LstmNet::momentum);

    Timer timer;
    double last_loss = 0;
    for (int e = 0; e < epochs; ++e) {
        std::vector<int> order = shuffled(train.n, rng);
        double tot = 0;
        int nb = 0;
        for (int s = 0; s + bs <= train.n; s += bs, ++nb) {
            Tensor loss = cross_entropy(net(time_major(train, order, s, bs)), batch_y(train, order, s, bs));
            opt.zero_grad();
            loss.backward();
            clip_grad_norm(params, LstmNet::clip);
            opt.step();
            tot += loss.item();
        }
        last_loss = tot / nb;
        std::fprintf(stderr, "epoch %d loss %.4f (%.1fs)\n", e + 1, last_loss, timer.seconds());
    }
    const double train_seconds = timer.seconds();

    NoGradGuard ng;
    std::vector<int> order = identity(test.n);
    int correct = 0;
    for (int s = 0; s < test.n; s += 500) {
        const int b = std::min(500, test.n - s);
        std::vector<int> pred = argmax_rows(net(time_major(test, order, s, b)));
        std::vector<int> y = batch_y(test, order, s, b);
        for (int i = 0; i < b; ++i) correct += pred[i] == y[i];
    }
    report("lstm_har", train_seconds, static_cast<double>(correct) / test.n, last_loss, epochs, train.n);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
