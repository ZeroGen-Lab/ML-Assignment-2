// CIFAR-10, small CNN, SGD with momentum. Provided by the course staff.
//   conv(3->16,3x3,p1)-relu-pool2 -> conv(16->32,3x3,p1)-relu-pool2 -> fc(2048,128)-relu -> fc(128,10)
#include <algorithm>
#include <cstdio>
#include <exception>

#include "data.hpp"
#include "nets.hpp"

using namespace tdl;

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    const int epochs = args.epochs > 0 ? args.epochs : 2, bs = 32;
    Dataset train = load_dataset(args.data, "cifar_train", args.limit > 0 ? args.limit : 20000);
    Dataset test = load_dataset(args.data, "cifar_test");
    for (Dataset* d : {&train, &test})
        normalize(*d, 1.f / 255.f, {0.4914f, 0.4822f, 0.4465f}, {0.2470f, 0.2435f, 0.2616f}, 32 * 32);

    Rng rng(args.seed);
    CnnNet model(rng);
    auto net = [&](const Tensor& x) { return model.forward(x); };
    SGD opt(model.parameters(), CnnNet::lr, CnnNet::momentum);

    Timer timer;
    double last_loss = 0;
    for (int e = 0; e < epochs; ++e) {
        std::vector<int> order = shuffled(train.n, rng);
        double tot = 0;
        int nb = 0;
        for (int s = 0; s + bs <= train.n; s += bs, ++nb) {
            Tensor loss = cross_entropy(net(batch_x(train, order, s, bs)), batch_y(train, order, s, bs));
            opt.zero_grad();
            loss.backward();
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
        std::vector<int> pred = argmax_rows(net(batch_x(test, order, s, b)));
        std::vector<int> y = batch_y(test, order, s, b);
        for (int i = 0; i < b; ++i) correct += pred[i] == y[i];
    }
    report("cnn_cifar10", train_seconds, static_cast<double>(correct) / test.n, last_loss, epochs, train.n);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
