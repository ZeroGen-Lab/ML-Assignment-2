#include "data.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tdl {

Array load_array(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (run the data preparation step first)");
    char magic[4];
    int32_t ndim = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&ndim), 4);
    if (!f || std::memcmp(magic, "TDL1", 4) != 0 || ndim < 1 || ndim > 8)
        throw std::runtime_error(path + ": not a TDL1 file");
    Array a;
    a.shape.resize(ndim);
    f.read(reinterpret_cast<char*>(a.shape.data()), 4 * ndim);
    int32_t dtype = -1;
    f.read(reinterpret_cast<char*>(&dtype), 4);
    const int64_t n = shape_numel(a.shape);
    a.values.resize(n);
    if (dtype == 0) {
        f.read(reinterpret_cast<char*>(a.values.data()), 4 * n);
    } else if (dtype == 1) {
        std::vector<uint8_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n);
        for (int64_t i = 0; i < n; ++i) a.values[i] = raw[i];
    } else {
        throw std::runtime_error(path + ": unknown dtype");
    }
    if (!f) throw std::runtime_error(path + ": truncated");
    return a;
}

Dataset load_dataset(const std::string& dir, const std::string& name, int limit) {
    Array x = load_array(dir + "/" + name + "_x.bin");
    Array y = load_array(dir + "/" + name + "_y.bin");
    if (y.shape.size() != 1 || y.shape[0] != x.shape[0]) throw std::runtime_error(name + ": x/y size mismatch");
    Dataset d;
    d.n = x.shape[0];
    if (limit > 0 && limit < d.n) d.n = limit;
    d.sample_shape.assign(x.shape.begin() + 1, x.shape.end());
    x.values.resize(static_cast<size_t>(d.n) * d.sample_size());
    d.x = std::move(x.values);
    d.y.resize(d.n);
    for (int i = 0; i < d.n; ++i) d.y[i] = static_cast<int>(y.values[i]);
    return d;
}

void normalize(Dataset& d, float mul, const std::vector<float>& mean, const std::vector<float>& stdev,
               int channel_stride) {
    const int C = static_cast<int>(mean.size());
    for (size_t i = 0; i < d.x.size(); ++i) {
        const int c = static_cast<int>((i / channel_stride) % C);
        d.x[i] = (d.x[i] * mul - mean[c]) / stdev[c];
    }
}

Tensor batch_x(const Dataset& d, const std::vector<int>& order, int start, int count) {
    const int s = d.sample_size();
    std::vector<float> v(static_cast<size_t>(count) * s);
    for (int i = 0; i < count; ++i)
        std::copy_n(d.x.begin() + static_cast<size_t>(order[start + i]) * s, s, v.begin() + static_cast<size_t>(i) * s);
    std::vector<int> shape = {count};
    shape.insert(shape.end(), d.sample_shape.begin(), d.sample_shape.end());
    return Tensor::from_vector(std::move(v), shape);
}

std::vector<int> batch_y(const Dataset& d, const std::vector<int>& order, int start, int count) {
    std::vector<int> y(count);
    for (int i = 0; i < count; ++i) y[i] = d.y[order[start + i]];
    return y;
}

std::vector<int> identity(int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = i;
    return v;
}

std::vector<int> shuffled(int n, Rng& rng) {
    std::vector<int> v = identity(n);
    for (int i = n - 1; i > 0; --i) std::swap(v[i], v[rng.randint(i + 1)]);
    return v;
}

std::vector<int> argmax_rows(const Tensor& logits) {
    const int N = logits.dim(0), C = logits.dim(1);
    std::vector<int> r(N);
    for (int i = 0; i < N; ++i) {
        const float* row = logits.data() + static_cast<size_t>(i) * C;
        r[i] = static_cast<int>(std::max_element(row, row + C) - row);
    }
    return r;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--data") a.data = argv[i + 1];
        else if (k == "--epochs") a.epochs = std::atoi(argv[i + 1]);
        else if (k == "--limit") a.limit = std::atoi(argv[i + 1]);
        else if (k == "--seed") a.seed = std::atoi(argv[i + 1]);
        else throw std::runtime_error("unknown argument " + k);
    }
    return a;
}

void report(const std::string& model, double train_seconds, double test_acc, double final_loss, int epochs,
            int train_samples) {
    std::printf("{\"model\":\"%s\",\"epochs\":%d,\"train_samples\":%d,\"train_seconds\":%.2f,"
                "\"test_acc\":%.4f,\"final_loss\":%.4f}\n",
                model.c_str(), epochs, train_samples, train_seconds, test_acc, final_loss);
    std::fflush(stdout);
}

}  // namespace tdl
