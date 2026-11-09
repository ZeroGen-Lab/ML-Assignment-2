// Dataset loading, batching and reporting for the training programs.
// Provided by the course staff. Do NOT modify.
#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "tinydl/tensor.hpp"

namespace tdl {

// One array from a .bin file: "TDL1", int32 ndim, int32 dims[ndim],
// int32 dtype (0 = float32, 1 = uint8), raw little-endian data.
struct Array {
    std::vector<int> shape;
    std::vector<float> values;  // uint8 data is converted to float 0..255
};
Array load_array(const std::string& path);

struct Dataset {
    std::vector<int> sample_shape;  // e.g. {1, 28, 28}
    std::vector<float> x;           // n * prod(sample_shape)
    std::vector<int> y;
    int n = 0;
    int sample_size() const { return static_cast<int>(shape_numel(sample_shape)); }
};

// Loads <dir>/<name>_x.bin and <dir>/<name>_y.bin; keeps at most `limit` samples (0 = all).
Dataset load_dataset(const std::string& dir, const std::string& name, int limit = 0);

// x = (x * mul - mean[c]) / std[c] where channel c = (index / channel_stride) % mean.size().
void normalize(Dataset& d, float mul, const std::vector<float>& mean, const std::vector<float>& stdev,
               int channel_stride);

// Samples order[start .. start+count) as a tensor of shape [count, sample_shape...].
Tensor batch_x(const Dataset& d, const std::vector<int>& order, int start, int count);
std::vector<int> batch_y(const Dataset& d, const std::vector<int>& order, int start, int count);

std::vector<int> shuffled(int n, Rng& rng);
std::vector<int> identity(int n);

// Index of the max entry in each row of logits [N, C].
std::vector<int> argmax_rows(const Tensor& logits);

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

struct Args {
    std::string data = "data";
    int epochs = -1;   // -1: program default
    int limit = 0;     // training samples, 0: program default
    int seed = 1;
};
Args parse_args(int argc, char** argv);

void report(const std::string& model, double train_seconds, double test_acc, double final_loss, int epochs,
            int train_samples);

}  // namespace tdl
