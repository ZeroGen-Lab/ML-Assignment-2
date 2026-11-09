// Provided by the course staff. Do NOT modify (the grader uses its own copy).
#include "tinydl/tensor.hpp"

#include <cmath>
#include <sstream>

namespace tdl {

TensorImpl::~TensorImpl() {
    std::vector<std::shared_ptr<Node>> pending;
    if (grad_fn) pending.push_back(std::move(grad_fn));
    while (!pending.empty()) {
        std::shared_ptr<Node> node = std::move(pending.back());
        pending.pop_back();
        if (node.use_count() != 1) continue;  // still referenced elsewhere: just drop our reference
        // We hold the last reference: detach the grad_fn of inputs that die with
        // this node, so destroying the node below does not recurse.
        for (Tensor& t : node->inputs)
            if (t.impl && t.impl.use_count() == 1 && t.impl->grad_fn) pending.push_back(std::move(t.impl->grad_fn));
    }
}

namespace {
bool g_grad_enabled = true;

std::shared_ptr<TensorImpl> new_impl(std::vector<float> data, const std::vector<int>& shape,
                                     bool requires_grad) {
    for (int d : shape) check(d >= 0, "negative dimension in shape " + shape_str(shape));
    check(static_cast<int64_t>(data.size()) == shape_numel(shape),
          "data size " + std::to_string(data.size()) + " does not match shape " + shape_str(shape));
    auto p = std::make_shared<TensorImpl>();
    p->data = std::move(data);
    p->shape = shape;
    p->requires_grad = requires_grad;
    return p;
}
}  // namespace

std::string shape_str(const std::vector<int>& shape) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
    os << ']';
    return os.str();
}

int64_t shape_numel(const std::vector<int>& shape) {
    int64_t n = 1;
    for (int d : shape) n *= d;
    return n;
}

void todo(const char* what) {
    throw std::logic_error(std::string("TODO not implemented: ") + what);
}

Tensor Tensor::zeros(const std::vector<int>& shape, bool requires_grad) {
    return full(shape, 0.f, requires_grad);
}

Tensor Tensor::full(const std::vector<int>& shape, float value, bool requires_grad) {
    return Tensor(new_impl(std::vector<float>(shape_numel(shape), value), shape, requires_grad));
}

Tensor Tensor::from_vector(std::vector<float> data, const std::vector<int>& shape, bool requires_grad) {
    return Tensor(new_impl(std::move(data), shape, requires_grad));
}

const std::vector<int>& Tensor::shape() const { return impl->shape; }
int Tensor::ndim() const { return static_cast<int>(impl->shape.size()); }
int Tensor::dim(int i) const {
    int n = ndim();
    if (i < 0) i += n;
    check(i >= 0 && i < n, "dim index out of range for shape " + shape_str(impl->shape));
    return impl->shape[i];
}
int64_t Tensor::numel() const { return static_cast<int64_t>(impl->data.size()); }

float* Tensor::data() { return impl->data.data(); }
const float* Tensor::data() const { return impl->data.data(); }

float* Tensor::grad() {
    if (impl->grad.size() != impl->data.size()) impl->grad.assign(impl->data.size(), 0.f);
    return impl->grad.data();
}
const float* Tensor::grad() const { return has_grad() ? impl->grad.data() : nullptr; }
bool Tensor::has_grad() const { return impl->grad.size() == impl->data.size() && !impl->data.empty(); }
void Tensor::zero_grad() {
    if (!impl->grad.empty()) std::fill(impl->grad.begin(), impl->grad.end(), 0.f);
}

bool Tensor::requires_grad() const { return impl->requires_grad; }
void Tensor::set_requires_grad(bool value) { impl->requires_grad = value; }
const std::shared_ptr<Node>& Tensor::grad_fn() const { return impl->grad_fn; }

float Tensor::item() const {
    check(numel() == 1, "item() needs a one-element tensor, got shape " + shape_str(impl->shape));
    return impl->data[0];
}

Tensor make_output(std::vector<float> data, std::vector<int> shape, std::shared_ptr<Node> node) {
    bool needs = false;
    if (g_grad_enabled && node) {
        for (const Tensor& in : node->inputs) needs = needs || (in.defined() && in.requires_grad());
    }
    Tensor out(new_impl(std::move(data), shape, needs));
    if (needs) out.impl->grad_fn = std::move(node);
    return out;
}

void accumulate_grad(Tensor& t, const float* g) {
    if (!t.defined() || !t.requires_grad()) return;
    float* dst = t.grad();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) dst[i] += g[i];
}

bool grad_enabled() { return g_grad_enabled; }
NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = prev_; }

// splitmix64: simple, fast, identical on every platform.
Rng::Rng(uint64_t seed) : state_(seed) {}

uint64_t Rng::next_u64() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float Rng::uniform() { return static_cast<float>((next_u64() >> 40) * (1.0 / 16777216.0)); }
float Rng::uniform(float lo, float hi) { return lo + (hi - lo) * uniform(); }

float Rng::normal() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    double u1 = ((next_u64() >> 11) + 1.0) * (1.0 / 9007199254740993.0);  // (0, 1]
    double u2 = (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    double r = std::sqrt(-2.0 * std::log(u1));
    double a = 6.283185307179586 * u2;
    spare_ = static_cast<float>(r * std::sin(a));
    has_spare_ = true;
    return static_cast<float>(r * std::cos(a));
}

int Rng::randint(int n) { return static_cast<int>(next_u64() % static_cast<uint64_t>(n)); }

}  // namespace tdl
