# Build:  make            (all programs into build/)
#         make test       (gradient checks)
#         make check      (fixed-seed consistency check against the reference values)
CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall
INC = -Iinclude -Imodels
LIB = src/tensor.cpp src/autograd.cpp src/ops.cpp src/nn.cpp src/optim.cpp
HDR = $(wildcard include/tinydl/*.hpp) models/data.hpp models/nets.hpp
PROGS = build/gradcheck build/consistency build/train_mlp build/train_cnn build/train_lstm

all: $(PROGS)

build/gradcheck: tests/gradcheck.cpp $(LIB) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) tests/gradcheck.cpp $(LIB) -o $@

build/consistency: tests/consistency.cpp $(LIB) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) tests/consistency.cpp $(LIB) -o $@

build/train_%: models/train_%.cpp models/data.cpp $(LIB) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) $< models/data.cpp $(LIB) -o $@

test: build/gradcheck
	./build/gradcheck

check: build/consistency
	python3 -I tools/check_consistency.py build/consistency tests/consistency_expected.json

clean:
	rm -rf build

.PHONY: all test check clean
