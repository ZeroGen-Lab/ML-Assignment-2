# Build:  make            (all programs into build/)
#         make test       (gradient checks)
#         make check      (fixed-seed consistency check against the reference values)
CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall
INC = -Iinclude -Imodels
LIB = build/obj/tensor.o build/obj/autograd.o build/obj/ops.o build/obj/nn.o build/obj/optim.o
HDR = $(wildcard include/tinydl/*.hpp) models/data.hpp models/nets.hpp
PROGS = build/gradcheck build/consistency build/train_mlp build/train_cnn build/train_lstm

all: $(PROGS)

# Each source is compiled once and shared by all programs.
build/obj/%.o: src/%.cpp $(HDR)
	@mkdir -p build/obj
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

build/obj/data.o: models/data.cpp $(HDR)
	@mkdir -p build/obj
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

build/gradcheck: tests/gradcheck.cpp $(LIB) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) tests/gradcheck.cpp $(LIB) -o $@

build/consistency: tests/consistency.cpp $(LIB) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) tests/consistency.cpp $(LIB) -o $@

build/train_%: models/train_%.cpp build/obj/data.o $(LIB) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) $< build/obj/data.o $(LIB) -o $@

test: build/gradcheck
	./build/gradcheck

check: build/consistency
	python3 -I tools/check_consistency.py build/consistency tests/consistency_expected.json

clean:
	rm -rf build

.PHONY: all test check clean
