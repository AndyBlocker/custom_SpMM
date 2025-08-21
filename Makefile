# Compiler and flags
CXX = g++
CXXFLAGS = -O2 -march=native -fopenmp -std=c++17
LDFLAGS = -lgomp -lpthread -lm -ldl

# MKL configuration
MKLROOT ?= /opt/intel/oneapi/mkl/latest
MKL_INCLUDE = -I$(MKLROOT)/include
MKL_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core

# Project directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
KERNEL_DIR = $(SRC_DIR)/kernels
UTILS_DIR = $(SRC_DIR)/utils
BENCHMARK_DIR = $(SRC_DIR)/benchmarks

# Include paths
INCLUDES = -I$(INCLUDE_DIR) $(MKL_INCLUDE)

# Source files
KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.cpp)
UTILS_SRCS = $(wildcard $(UTILS_DIR)/*.cpp)
BENCHMARK_SRCS = $(wildcard $(BENCHMARK_DIR)/*.cpp)
MAIN_SRC = $(SRC_DIR)/main.cpp

# Object files
KERNEL_OBJS = $(KERNEL_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
UTILS_OBJS = $(UTILS_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
BENCHMARK_OBJS = $(BENCHMARK_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(MAIN_SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# All object files
ALL_OBJS = $(KERNEL_OBJS) $(UTILS_OBJS) $(MAIN_OBJ)

# Targets
TARGETS = custom_spmm benchmark_unified benchmark_simple benchmark_extended

# Default target
all: directories $(TARGETS)

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)/kernels
	@mkdir -p $(BUILD_DIR)/utils
	@mkdir -p $(BUILD_DIR)/benchmarks

# Main executable
custom_spmm: $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(MKL_LIBS) $(LDFLAGS)

# Benchmark executables
benchmark_unified: $(KERNEL_OBJS) $(UTILS_OBJS) $(BUILD_DIR)/benchmarks/benchmark_unified.o $(BUILD_DIR)/benchmarks/yaml_config.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(MKL_LIBS) $(LDFLAGS)

benchmark_simple: $(KERNEL_OBJS) $(UTILS_OBJS) $(BUILD_DIR)/benchmarks/benchmark_simple.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(MKL_LIBS) $(LDFLAGS)

benchmark_extended: $(KERNEL_OBJS) $(UTILS_OBJS) $(BUILD_DIR)/benchmarks/benchmark_extended.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(MKL_LIBS) $(LDFLAGS)

# Pattern rules for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGETS)

# Install (optional)
install: all
	@echo "Installing executables..."
	@mkdir -p ~/bin
	@cp $(TARGETS) ~/bin/

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all executables"
	@echo "  custom_spmm      - Build main SpMM executable"
	@echo "  benchmark_*      - Build specific benchmark"
	@echo "  clean            - Remove build artifacts"
	@echo "  install          - Install to ~/bin"
	@echo "  help             - Show this help message"

.PHONY: all clean directories install help