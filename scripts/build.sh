#!/bin/bash
# Master build script for SpMM benchmarks

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "======================================"
echo "SpMM Benchmark Build System"
echo "======================================"

# Function to find MKL
find_mkl() {
    if [ -z "$MKLROOT" ]; then
        echo "MKLROOT not set. Searching for MKL installation..."
        
        # Common MKL locations
        MKL_PATHS=(
            "/opt/intel/oneapi/mkl/latest"
            "/opt/intel/mkl"
            "/usr/local/intel/mkl"
            "$HOME/intel/oneapi/mkl/latest"
        )
        
        for path in "${MKL_PATHS[@]}"; do
            if [ -d "$path" ]; then
                export MKLROOT="$path"
                echo "Found MKL at: $MKLROOT"
                return 0
            fi
        done
        
        echo "Error: Could not find MKL installation."
        echo "Please set MKLROOT environment variable or source Intel MKL environment."
        echo "Try one of:"
        echo "  source /opt/intel/oneapi/setvars.sh"
        echo "  export MKLROOT=/path/to/mkl"
        return 1
    fi
    
    echo "Using MKL from: $MKLROOT"
    return 0
}

# Parse command line arguments
BUILD_TYPE="Release"
TARGETS=("all")
USE_MAKEFILE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --target)
            TARGETS=("$2")
            shift 2
            ;;
        --clean)
            if [ "$USE_MAKEFILE" = true ] && [ -f "$PROJECT_ROOT/Makefile" ]; then
                echo "Cleaning with Makefile..."
                cd "$PROJECT_ROOT" && make clean
            else
                echo "Cleaning build directory..."
                rm -rf "$BUILD_DIR"/*
            fi
            echo "Clean complete."
            exit 0
            ;;
        --no-makefile)
            USE_MAKEFILE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug             Build with debug symbols"
            echo "  --target <name>     Build specific target (unified, simple, extended, original, all)"
            echo "  --clean             Clean build directory"
            echo "  --no-makefile       Build without Makefile (direct compilation)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find MKL
if ! find_mkl; then
    exit 1
fi

# Use Makefile if available and enabled
if [ "$USE_MAKEFILE" = true ] && [ -f "$PROJECT_ROOT/Makefile" ]; then
    echo "Building with Makefile..."
    cd "$PROJECT_ROOT"
    
    # Export MKLROOT for Makefile
    export MKLROOT
    
    # Determine make targets
    if [ "${TARGETS[0]}" = "all" ]; then
        make_targets="all"
    else
        make_targets=""
        for target in "${TARGETS[@]}"; do
            case $target in
                unified)
                    make_targets="$make_targets benchmark_unified"
                    ;;
                simple)
                    make_targets="$make_targets benchmark_simple"
                    ;;
                extended)
                    make_targets="$make_targets benchmark_extended"
                    ;;
                original)
                    make_targets="$make_targets custom_spmm"
                    ;;
            esac
        done
    fi
    
    # Build with make
    if [ "$BUILD_TYPE" = "Debug" ]; then
        make CXXFLAGS="-g -O0 -DDEBUG -march=native -fopenmp -std=c++17" $make_targets
    else
        make $make_targets
    fi
    
    echo ""
    echo "======================================"
    echo "Build complete!"
    echo "Executables are in project root"
    echo "======================================"
    exit 0
fi

# Fallback to direct compilation (for backward compatibility)
echo "Building without Makefile (direct compilation)..."

# Create build directory
mkdir -p "$BUILD_DIR"

# Compiler settings
CXX="g++"
if [ "$BUILD_TYPE" = "Debug" ]; then
    CXXFLAGS="-g -O0 -DDEBUG"
else
    CXXFLAGS="-O3 -march=native -DNDEBUG"
fi
CXXFLAGS="$CXXFLAGS -fopenmp -std=c++17"

# Include directories
INCLUDES="-I$PROJECT_ROOT/include"

# MKL settings
MKL_INCLUDES="-I${MKLROOT}/include"
MKL_LIBS="-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"

# Common source files (with new structure)
COMMON_SOURCES=(
    "$PROJECT_ROOT/src/kernels/MKL_Sparse_Methods.cpp"
    "$PROJECT_ROOT/src/kernels/custom_spmm.cpp"
    "$PROJECT_ROOT/src/kernels/custom_spmm_yk.cpp"
    "$PROJECT_ROOT/src/kernels/custom_spmm_multi_thread.cpp"
    "$PROJECT_ROOT/src/kernels/custom_spmm_single_thread.cpp"
    "$PROJECT_ROOT/src/kernels/kernel_registry.cpp"
    "$PROJECT_ROOT/src/kernels/mkl_kernel.cpp"
    "$PROJECT_ROOT/src/kernels/gustavson_kernel.cpp"
    "$PROJECT_ROOT/src/utils/csr_builder.cpp"
    "$PROJECT_ROOT/src/utils/make_lut.cpp"
    "$PROJECT_ROOT/src/utils/tilling.cpp"
)

# Additional sources for unified benchmark
UNIFIED_SOURCES=(
    "$PROJECT_ROOT/src/benchmarks/yaml_config.cpp"
)

# Function to build a target
build_target() {
    local target_name=$1
    local main_source=$2
    local output_name=$3
    local extra_sources=("${@:4}")
    
    echo ""
    echo "Building $target_name..."
    
    local cmd="$CXX $CXXFLAGS $main_source ${COMMON_SOURCES[@]} ${extra_sources[@]} $INCLUDES $MKL_INCLUDES $MKL_LIBS -o $BUILD_DIR/$output_name"
    
    echo "Compile command:"
    echo "$cmd"
    
    if $cmd; then
        echo "✓ $target_name built successfully: $BUILD_DIR/$output_name"
        
        # Create symlink in project root for backward compatibility
        ln -sf "$BUILD_DIR/$output_name" "$PROJECT_ROOT/$output_name"
    else
        echo "✗ Failed to build $target_name"
        return 1
    fi
}

# Determine what to build
if [ "${TARGETS[0]}" = "all" ]; then
    TARGETS=("unified" "simple" "extended" "original")
fi

# Build targets
echo ""
echo "Build configuration:"
echo "  Type: $BUILD_TYPE"
echo "  Targets: ${TARGETS[@]}"
echo "  Compiler: $CXX"

for target in "${TARGETS[@]}"; do
    case $target in
        unified)
            build_target "Unified Benchmark" \
                "$PROJECT_ROOT/src/benchmarks/benchmark_unified.cpp" \
                "benchmark_unified" \
                "${UNIFIED_SOURCES[@]}"
            ;;
        simple)
            build_target "Simple Benchmark" \
                "$PROJECT_ROOT/src/benchmarks/benchmark_simple.cpp" \
                "benchmark_simple"
            ;;
        extended)
            build_target "Extended Benchmark" \
                "$PROJECT_ROOT/src/benchmarks/benchmark_extended.cpp" \
                "benchmark_extended"
            ;;
        original)
            build_target "Original Main" \
                "$PROJECT_ROOT/src/main.cpp" \
                "custom_spmm"
            ;;
        *)
            echo "Unknown target: $target"
            ;;
    esac
done

echo ""
echo "======================================"
echo "Build complete!"
echo "Binaries are in: $BUILD_DIR/"
echo "======================================" 