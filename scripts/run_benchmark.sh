#!/bin/bash
# Unified benchmark runner script

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/results"

# Default settings
PRESET="simple"
OUTPUT_FILE=""
BENCHMARK_BINARY="$PROJECT_ROOT/benchmark_unified"
EXTRA_ARGS=""

# Function to extract output_file from YAML config
extract_yaml_output_file() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        # Simple grep-based YAML parsing for output_file
        local output_line=$(grep -E "^[[:space:]]*output_file[[:space:]]*:" "$config_file" | head -1)
        if [ -n "$output_line" ]; then
            # Extract value, remove quotes, comments, and whitespace
            echo "$output_line" | sed -E 's/^[[:space:]]*output_file[[:space:]]*:[[:space:]]*//' | sed -E 's/[[:space:]]*#.*$//' | sed -E 's/^["'\''](.*?)["'\'']$/\1/' | sed 's/[[:space:]]*$//'
        fi
    fi
}

# Function to create temporary config file with modified output path
create_temp_config() {
    local original_config="$1"
    local new_output_file="$2"
    local temp_config="/tmp/benchmark_config_$$.yaml"
    
    # Copy config file and update output_file path
    if [ -f "$original_config" ]; then
        # Replace output_file line with new path
        sed "s|^[[:space:]]*output_file[[:space:]]*:.*|output_file: \"$new_output_file\"|" "$original_config" > "$temp_config"
        echo "$temp_config"
    fi
}

# Function to setup MKL environment
setup_mkl_env() {
    if [ -z "$MKLROOT" ]; then
        # Try to find MKL in common locations
        MKL_PATHS=(
            "/opt/intel/oneapi/mkl/latest"
            "/opt/intel/mkl"
            "/usr/local/intel/mkl"
            "$HOME/intel/oneapi/mkl/latest"
        )
        
        for path in "${MKL_PATHS[@]}"; do
            if [ -d "$path" ]; then
                export MKLROOT="$path"
                break
            fi
        done
    fi
    
    # Set library path for MKL
    if [ -n "$MKLROOT" ]; then
        export LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            # Don't add to EXTRA_ARGS yet, we'll modify the config file later
            shift 2
            ;;
        --preset|-p)
            PRESET="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --warmup)
            EXTRA_ARGS="$EXTRA_ARGS --warmup $2"
            shift 2
            ;;
        --test)
            EXTRA_ARGS="$EXTRA_ARGS --test $2"
            shift 2
            ;;
        --threads)
            EXTRA_ARGS="$EXTRA_ARGS --threads $2"
            shift 2
            ;;
        --quiet|-q)
            EXTRA_ARGS="$EXTRA_ARGS --quiet"
            shift
            ;;
        --build)
            echo "Building benchmark first..."
            "$SCRIPT_DIR/build.sh" --target unified
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config, -c <file>   Load configuration from YAML file"
            echo "  --preset, -p <name>   Benchmark preset (simple, square, tall-skinny, extended, quick)"
            echo "  --output, -o <file>   Output CSV file (default: results/<preset>_<timestamp>.csv)"
            echo "  --warmup <n>          Number of warmup iterations"
            echo "  --test <n>            Number of test iterations"
            echo "  --threads <n>         Number of threads to use"
            echo "  --quiet, -q           Suppress verbose output"
            echo "  --build               Build the benchmark before running"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Available presets:"
            echo "  simple      - Your 4 specific test cases (197x3072x768, etc.)"
            echo "  square      - Square matrices (512² to 4096²)"
            echo "  tall-skinny - Tall-skinny matrices"
            echo "  extended    - Full extended benchmark suite"
            echo "  quick       - Quick test for debugging"
            echo ""
            echo "Example YAML configs in configs/ directory:"
            echo "  simple.yaml          - Same as --preset simple"
            echo "  custom.yaml          - Example custom configuration"
            echo "  quick_test.yaml      - Quick test configuration"
            echo "  performance_sweep.yaml - Comprehensive performance test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Generate timestamp for directory structure
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create timestamped results directory
TIMESTAMPED_RESULTS_DIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$TIMESTAMPED_RESULTS_DIR"

# Handle config file output_file setting
TEMP_CONFIG_FILE=""
if [ -n "$CONFIG_FILE" ]; then
    # Extract output_file from config if it exists
    CONFIG_OUTPUT_FILE=$(extract_yaml_output_file "$CONFIG_FILE")
    if [ -n "$CONFIG_OUTPUT_FILE" ]; then
        # Config file specifies output_file, use timestamped version
        OUTPUT_FILENAME=$(basename "$CONFIG_OUTPUT_FILE")
        NEW_OUTPUT_PATH="$TIMESTAMPED_RESULTS_DIR/$OUTPUT_FILENAME"
        # Create temporary config file with updated path
        TEMP_CONFIG_FILE=$(create_temp_config "$CONFIG_FILE" "$NEW_OUTPUT_PATH")
        CONFIG_FILE="$TEMP_CONFIG_FILE"
        OUTPUT_FILE="$NEW_OUTPUT_PATH"
    else
        # No output_file in config, generate default
        OUTPUT_FILE="$TIMESTAMPED_RESULTS_DIR/benchmark_${PRESET}.csv"
    fi
else
    # Generate output filename if not provided
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="$TIMESTAMPED_RESULTS_DIR/benchmark_${PRESET}.csv"
    else
        # Extract just the filename from the path
        OUTPUT_FILENAME=$(basename "$OUTPUT_FILE")
        OUTPUT_FILE="$TIMESTAMPED_RESULTS_DIR/$OUTPUT_FILENAME"
    fi
fi

# Check if benchmark binary exists
if [ ! -f "$BENCHMARK_BINARY" ]; then
    echo "Benchmark binary not found. Building..."
    "$SCRIPT_DIR/build.sh" --target unified
fi

# Setup environment
setup_mkl_env

# Print configuration
echo "======================================"
echo "SpMM Benchmark Runner"
echo "======================================"
echo "Preset: $PRESET"
echo "Output: $OUTPUT_FILE"
echo "Binary: $BENCHMARK_BINARY"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args: $EXTRA_ARGS"
fi
echo ""

# Run benchmark
echo "Starting benchmark..."
if [ -n "$CONFIG_FILE" ]; then
    # Using config file, let config file handle output_file
    "$BENCHMARK_BINARY" --config "$CONFIG_FILE" $EXTRA_ARGS
else
    # Using preset
    "$BENCHMARK_BINARY" --preset "$PRESET" --output "$OUTPUT_FILE" $EXTRA_ARGS
fi

# Clean up temporary config file
cleanup_temp_files() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}

# Set up trap for cleanup
trap cleanup_temp_files EXIT

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Benchmark completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "======================================"
    
    # Show summary statistics
    if command -v awk &> /dev/null; then
        echo ""
        echo "Quick summary (best Gustavson_New times):"
        if [ -f "$OUTPUT_FILE" ]; then
            awk -F',' 'NR>1 && $8!="" {print $1 ": " $8 "s"}' "$OUTPUT_FILE" | sort -t: -k2 -n | head -5
        fi
    fi
else
    echo "Benchmark failed!"
    exit 1
fi