#!/usr/bin/env python3
"""
Plotting script for SpMM benchmark results
Generates multiple plots comparing different kernels across workloads and sparsities
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.gridspec import GridSpec

def load_data(csv_file):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found!")
        sys.exit(1)

def plot_performance_by_sparsity(df, output_dir):
    """Plot performance (GFLOPS) vs sparsity for different matrix sizes"""
    sizes = df['M'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = df[df['M'] == size]
        
        ax.plot(data['sparsity'], data['mkl_sparse_gflops'], 'o-', label='MKL Sparse', linewidth=2, markersize=8)
        ax.plot(data['sparsity'], data['gustavson_gflops'], 's-', label='Gustavson', linewidth=2, markersize=8)
        ax.plot(data['sparsity'], data['gustavson_new_gflops'], '^-', label='Gustavson New', linewidth=2, markersize=8)
        ax.plot(data['sparsity'], data['mkl_dense_gflops'], 'v--', label='MKL Dense', linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Matrix Size: {size}×{size}×{size}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.45, 1.0)
    
    plt.suptitle('SpMM Performance vs Sparsity', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_sparsity.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_by_size(df, output_dir):
    """Plot performance vs matrix size for different sparsities"""
    sparsities = df['sparsity'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, sparsity in enumerate(sparsities):
        ax = axes[idx]
        data = df[df['sparsity'] == sparsity]
        
        sizes = data['M'].values
        x_pos = np.arange(len(sizes))
        
        ax.plot(x_pos, data['mkl_sparse_gflops'], 'o-', label='MKL Sparse', linewidth=2, markersize=8)
        ax.plot(x_pos, data['gustavson_gflops'], 's-', label='Gustavson', linewidth=2, markersize=8)
        ax.plot(x_pos, data['gustavson_new_gflops'], '^-', label='Gustavson New', linewidth=2, markersize=8)
        ax.plot(x_pos, data['mkl_dense_gflops'], 'v--', label='MKL Dense', linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sizes)
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Sparsity: {sparsity:.2f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('SpMM Performance vs Matrix Size', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_size.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_speedup_comparison(df, output_dir):
    """Plot speedup of custom implementations vs MKL baseline"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate speedups
    df['gustavson_speedup'] = df['mkl_sparse_time'] / df['gustavson_time']
    df['gustavson_new_speedup'] = df['mkl_sparse_time'] / df['gustavson_new_time']
    
    # Speedup vs sparsity (averaged across sizes)
    speedup_by_sparsity = df.groupby('sparsity')[['gustavson_speedup', 'gustavson_new_speedup']].mean()
    
    ax1.plot(speedup_by_sparsity.index, speedup_by_sparsity['gustavson_speedup'], 
             'o-', label='Gustavson vs MKL', linewidth=2, markersize=8)
    ax1.plot(speedup_by_sparsity.index, speedup_by_sparsity['gustavson_new_speedup'], 
             's-', label='Gustavson New vs MKL', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Sparsity', fontsize=12)
    ax1.set_ylabel('Speedup vs MKL Sparse', fontsize=12)
    ax1.set_title('Average Speedup vs Sparsity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speedup vs size (averaged across sparsities)
    speedup_by_size = df.groupby('M')[['gustavson_speedup', 'gustavson_new_speedup']].mean()
    
    sizes = speedup_by_size.index.values
    x_pos = np.arange(len(sizes))
    
    ax2.plot(x_pos, speedup_by_size['gustavson_speedup'], 
             'o-', label='Gustavson vs MKL', linewidth=2, markersize=8)
    ax2.plot(x_pos, speedup_by_size['gustavson_new_speedup'], 
             's-', label='Gustavson New vs MKL', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Speedup vs MKL Sparse', fontsize=12)
    ax2.set_title('Average Speedup vs Matrix Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Custom Implementation Speedup vs MKL Baseline', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap(df, output_dir):
    """Create heatmap of best performing kernel for each configuration"""
    # Determine best kernel for each configuration
    kernels = ['mkl_sparse', 'gustavson', 'gustavson_new']
    time_cols = ['mkl_sparse', 'gustavson', 'gustavson_new']
    
    best_kernel = []
    for _, row in df.iterrows():
        times = [row[col] for col in time_cols]
        best_idx = np.argmin(times)
        best_kernel.append(kernels[best_idx])
    
    df['best_kernel'] = best_kernel
    
    # Create pivot table
    pivot = df.pivot(index='sparsity', columns='M', values='best_kernel')
    
    # Map kernels to numbers for coloring
    kernel_map = {'mkl_sparse': 0, 'gustavson': 1, 'gustavson_new': 2}
    pivot_numeric = pivot.applymap(lambda x: kernel_map[x])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    n_bins = 3
    cmap = plt.cm.colors.ListedColormap(colors[:n_bins])
    
    # Plot heatmap
    im = ax.imshow(pivot_numeric, cmap=cmap, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels([f'{s:.2f}' for s in pivot.index])
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, pivot.iloc[i, j].replace('_', '\n'), 
                         ha="center", va="center", color="white", fontsize=10)
    
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Sparsity', fontsize=12)
    ax.set_title('Best Performing Kernel by Configuration', fontsize=14)
    
    # Create colorbar with kernel names
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['MKL Sparse', 'Gustavson', 'Gustavson New'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_kernel_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_plot(df, output_dir):
    """Generate a comprehensive summary plot"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Average performance across all configurations
    ax1 = fig.add_subplot(gs[0, :2])
    avg_perf = df[['mkl_sparse_gflops', 'gustavson_gflops', 'gustavson_new_gflops', 'mkl_dense_gflops']].mean()
    kernels = ['MKL Sparse', 'Gustavson', 'Gustavson New', 'MKL Dense']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(kernels, avg_perf.values, color=colors)
    ax1.set_ylabel('Average GFLOPS', fontsize=12)
    ax1.set_title('Average Performance Across All Configurations', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # 2. Performance distribution
    ax2 = fig.add_subplot(gs[0, 2])
    data_to_plot = [df['mkl_sparse_gflops'], df['gustavson_gflops'], 
                    df['gustavson_new_gflops'], df['mkl_dense_gflops']]
    bp = ax2.boxplot(data_to_plot, labels=['MKL\nSparse', 'Gustav-\nson', 'Gustav-\nson New', 'MKL\nDense'])
    ax2.set_ylabel('GFLOPS', fontsize=12)
    ax2.set_title('Performance Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Performance by sparsity (selected size)
    ax3 = fig.add_subplot(gs[1, :])
    selected_size = 2048
    data_size = df[df['M'] == selected_size]
    ax3.plot(data_size['sparsity'], data_size['mkl_sparse_gflops'], 'o-', label='MKL Sparse', linewidth=2)
    ax3.plot(data_size['sparsity'], data_size['gustavson_gflops'], 's-', label='Gustavson', linewidth=2)
    ax3.plot(data_size['sparsity'], data_size['gustavson_new_gflops'], '^-', label='Gustavson New', linewidth=2)
    ax3.plot(data_size['sparsity'], data_size['mkl_dense_gflops'], 'v--', label='MKL Dense', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Sparsity', fontsize=12)
    ax3.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax3.set_title(f'Performance vs Sparsity (Size={selected_size})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    # 4. Speedup summary
    ax4 = fig.add_subplot(gs[2, :])
    df['gustavson_speedup'] = df['mkl_sparse_time'] / df['gustavson_time']
    df['gustavson_new_speedup'] = df['mkl_sparse_time'] / df['gustavson_new_time']
    
    sparsities = sorted(df['sparsity'].unique())
    width = 0.35
    x = np.arange(len(sparsities))
    
    speedup_g = df.groupby('sparsity')['gustavson_speedup'].mean()
    speedup_gn = df.groupby('sparsity')['gustavson_new_speedup'].mean()
    
    ax4.bar(x - width/2, speedup_g.values, width, label='Gustavson', color='#ff7f0e')
    ax4.bar(x + width/2, speedup_gn.values, width, label='Gustavson New', color='#2ca02c')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='MKL Baseline')
    ax4.set_xlabel('Sparsity', fontsize=12)
    ax4.set_ylabel('Speedup vs MKL Sparse', fontsize=12)
    ax4.set_title('Average Speedup by Sparsity', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{s:.2f}' for s in sparsities])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('SpMM Benchmark Summary', fontsize=18)
    plt.savefig(os.path.join(output_dir, 'benchmark_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        csv_file = 'benchmark_results.csv'
    else:
        csv_file = sys.argv[1]
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {csv_file}...")
    df = load_data(csv_file)
    
    print("Generating plots...")
    plot_performance_by_sparsity(df, output_dir)
    print("  - Performance by sparsity plot saved")
    
    plot_performance_by_size(df, output_dir)
    print("  - Performance by size plot saved")
    
    plot_speedup_comparison(df, output_dir)
    print("  - Speedup comparison plot saved")
    
    plot_heatmap(df, output_dir)
    print("  - Best kernel heatmap saved")
    
    generate_summary_plot(df, output_dir)
    print("  - Summary plot saved")
    
    print(f"\nAll plots saved to {output_dir}/ directory")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total configurations tested: {len(df)}")
    print(f"Matrix sizes: {sorted(df['M'].unique())}")
    print(f"Sparsities: {sorted(df['sparsity'].unique())}")
    
    # Best performers
    best_mkl = df.loc[df['mkl_sparse_gflops'].idxmax()]
    best_g = df.loc[df['gustavson_gflops'].idxmax()]
    best_gn = df.loc[df['gustavson_new_gflops'].idxmax()]
    
    print(f"\nBest MKL Sparse: {best_mkl['mkl_sparse_gflops']:.2f} GFLOPS "
          f"(Size={best_mkl['M']}, Sparsity={best_mkl['sparsity']:.2f})")
    print(f"Best Gustavson: {best_g['gustavson_gflops']:.2f} GFLOPS "
          f"(Size={best_g['M']}, Sparsity={best_g['sparsity']:.2f})")
    print(f"Best Gustavson New: {best_gn['gustavson_new_gflops']:.2f} GFLOPS "
          f"(Size={best_gn['M']}, Sparsity={best_gn['sparsity']:.2f})")

if __name__ == "__main__":
    main()