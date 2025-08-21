#!/usr/bin/env python3
"""
Extended plotting script for SpMM benchmark results
Handles both square and rectangular matrices
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

def plot_by_matrix_type(df, output_dir):
    """Plot performance grouped by matrix type"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Group by matrix type
    matrix_types = df['matrix_type'].unique()
    
    for idx, mtype in enumerate(matrix_types[:6]):  # Plot up to 6 types
        ax = axes[idx]
        data = df[df['matrix_type'] == mtype]
        
        # Average across sparsities for clarity
        avg_data = data.groupby(['M', 'N', 'K']).agg({
            'mkl_sparse_gflops': 'mean',
            'gustavson_gflops': 'mean',
            'gustavson_new_gflops': 'mean',
            'mkl_dense_gflops': 'mean'
        }).reset_index()
        
        # Create labels for x-axis
        labels = [f"{row['M']}x{row['N']}x{row['K']}" for _, row in avg_data.iterrows()]
        x_pos = np.arange(len(labels))
        
        width = 0.2
        ax.bar(x_pos - 1.5*width, avg_data['mkl_sparse_gflops'], width, label='MKL Sparse')
        ax.bar(x_pos - 0.5*width, avg_data['gustavson_gflops'], width, label='Gustavson')
        ax.bar(x_pos + 0.5*width, avg_data['gustavson_new_gflops'], width, label='Gustavson New')
        ax.bar(x_pos + 1.5*width, avg_data['mkl_dense_gflops'], width, label='MKL Dense', alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Matrix Type: {mtype}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.suptitle('SpMM Performance by Matrix Type', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_matrix_type.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tall_skinny_analysis(df, output_dir):
    """Special analysis for tall-skinny matrices"""
    # Filter tall-skinny matrices
    tall_skinny = df[df['matrix_type'] == 'tall_skinny'].copy()
    if len(tall_skinny) == 0:
        tall_skinny = df[df['M'] > df['N']].copy()  # Fallback
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance vs aspect ratio
    tall_skinny['aspect_ratio'] = tall_skinny['M'] / tall_skinny['N']
    avg_by_aspect = tall_skinny.groupby('aspect_ratio').agg({
        'mkl_sparse_gflops': 'mean',
        'gustavson_gflops': 'mean',
        'gustavson_new_gflops': 'mean'
    }).reset_index()
    
    ax1.plot(avg_by_aspect['aspect_ratio'], avg_by_aspect['mkl_sparse_gflops'], 
             'o-', label='MKL Sparse', linewidth=2, markersize=8)
    ax1.plot(avg_by_aspect['aspect_ratio'], avg_by_aspect['gustavson_gflops'], 
             's-', label='Gustavson', linewidth=2, markersize=8)
    ax1.plot(avg_by_aspect['aspect_ratio'], avg_by_aspect['gustavson_new_gflops'], 
             '^-', label='Gustavson New', linewidth=2, markersize=8)
    ax1.set_xlabel('Aspect Ratio (M/N)', fontsize=12)
    ax1.set_ylabel('Average GFLOPS', fontsize=12)
    ax1.set_title('Performance vs Aspect Ratio (Tall-Skinny)', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Performance vs matrix size
    tall_skinny['total_size'] = tall_skinny['M'] * tall_skinny['N'] * tall_skinny['K']
    avg_by_size = tall_skinny.groupby('total_size').agg({
        'mkl_sparse_gflops': 'mean',
        'gustavson_gflops': 'mean',
        'gustavson_new_gflops': 'mean'
    }).reset_index()
    
    ax2.plot(avg_by_size['total_size'], avg_by_size['mkl_sparse_gflops'], 
             'o-', label='MKL Sparse', linewidth=2, markersize=8)
    ax2.plot(avg_by_size['total_size'], avg_by_size['gustavson_gflops'], 
             's-', label='Gustavson', linewidth=2, markersize=8)
    ax2.plot(avg_by_size['total_size'], avg_by_size['gustavson_new_gflops'], 
             '^-', label='Gustavson New', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Matrix Operations (M×N×K)', fontsize=12)
    ax2.set_ylabel('Average GFLOPS', fontsize=12)
    ax2.set_title('Performance vs Problem Size (Tall-Skinny)', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Tall-Skinny Matrix Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tall_skinny_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_extreme_cases(df, output_dir):
    """Analyze extreme cases like 1x4096x4096"""
    # Filter extreme cases (very small M or very large aspect ratios)
    extreme = df[(df['M'] <= 64) | (df['M'] / df['N'] > 100)].copy()
    
    if len(extreme) == 0:
        print("No extreme cases found in data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar chart
    extreme_sorted = extreme.sort_values('M')
    labels = [f"{row['M']}×{row['N']}×{row['K']}" for _, row in extreme_sorted.iterrows()]
    
    # Average across sparsities
    avg_extreme = extreme_sorted.groupby(['M', 'N', 'K']).agg({
        'mkl_sparse_gflops': 'mean',
        'gustavson_gflops': 'mean',
        'gustavson_new_gflops': 'mean',
        'mkl_dense_gflops': 'mean'
    }).reset_index()
    
    x_pos = np.arange(len(avg_extreme))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, avg_extreme['mkl_sparse_gflops'], width, 
           label='MKL Sparse', color='#1f77b4')
    ax.bar(x_pos - 0.5*width, avg_extreme['gustavson_gflops'], width, 
           label='Gustavson', color='#ff7f0e')
    ax.bar(x_pos + 0.5*width, avg_extreme['gustavson_new_gflops'], width, 
           label='Gustavson New', color='#2ca02c')
    ax.bar(x_pos + 1.5*width, avg_extreme['mkl_dense_gflops'], width, 
           label='MKL Dense', color='#d62728', alpha=0.7)
    
    ax.set_xticks(x_pos)
    labels = [f"{row['M']}×{row['N']}×{row['K']}" for _, row in avg_extreme.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('Performance on Extreme Matrix Configurations', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add speedup annotations
    for i, row in avg_extreme.iterrows():
        speedup = row['gustavson_new_gflops'] / row['mkl_sparse_gflops']
        ax.text(i + 0.5*width, row['gustavson_new_gflops'] + 1, 
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_cases_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sparsity_impact_by_shape(df, output_dir):
    """Show how sparsity affects different matrix shapes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define shape categories
    shape_conditions = [
        ('Square', df['M'] == df['N']),
        ('Tall-Skinny', (df['M'] > df['N']) & (df['M'] / df['N'] > 2)),
        ('Short-Wide', (df['M'] < df['N']) & (df['N'] / df['M'] > 2)),
        ('Extreme (M≤64)', df['M'] <= 64)
    ]
    
    for idx, (shape_name, condition) in enumerate(shape_conditions):
        ax = axes[idx // 2, idx % 2]
        shape_data = df[condition]
        
        if len(shape_data) == 0:
            ax.text(0.5, 0.5, f'No {shape_name} data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Group by sparsity and calculate mean performance
        perf_by_sparsity = shape_data.groupby('sparsity').agg({
            'mkl_sparse_gflops': 'mean',
            'gustavson_gflops': 'mean',
            'gustavson_new_gflops': 'mean'
        }).reset_index()
        
        ax.plot(perf_by_sparsity['sparsity'], perf_by_sparsity['mkl_sparse_gflops'], 
               'o-', label='MKL Sparse', linewidth=2, markersize=8)
        ax.plot(perf_by_sparsity['sparsity'], perf_by_sparsity['gustavson_gflops'], 
               's-', label='Gustavson', linewidth=2, markersize=8)
        ax.plot(perf_by_sparsity['sparsity'], perf_by_sparsity['gustavson_new_gflops'], 
               '^-', label='Gustavson New', linewidth=2, markersize=8)
        
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Average GFLOPS', fontsize=12)
        ax.set_title(f'{shape_name} Matrices', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Sparsity Impact by Matrix Shape', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_impact_by_shape.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_extended_summary(df, output_dir):
    """Generate comprehensive summary for extended benchmark"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Overall performance distribution
    ax1 = fig.add_subplot(gs[0, :])
    kernels = ['mkl_sparse_gflops', 'gustavson_gflops', 'gustavson_new_gflops']
    kernel_names = ['MKL Sparse', 'Gustavson', 'Gustavson New']
    data_to_plot = [df[k].values for k in kernels]
    bp = ax1.boxplot(data_to_plot, labels=kernel_names, vert=True, patch_artist=True)
    ax1.set_ylabel('GFLOPS', fontsize=12)
    ax1.set_title('Overall Performance Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 2. Performance by matrix category
    ax2 = fig.add_subplot(gs[1, :])
    categories = []
    performances = {'MKL Sparse': [], 'Gustavson': [], 'Gustavson New': []}
    
    # Define categories based on dimensions
    category_conditions = [
        ('Small\n(≤512³)', (df['M'] <= 512) & (df['N'] <= 512) & (df['K'] <= 512)),
        ('Medium\n(512-2048³)', (df['M'] > 512) & (df['M'] <= 2048) & 
                               (df['N'] > 512) & (df['N'] <= 2048)),
        ('Large\n(>2048³)', (df['M'] > 2048) | (df['N'] > 2048) | (df['K'] > 2048)),
        ('Tall-Skinny', (df['M'] > df['N'] * 2)),
        ('Short-Wide', (df['N'] > df['M'] * 2)),
        ('Extreme\n(M≤64)', df['M'] <= 64)
    ]
    
    for cat_name, condition in category_conditions:
        cat_data = df[condition]
        if len(cat_data) > 0:
            categories.append(cat_name)
            performances['MKL Sparse'].append(cat_data['mkl_sparse_gflops'].mean())
            performances['Gustavson'].append(cat_data['gustavson_gflops'].mean())
            performances['Gustavson New'].append(cat_data['gustavson_new_gflops'].mean())
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax2.bar(x - width, performances['MKL Sparse'], width, label='MKL Sparse', color='#1f77b4')
    ax2.bar(x, performances['Gustavson'], width, label='Gustavson', color='#ff7f0e')
    ax2.bar(x + width, performances['Gustavson New'], width, label='Gustavson New', color='#2ca02c')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Average GFLOPS', fontsize=12)
    ax2.set_title('Performance by Matrix Category', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Best kernel heatmap for different configurations
    ax3 = fig.add_subplot(gs[2:, :])
    
    # Sample some representative configurations
    sample_configs = df.drop_duplicates(subset=['M', 'N', 'K']).head(20)
    
    # Create matrix of best kernels
    config_labels = [f"{row['M']}×{row['N']}×{row['K']}" for _, row in sample_configs.iterrows()]
    sparsity_levels = sorted(df['sparsity'].unique())
    
    best_kernel_matrix = np.zeros((len(config_labels), len(sparsity_levels)))
    
    for i, (_, config) in enumerate(sample_configs.iterrows()):
        for j, sparsity in enumerate(sparsity_levels):
            matching = df[(df['M'] == config['M']) & 
                         (df['N'] == config['N']) & 
                         (df['K'] == config['K']) & 
                         (df['sparsity'] == sparsity)]
            if len(matching) > 0:
                row = matching.iloc[0]
                perfs = [row['mkl_sparse_gflops'], row['gustavson_gflops'], 
                        row['gustavson_new_gflops']]
                best_kernel_matrix[i, j] = np.argmax(perfs)
    
    im = ax3.imshow(best_kernel_matrix, cmap='viridis', aspect='auto')
    ax3.set_xticks(np.arange(len(sparsity_levels)))
    ax3.set_yticks(np.arange(len(config_labels)))
    ax3.set_xticklabels([f'{s:.2f}' for s in sparsity_levels])
    ax3.set_yticklabels(config_labels, fontsize=8)
    ax3.set_xlabel('Sparsity', fontsize=12)
    ax3.set_ylabel('Matrix Configuration', fontsize=12)
    ax3.set_title('Best Performing Kernel by Configuration', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['MKL Sparse', 'Gustavson', 'Gustavson New'])
    
    plt.suptitle('Extended SpMM Benchmark Summary', fontsize=18)
    plt.savefig(os.path.join(output_dir, 'extended_benchmark_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        csv_file = 'benchmark_extended_results.csv'
    else:
        csv_file = sys.argv[1]
    
    output_dir = 'plots_extended'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {csv_file}...")
    df = load_data(csv_file)
    
    # Add matrix_type column if not present
    if 'matrix_type' not in df.columns:
        def get_matrix_type(row):
            M, N, K = row['M'], row['N'], row['K']
            if M == N and N == K: return "square"
            elif M > N * 2: return "tall_skinny"
            elif N > M * 2: return "short_wide"
            elif M <= 64: return "extreme_small"
            else: return "rectangular"
        
        df['matrix_type'] = df.apply(get_matrix_type, axis=1)
    
    print("Generating plots...")
    
    try:
        plot_by_matrix_type(df, output_dir)
        print("  - Performance by matrix type plot saved")
    except Exception as e:
        print(f"  - Warning: Could not create matrix type plot: {e}")
    
    try:
        plot_tall_skinny_analysis(df, output_dir)
        print("  - Tall-skinny analysis plot saved")
    except Exception as e:
        print(f"  - Warning: Could not create tall-skinny analysis: {e}")
    
    try:
        plot_extreme_cases(df, output_dir)
        print("  - Extreme cases plot saved")
    except Exception as e:
        print(f"  - Warning: Could not create extreme cases plot: {e}")
    
    try:
        plot_sparsity_impact_by_shape(df, output_dir)
        print("  - Sparsity impact by shape plot saved")
    except Exception as e:
        print(f"  - Warning: Could not create sparsity impact plot: {e}")
    
    try:
        generate_extended_summary(df, output_dir)
        print("  - Extended summary plot saved")
    except Exception as e:
        print(f"  - Warning: Could not create summary plot: {e}")
    
    print(f"\nAll plots saved to {output_dir}/ directory")
    
    # Print summary statistics
    print("\n=== Extended Benchmark Summary ===")
    print(f"Total configurations tested: {len(df)}")
    print(f"Unique matrix sizes: {len(df.drop_duplicates(subset=['M', 'N', 'K']))}")
    print(f"Matrix types: {df['matrix_type'].value_counts().to_dict()}")
    
    # Performance summary by type
    print("\nAverage GFLOPS by matrix type:")
    type_summary = df.groupby('matrix_type')[['mkl_sparse_gflops', 'gustavson_new_gflops']].mean()
    print(type_summary)
    
    # Best performers
    best_overall = df.loc[df['gustavson_new_gflops'].idxmax()]
    print(f"\nBest Gustavson New performance: {best_overall['gustavson_new_gflops']:.2f} GFLOPS")
    print(f"  Configuration: {best_overall['M']}×{best_overall['N']}×{best_overall['K']}")
    print(f"  Sparsity: {best_overall['sparsity']:.2f}")
    
    # Extreme case performance
    extreme = df[df['M'] <= 64]
    if len(extreme) > 0:
        print(f"\nExtreme cases (M≤64) average speedup vs MKL:")
        speedup = (extreme['mkl_sparse_time'] / extreme['gustavson_new_time']).mean()
        print(f"  Gustavson New: {speedup:.2f}x")

if __name__ == "__main__":
    main()