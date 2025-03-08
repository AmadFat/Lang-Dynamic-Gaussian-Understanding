import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects

POINTS = OrderedDict({
    0: 39107,
    200: 39107,
    1000: 55598,
    5000: 228251,
    5400: 240718,
    10000: 341993,
    10800: 341993,
    15000: 341993,
    16200: 341993,
    20000: 341993,
})

LOSSES_TRAIN = OrderedDict({
    200: 0.0581020,
    1000: 0.0416027,
    5000: 0.0178894,
    5400: 0.0169817,
    10000: 0.0145197,
    10800: 0.0155585,
    15000: 0.0117307,
    16200: 0.0135862,
    20000: 0.0122995,
})

L1S_TRAIN = OrderedDict({
    5: 0.1561880234409781,
    200: 0.05739559649544604,
    1000: 0.04837213127928622,
    5000: 0.01853864168857827,
    10000: 0.015216829419574317,
    15000: 0.013607966987525715,
    20000: 0.012644963229403776,
})

PSNRS_TRAIN = OrderedDict({
    5: 13.594167036168715,
    200: 20.801288043751438,
    1000: 22.779477624332202,
    5000: 28.46585812288172,
    10000: 30.07019099067239,
    15000: 31.001333909876205,
    20000: 31.632848066442154,
})

L1S_TEST = OrderedDict({
    5: 0.1308441021863152,
    200: 0.04572545802768539,
    1000: 0.041686304132728016,
    5000: 0.022944150492548943,
    10000: 0.021244358600062484,
    15000: 0.020258084377821752,
    20000: 0.02026314430815332,
})

PSNRS_TEST = OrderedDict({
    5: 14.950805551865521,
    200: 22.300729527192956,
    1000: 23.86607136445887,
    5000: 27.44228665968951,
    10000: 28.208923564237708,
    15000: 28.527989331413718,
    20000: 28.213235967299518,
})

# Extract iteration values
ITERS_METRICS = list(L1S_TRAIN.keys())
ITERS_POINTS = list(POINTS.keys())

def apply_aesthetics(ax, title, subtitle=None, xlabel="Iterations", ylabel=None, log_y=True):
    """Apply consistent aesthetic styling to an axis"""
    # Title and labels
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    if subtitle:
        ax.text(0.5, 0.94, subtitle, 
                horizontalalignment='center', 
                transform=ax.transAxes, 
                fontsize=13, 
                fontstyle='italic',
                alpha=0.8)
    
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    
    # Scale
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

def create_custom_colormap():
    """Create a custom colormap for the visualization"""
    colors = ['#1A3B71', '#3B6CB2', '#78A6E2']  # Blue palette
    return LinearSegmentedColormap.from_list('custom_blue', colors, N=100)

def plot_training_metrics():
    """Create the training metrics plots with improved aesthetics"""
    # Create color palette
    train_color = '#2B6EB3'
    test_color = '#DD5143'
    loss_color = '#4CAF50'
    
    # Create figure with improved layout - now with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), dpi=100, facecolor='white')
    plt.subplots_adjust(hspace=0.25)
    
    # Customizing figure background
    fig.patch.set_facecolor('white')
    
    # ---- Training Loss Plot (New) ----
    ax0 = axes[0]
    
    # Get iterations and values
    loss_iters = list(LOSSES_TRAIN.keys())
    loss_values = list(LOSSES_TRAIN.values())
    
    # Plot training loss with gradient fill
    ax0.plot(loss_iters, loss_values, '-', linewidth=3, markersize=9, color=loss_color)
    
    # Add markers
    ax0.plot(loss_iters, loss_values, 'o', markersize=8, color=loss_color, 
             markeredgecolor='white', markeredgewidth=1.5)
    
    # Add fill below curves
    ax0.fill_between(loss_iters, loss_values, alpha=0.2, color=loss_color)
    
    # Set axis limits with some padding
    y_min = min(loss_values) * 0.8
    y_max = max(loss_values) * 1.2
    ax0.set_ylim(y_min, y_max)
    
    # Apply consistent aesthetics
    apply_aesthetics(ax0, "Training Loss", "", ylabel="Loss")
    
    # Annotate best point with improved styling
    min_loss_value = min(loss_values)
    min_loss_idx = loss_values.index(min_loss_value)
    min_loss_iter = loss_iters[min_loss_idx]
    
    # Create improved annotations with callouts
    loss_annotation = ax0.annotate(f'Best: {min_loss_value:.6f}',
                xy=(min_loss_iter, min_loss_value),
                xytext=(min_loss_iter*0.3, min_loss_value*1.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=loss_color, 
                                connectionstyle="arc3,rad=.2"),
                fontsize=13, color=loss_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', alpha=0.8, ec=loss_color))
    
    # Add relative improvement
    first_loss = loss_values[0]
    best_loss = min_loss_value
    improvement = ((first_loss - best_loss) / first_loss) * 100
    
    improvement_text = ""
    ax0.text(0.02, 0.05, improvement_text, 
             transform=ax0.transAxes,
             fontsize=13,
             bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='gray', alpha=0.9))
    
    # ---- L1 Loss Plot ----
    ax1 = axes[1]
    
    # Plot train data with gradient fill
    train_line, = ax1.plot(ITERS_METRICS, list(L1S_TRAIN.values()), '-', linewidth=3, 
                          markersize=9, label='Training', color=train_color)
    test_line, = ax1.plot(ITERS_METRICS, list(L1S_TEST.values()), '--', linewidth=2.5, 
                         markersize=9, label='Testing', color=test_color)
    
    # Add markers
    ax1.plot(ITERS_METRICS, list(L1S_TRAIN.values()), 'o', markersize=8, color=train_color, 
            markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(ITERS_METRICS, list(L1S_TEST.values()), 's', markersize=8, color=test_color,
            markeredgecolor='white', markeredgewidth=1.5)
    
    # Add fill below curves
    ax1.fill_between(ITERS_METRICS, list(L1S_TRAIN.values()), alpha=0.2, color=train_color)
    ax1.fill_between(ITERS_METRICS, list(L1S_TEST.values()), alpha=0.15, color=test_color)
    
    # Set axis limits with some padding
    y_min = min(min(L1S_TRAIN.values()), min(L1S_TEST.values())) * 0.8
    y_max = max(max(L1S_TRAIN.values()), max(L1S_TEST.values())) * 1.2
    ax1.set_ylim(y_min, y_max)
    
    # Apply consistent aesthetics
    apply_aesthetics(ax1, "L1 Loss (↓)", "", ylabel="L1 Loss")
    
    # Annotate best points with improved styling
    min_train_value = min(L1S_TRAIN.values())
    min_train_idx = list(L1S_TRAIN.values()).index(min_train_value)
    min_train_iter = list(L1S_TRAIN.keys())[min_train_idx]
    
    min_test_value = min(L1S_TEST.values())
    min_test_idx = list(L1S_TEST.values()).index(min_test_value)
    min_test_iter = list(L1S_TEST.keys())[min_test_idx]
    
    # Create improved annotations with callouts
    train_annotation = ax1.annotate(f'Best: {min_train_value:.4f}',
                xy=(min_train_iter, min_train_value),
                xytext=(min_train_iter*0.3, min_train_value*2.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=train_color, 
                                connectionstyle="arc3,rad=.2"),
                fontsize=13, color=train_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', alpha=0.8, ec=train_color))
    
    test_annotation = ax1.annotate(f'Best: {min_test_value:.4f}',
                xy=(min_test_iter, min_test_value),
                xytext=(min_test_iter*0.4, min_test_value*2.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=test_color, 
                                connectionstyle="arc3,rad=-.2"),
                fontsize=13, color=test_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', alpha=0.8, ec=test_color))
    
    # Add legend with improved styling
    legend = ax1.legend(fontsize=13, loc='upper right', frameon=True, 
                       framealpha=0.9, edgecolor='gray')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_boxstyle('round,pad=0.5')
    
    # ---- PSNR Plot ----
    ax2 = axes[2]
    
    # Plot train data
    ax2.plot(ITERS_METRICS, list(PSNRS_TRAIN.values()), '-', linewidth=3, 
             color=train_color)
    ax2.plot(ITERS_METRICS, list(PSNRS_TEST.values()), '--', linewidth=2.5, 
             color=test_color)
    
    # Add markers
    ax2.plot(ITERS_METRICS, list(PSNRS_TRAIN.values()), 'o', markersize=8, color=train_color, 
            markeredgecolor='white', markeredgewidth=1.5)
    ax2.plot(ITERS_METRICS, list(PSNRS_TEST.values()), 's', markersize=8, color=test_color,
            markeredgecolor='white', markeredgewidth=1.5)
    
    # Add fill below curves
    ax2.fill_between(ITERS_METRICS, list(PSNRS_TRAIN.values()), min(PSNRS_TRAIN.values())*0.9, 
                    alpha=0.2, color=train_color)
    ax2.fill_between(ITERS_METRICS, list(PSNRS_TEST.values()), min(PSNRS_TEST.values())*0.9, 
                    alpha=0.15, color=test_color)
    
    # Apply consistent aesthetics
    apply_aesthetics(ax2, "PSNR Trend (↑)", "", ylabel="PSNR (dB)")
    
    # Annotate best points
    max_train_value = max(PSNRS_TRAIN.values())
    max_train_idx = list(PSNRS_TRAIN.values()).index(max_train_value)
    max_train_iter = list(PSNRS_TRAIN.keys())[max_train_idx]
    
    max_test_value = max(PSNRS_TEST.values())
    max_test_idx = list(PSNRS_TEST.values()).index(max_test_value)
    max_test_iter = list(PSNRS_TEST.keys())[max_test_idx]
    
    train_annotation = ax2.annotate(f'Best: {max_train_value:.2f} dB',
                xy=(max_train_iter, max_train_value),
                xytext=(max_train_iter*0.13, max_train_value*0.95),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=train_color, 
                                connectionstyle="arc3,rad=-.2"),
                fontsize=13, color=train_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', alpha=0.8, ec=train_color))
    
    test_annotation = ax2.annotate(f'Best: {max_test_value:.2f} dB',
                xy=(max_test_iter, max_test_value),
                xytext=(max_test_iter*0.3, max_test_value*0.89),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=test_color, 
                                connectionstyle="arc3,rad=.2"),
                fontsize=13, color=test_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', alpha=0.8, ec=test_color))
    
    # Add legend with improved styling
    legend = ax2.legend(['Training', 'Testing'], fontsize=13, 
                        loc='lower right', frameon=True, framealpha=0.9, edgecolor='gray')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_boxstyle('round,pad=0.5')
    
    # Add improvement summary box
    train_improvement = (PSNRS_TRAIN[list(PSNRS_TRAIN.keys())[-1]] - PSNRS_TRAIN[list(PSNRS_TRAIN.keys())[0]]) 
    test_improvement = (PSNRS_TEST[list(PSNRS_TEST.keys())[-1]] - PSNRS_TEST[list(PSNRS_TEST.keys())[0]])
    
    summary_text = (f"Training improvement: +{train_improvement:.2f} dB\n"
                  f"Testing improvement: +{test_improvement:.2f} dB")
    
    summary_box = ax2.text(0.02, 0.05, summary_text, 
                         transform=ax2.transAxes,
                         fontsize=13,
                         bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='gray', alpha=0.9))
    
    # Add figure title
    # fig.suptitle("Training Performance Metrics", fontsize=22, fontweight='bold', y=0.98)
    
    # Final figure adjustments
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.95)
    plt.savefig('train_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_dataset_growth():
    """Plot dataset growth with improved aesthetics"""
    plt.figure(figsize=(12, 7), facecolor='white')
    
    # Create a custom color palette for plateaus
    colors = ['#303F9F', '#7B1FA2', '#C2185B', '#D32F2F', '#F57C00']
    
    # Identify plateaus in the data
    unique_points = []
    point_ranges = []
    last_point = None
    start_idx = 0
    
    for i, point in enumerate(POINTS.values()):
        if point != last_point:
            if last_point is not None:
                unique_points.append(last_point)
                point_ranges.append((start_idx, i-1))
            start_idx = i
            last_point = point
    
    # Add the last segment
    unique_points.append(last_point)
    point_ranges.append((start_idx, len(POINTS)-1))
    
    # Create a softer gradient background
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Plot segments with improved styling
    for i, ((start, end), point) in enumerate(zip(point_ranges, unique_points)):
        segment_x = list(POINTS.keys())[start:end+1]
        segment_y = [point for _ in range(end-start+1)]
        
        # Plot line
        plt.plot(segment_x, segment_y, '-', linewidth=4, color=colors[i % len(colors)])
        
        # Plot markers
        plt.plot(segment_x, segment_y, 'o', markersize=9, color=colors[i % len(colors)],
                markeredgecolor='white', markeredgewidth=1.5)
        
        # Add shaded area below for visual weight
        plt.fill_between(segment_x, [0]*len(segment_x), segment_y, 
                        alpha=0.1, color=colors[i % len(colors)])
        
        # Add plateau annotations with improved styling
        if start == end:  # Single point
            plt.annotate(f"{point:,}", 
                       (segment_x[0], point),
                       xytext=(0, 15),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i % len(colors)], alpha=0.9))
        else:  # Plateau
            mid_idx = (start + end) // 2
            mid_x = list(POINTS.keys())[mid_idx]
            
            label = f"{point:,} points"
            if end - start > 1:
                label += f"\n(iterations {segment_x[0]:,}-{segment_x[-1]:,})"
                
            text = plt.annotate(label, 
                       (mid_x, point),
                       xytext=(0, 20),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors[i % len(colors)], alpha=0.9))
            
            # Add outline to text for better visibility
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add arrows to show transitions with improved styling
    for i in range(1, len(unique_points)):
        prev_end = point_ranges[i-1][1]
        curr_start = point_ranges[i][0]
        
        # Only add arrow if there's an actual transition (not within a plateau)
        if unique_points[i] != unique_points[i-1]:
            start_x = list(POINTS.keys())[prev_end]
            end_x = list(POINTS.keys())[curr_start]
            start_y = unique_points[i-1]
            end_y = unique_points[i]
            
            # Calculate midpoint for positioning the annotation
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Draw arrow
            plt.annotate('', 
                       xy=(end_x, end_y),
                       xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='#546E7A',
                                      connectionstyle="arc3,rad=0.1"))
            
            # Add percent increase
            increase = (end_y - start_y) / start_y * 100
            
            # Position the text based on the arrow direction
            plt.text(mid_x, mid_y * 0.98,
                    f"+{increase:.1f}%", 
                    ha='center', va='center', fontweight='bold',
                    fontsize=13, color='#37474F',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#546E7A", alpha=0.9))
    
    # Apply consistent aesthetics
    apply_aesthetics(plt.gca(), "Points Growth", 
                     f"From {min(POINTS.values()):,} to {max(POINTS.values()):,} points",
                     ylabel="Number of Points")
    
    # Add a summary box with improved styling
    total_growth = (list(POINTS.values())[-1] - list(POINTS.values())[0]) / list(POINTS.values())[0] * 100
    
    summary_text = (f"Total growth: +{total_growth:.1f}%\n"
                   f"Final size: {list(POINTS.values())[-1]:,} points")
    
    plt.text(0.02, 0.97, summary_text,
             transform=plt.gca().transAxes,
             fontsize=13, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # Final adjustments
    plt.tight_layout(pad=2.0)
    plt.savefig('dataset_growth.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    # Set global plotting styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    
    # Generate plots
    plot_training_metrics()
    plot_dataset_growth()
    
    print("Plots generated successfully: train_metrics.png and dataset_growth.png")