import matplotlib
# Try setting a different backend explicitly
matplotlib.use('TkAgg')  # Alternatives: 'Qt5Agg', 'Agg'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects



EFMS = OrderedDict({
    5: 0.5661648506671655,
    200: 0.486585219478738,
    1000: 0.7141839693228584,
    5000: 0.8380303225152762,
    20000: 0.8376255709097145,
})

if __name__ == "__main__":
    print("Starting visualization...")
    
    # Reset matplotlib to clear any previous state
    plt.close('all')
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure and axis with explicit size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data points
    x = list(EFMS.keys())
    y = list(EFMS.values())
    print(f"Plotting data points: x={x}, y={y}")
    
    # Basic plot to check data visibility
    ax.plot(x, y, marker='o', markersize=10, linewidth=2.5, color='#3498db')
    
    # Set the scale for x-axis to log (this spreads out your points nicely)
    ax.set_xscale('log')
    
    # Set explicit limits with some padding
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(x)*0.5, max(x)*1.2)
    
    # Add scatter points with gradient colors
    cmap = sns.color_palette("viridis", as_cmap=True)
    colors = cmap(np.linspace(0.1, 0.9, len(x)))
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"Scatter point: x={xi}, y={yi}, color={colors[i]}")
        ax.scatter(xi, yi, color=colors[i], s=150, zorder=5, 
                   edgecolor='white', linewidth=1.5)
        
        # Add value annotations
        txt = ax.annotate(f"{yi:.3f}", (xi, yi), 
                         textcoords="offset points",
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=10,
                         fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Labels and title
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')  # Changed to "Iteration" to match your description
    ax.set_ylabel('EFM Factor', fontsize=14, fontweight='bold')  # Changed to "EFM Value" for clarity
    ax.set_title('EFM Factor Growth: flame_salmon_1', fontsize=16, fontweight='bold', pad=20)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve aesthetics
    plt.tight_layout()
    sns.despine(left=False, bottom=False)
    
    # Save the plot
    savepath = 'event_flow_matching_growth.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {savepath}")
    
    # Show the plot
    print("Attempting to display plot...")
    plt.show()
    print("Script completed")