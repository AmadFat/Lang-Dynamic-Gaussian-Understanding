import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class CurveCollapseVisualizer:
    def __init__(self, num_curves=10000, max_degree=3):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        
        self.x_range = np.linspace(-5, 5, 1000)
        self.observations = []
        self.num_curves = num_curves
        self.tolerance = 0.2
        
        # Generate random polynomial functions
        self.functions = []
        for _ in range(num_curves):
            degree = np.random.randint(1, max_degree + 1)
            coeffs = np.random.randn(degree + 1) * 2
            func = lambda x, c=coeffs: np.polyval(c, x)
            self.functions.append(func)
        
        # Add button for adding observations by clicking
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Reset button
        self.reset_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.reset_button = Button(self.reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        self.update_plot()
        plt.title("Click to add observations and constrain the functions")
    
    def is_function_valid(self, func):
        """Check if a function passes near all observation points"""
        if not self.observations:
            return True
        
        for x_obs, y_obs in self.observations:
            if abs(func(x_obs) - y_obs) > self.tolerance:
                return False
        return True
    
    def update_plot(self):
        self.ax.clear()
        
        # Identify valid functions
        valid_functions = [f for f in self.functions if self.is_function_valid(f)]
        
        # Plot all functions faintly
        for f in self.functions:
            y = f(self.x_range)
            self.ax.plot(self.x_range, y, 'gray', alpha=0.05)
        
        # Plot valid functions more prominently
        for f in valid_functions:
            y = f(self.x_range)
            self.ax.plot(self.x_range, y, 'blue', alpha=0.3)
        
        # Plot observations
        if self.observations:
            x_obs, y_obs = zip(*self.observations)
            self.ax.scatter(x_obs, y_obs, color='red', s=100, zorder=10, label='Observations')
        
        # Set plot limits and labels
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title(f'Function Uncertainty with {len(self.observations)} Observations')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        # Display stats
        valid_count = len(valid_functions)
        total_count = len(self.functions)
        self.ax.text(0.02, 0.98, f'Valid functions: {valid_count}/{total_count}', 
                    transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        
        if self.observations:
            self.ax.legend()
        
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        # Ignore clicks on buttons
        if event.inaxes != self.ax:
            return
            
        # Add observation where user clicked
        x_obs = event.xdata
        y_obs = event.ydata
        self.observations.append((x_obs, y_obs))
        self.update_plot()
    
    def reset(self, event):
        self.observations = []
        self.update_plot()

if __name__ == "__main__":
    visualizer = CurveCollapseVisualizer()
    plt.show()