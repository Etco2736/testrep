import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Generate some example data for multiple lines
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi / 4)

# Create a custom colormap that transitions from purple to yellow
cmap = LinearSegmentedColormap.from_list("PurpleYellow", ["purple", "yellow"])
norm = plt.Normalize(x.min(), x.max())

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.5, 1.5)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Animating Lines with Color Gradient from Purple to Yellow")

# Function to create a gradient line collection for a portion of the line
def create_gradient_line(x, y, end_idx):
    points = np.array([x[:end_idx], y[:end_idx]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x[:end_idx])  # Set colors based on the x-values
    lc.set_linewidth(2)
    
    return lc

# Update function for the animation
def update(frame):
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-1.5, 1.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Animating Lines with Color Gradient from Purple to Yellow")

    # Create the partial lines up to the current frame (building effect)
    lc1 = create_gradient_line(x, y1, frame)
    lc2 = create_gradient_line(x, y2, frame)
    lc3 = create_gradient_line(x, y3, frame)

    # Add the lines to the plot
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x), interval=50)
ani.save("testplot.mp4", fps=50)
