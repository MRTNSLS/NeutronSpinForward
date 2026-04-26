import ray_wrapper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
import numpy as np


def plot_grid(n, ax):


    # Create a grid of pixels with size n by n
    for i in range(n):
        for j in range(n):
            rect = patches.Rectangle((i, j), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            ax.text(j, i, str(i * n + j), ha='left', va='bottom', fontsize=8, color='black')


def plot_path(n, ax, colored_pixels, source_x, source_z, det_x, det_z):

    # Color the pixels specified in the colored_pixels list as semi-transparent gray
    for pixel_number in colored_pixels:
        row = pixel_number // n
        col = pixel_number % n
        rect = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor='0.7', alpha=0.5)
        ax.add_patch(rect)

    # Overlay a red line on the grid
    line_x = [source_x+n*0.5, det_x+n*0.5]
    line_y = [source_z+n*0.5, det_z+n*0.5]
    ax.plot(line_x, line_y, color='red')
    ax.arrow(sum(line_x)/2, sum(line_y)/2, (line_x[1]-line_x[0])*0.01,  (line_y[1]-line_y[0])*0.01, shape='full', lw=0, length_includes_head=True, head_width=.5, color='red')
    # print(line_x)
    # print(line_y)