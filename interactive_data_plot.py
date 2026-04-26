import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

def interactive_plot(data):
    # Get dimensions from data
    Ray, Angle, Wavelength_dim, dim_1, dim_2 = data.shape

    # Initial indices for Wavelength, dim_1, dim_2
    Wavelength_idx = 0
    current_dim_1 = 0
    current_dim_2 = 0
    Ray_idx = 0
    Angle_idx = 0

    # Create the figure and the plot
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.9, wspace=0.4)

    # 2D plot setup
    img = ax_img.imshow(data[:, :, Wavelength_idx, current_dim_1, current_dim_2], cmap='viridis', vmin=-1, vmax=1)
    ax_img.set_xlabel('Angle')
    ax_img.set_ylabel('Ray')
    plt.colorbar(img, ax=ax_img)

    # Create a 3x3 grid of RadioButtons for d1 and d2
    radio_buttons_d1 = plt.axes([0.05, 0.45, 0.2, 0.3], facecolor='lightgoldenrodyellow')
    radio_buttons_d2 = plt.axes([0.05, 0.15, 0.2, 0.3], facecolor='lightgoldenrodyellow')

    # Define labels for the radio buttons
    labels_d1 = ['Px', 'Py', 'Pz']
    labels_d2 = ['Ax', 'Ay', 'Az']

    radio_d1 = RadioButtons(radio_buttons_d1, labels_d1)
    radio_d2 = RadioButtons(radio_buttons_d2, labels_d2)

    # Title for RadioButtons
    ax_title = plt.axes([0.05, 0.75, 0.2, 0.05], facecolor='lightgoldenrodyellow')
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    title_text = ax_title.text(0.5, 0.5, f'Selected: ({labels_d1[current_dim_1]},{labels_d2[current_dim_2]})', ha='center', va='center', fontsize=12)

    # Buttons for incrementing and decrementing the Wavelength value
    ax_button_down = plt.axes([0.3, 0.05, 0.1, 0.05], facecolor='lightgoldenrodyellow')
    ax_button_up = plt.axes([0.45, 0.05, 0.1, 0.05], facecolor='lightgoldenrodyellow')

    button_up = Button(ax_button_up, 'Up')
    button_down = Button(ax_button_down, 'Down')

    # Create an empty line plot for 1D data
    line, = ax_plot.plot([], [], 'r-')
    ax_plot.set_xlabel('Wavelength dimension')
    ax_plot.set_ylabel('Data value')

    # Add a vertical line to the 1D plot to indicate the chosen Wavelength
    vertical_line = ax_plot.axvline(x=Wavelength_idx, color='blue', linestyle='--')

    # Function to update both 2D and 1D plots
    def update_plots():
        try:
            # Update the 2D plot
            img.set_data(data[:, :, Wavelength_idx, current_dim_1, current_dim_2])
            # Update the 1D plot if Ray_idx and Angle_idx are valid
            if 0 <= Ray_idx < Ray and 0 <= Angle_idx < Angle:
                line.set_xdata(np.arange(Wavelength_dim))
                line.set_ydata(data[Ray_idx, Angle_idx, :, current_dim_1, current_dim_2])
                ax_plot.set_title(f'1D Plot for Ray={Ray_idx}, Angle={Angle_idx}, dim_1={current_dim_1}, dim_2={current_dim_2}')
                ax_plot.relim()
                ax_plot.autoscale_view()
                # Update the vertical line position
                vertical_line.set_xdata([Wavelength_idx, Wavelength_idx])
            fig.canvas.draw_idle()
        except KeyError:
            print(f"KeyError: {current_dim_1}, {current_dim_2} not found in data.")

    # Click event function to plot 1D data along dimension Wavelength
    def onclick(event):
        nonlocal Ray_idx, Angle_idx
        if event.inaxes == ax_img:
            Ray_idx = int(np.clip(event.ydata, 0, Ray - 1))
            Angle_idx = int(np.clip(event.xdata, 0, Angle - 1))
            update_plots()

    # Callback function for the Up button
    def button_up_click(event):
        nonlocal Wavelength_idx
        Wavelength_idx = min(Wavelength_idx + 1, Wavelength_dim - 1)
        update_plots()

    # Callback function for the Down button
    def button_down_click(event):
        nonlocal Wavelength_idx
        Wavelength_idx = max(Wavelength_idx - 1, 0)
        update_plots()

    # Callback function for the RadioButtons
    def radio_d1_changed(label):
        nonlocal current_dim_1
        current_dim_1 = labels_d1.index(label)
        title_text.set_text(f'Selected: ({labels_d1[current_dim_1]},{labels_d2[current_dim_2]})')
        update_plots()

    def radio_d2_changed(label):
        nonlocal current_dim_2
        current_dim_2 = labels_d2.index(label)
        title_text.set_text(f'Selected: ({labels_d1[current_dim_1]},{labels_d2[current_dim_2]})')
        update_plots()

    # Connect the radio buttons to their callback functions
    radio_d1.on_clicked(radio_d1_changed)
    radio_d2.on_clicked(radio_d2_changed)

    # Connect the button increment and decrement functions
    button_up.on_clicked(button_up_click)
    button_down.on_clicked(button_down_click)

    # Connect the click event to the imshow plot
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Generate some random data for demonstration
    Ray1, Angle, Wavelength_dim, dim_1, dim_2 = 10, 12, 5, 3, 3
    data = np.random.rand(Ray1, Angle, Wavelength_dim, dim_1, dim_2)

    # Call the function with the generated data
    interactive_plot(data)
