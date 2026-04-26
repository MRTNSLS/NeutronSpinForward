from plot_tools import *
from matplotlib.widgets import Slider, Button

ver = 105
with np.load(f'data/meta_{ver}.npz', allow_pickle=True) as data:
    im_size = data['im_size']
    voxel_index = data['voxel_index']
    voxel_data = data['voxel_data']
    nNeutrons = data['nNeutrons']
    nAngles = data['nAngles']
    angles = data['angles']
    source_xs = data['source_xs']
    source_zs = data['source_zs']
    det_xs = data['det_xs']
    det_zs = data['det_zs']

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# %
# Create a function to update the plot based on the selected data
def update(val):
    choice_n = int(slider_n.val)
    choice_a = int(slider_a.val)
    # ax.clear()
    try:
        ax.lines[0].remove()
    except IndexError:
        pass
    
            
    plot_grid(im_size, ax)

    vi = voxel_index[choice_a, choice_n]
    vd = voxel_data[choice_a, choice_n]

    plot_path(im_size, ax, vi,
            source_xs[choice_a, choice_n], source_zs[choice_a, choice_n],
            det_xs[choice_a, choice_n], det_zs[choice_a, choice_n])
    ax.set_xlim(0 - 1, im_size + 1)
    ax.set_ylim(0 - 1, im_size + 1)
    # ax.set_xlim(min(np.concatenate((source_xs, det_xs))), max(np.concatenate((source_xs, det_xs))))
    # ax.set_ylim(min(np.concatenate((source_zs, det_zs))), max(np.concatenate((source_zs, det_zs))))

    texta.set_text(f'angle: {angles[choice_a]:.2f} radians')
    
    # Ensure that the grid is square (equal aspect ratio)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f"{vi}\n{[ round(elem, 2) for elem in vd ]}", fontsize=10)
    plt.draw()


# Create a slider to select the data
axcolor = 'lightgoldenrodyellow'
ax_slider_n = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_n = Slider(ax_slider_n, 'n', 0, nNeutrons-1, valinit=0, valstep=1)
ax_slider_a = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
slider_a = Slider(ax_slider_a, 'a', 0, nAngles-1, valinit=0, valstep=1)
slider_n.on_changed(update)
slider_a.on_changed(update)


def forwardn(v1):
    nb = slider_n.val
    nb2 = nb+1
    if nb2 < nNeutrons:
        slider_n.set_val(nb2)
def backwardn(v1):
    nb = slider_n.val
    nb2 = nb-1
    if nb2 >= 0:
        slider_n.set_val(nb2)
def forwarda(v1):
    nb = slider_a.val
    nb2 = nb+1
    if nb2 < nAngles:
        slider_a.set_val(nb2)
def backwarda(v1):
    nb = slider_a.val
    nb2 = nb-1
    if nb2 >= 0:
        slider_a.set_val(nb2)


ax_slider_nb1 = plt.axes([0.95, 0.1, 0.015, 0.03], facecolor=axcolor)
ax_slider_nb2 = plt.axes([0.97, 0.1, 0.015, 0.03], facecolor=axcolor)
button1n = Button(ax_slider_nb1, '<', color='w', hovercolor='b')
button2n = Button(ax_slider_nb2, '>', color='w', hovercolor='b')
button1n.on_clicked(backwardn)
button2n.on_clicked(forwardn)

ax_slider_ab1 = plt.axes([0.95, 0.05, 0.015, 0.03], facecolor=axcolor)
ax_slider_ab2 = plt.axes([0.97, 0.05, 0.015, 0.03], facecolor=axcolor)
button1a = Button(ax_slider_ab1, '<', color='w', hovercolor='b')
button2a = Button(ax_slider_ab2, '>', color='w', hovercolor='b')
button1a.on_clicked(backwarda)
button2a.on_clicked(forwarda)


ax_angle_text = plt.axes([0.25, 0.019, 0.65, 0.03], facecolor=axcolor)
ax_angle_text.set_axis_off()
texta = ax_angle_text.text(0.25,0.5,f'angle: {angles[0]:.2f} radians')

# Show the initial plot
update(0)
plt.show()




# print(voxel_index[choice_n])
# print(voxel_data[choice_n])

# %
