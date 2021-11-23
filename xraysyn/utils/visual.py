import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MedicalImageAnimator(object):
    i = 0
    pause = False

    def __init__(
        self, data, annotation=[], dim=0, marker_size=[4, 10, 10], 
        name=None, save=None, text=False, cmap='gray'
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        dims = [0, 1, 2]
        dims.pop(dim)

        ant, = ax.plot([], [], "o", markersize=20, fillstyle='full')

        if dim == 0:
            img = ax.imshow(data[0, :, :], cmap=cmap, origin='lower')
            ax.yaxis.set_ticks(np.arange(0, data.shape[1], 10))
            ax.xaxis.set_ticks(np.arange(0, data.shape[2], 10))
        elif dim == 1:
            img = ax.imshow(data[:, 0, :], cmap=cmap, origin='lower')
            ax.yaxis.set_ticks(np.arange(0, data.shape[0], 5))
            ax.xaxis.set_ticks(np.arange(0, data.shape[2], 10))
        elif dim == 2:
            img = ax.imshow(data[:, :, 0], cmap=cmap, origin='lower')
            ax.yaxis.set_ticks(np.arange(0, data.shape[0], 5))
            ax.xaxis.set_ticks(np.arange(0, data.shape[1], 10))

        ax.axis('off')

        time_text = ax.text(
            0.03, 0.95, '',
            color='red',
            fontsize=24,
            horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes
        )
        if name:
            ax.set_title(name)

        fig.canvas.mpl_connect('key_press_event', self.onkey)

        self.data = data
        self.annotation = annotation
        self.marker_size = marker_size
        self.dim = dim
        self.dims = dims
        self.fig = fig
        self.ax = ax
        self.ant = ant
        self.img = img
        self.time_text = time_text
        self.save = save

    def onkey(self, event):
        key = event.key
        if key == 'a':
            self.pause ^= True
        return

    def update(self, data):
        if not self.pause:
            self.img.set_array(data)

            markers_x = []
            markers_y = []
            if np.array(self.annotation).size != 0:
                for index, location in enumerate(self.annotation[:, self.dim]):
                    if location - self.marker_size[self.dim] <= self.i and \
                    self.i <= location + self.marker_size[self.dim]:
                        markers_x.append(self.annotation[index, self.dims[1]])
                        markers_y.append(self.annotation[index, self.dims[0]])

            self.ant.set_data(markers_x, markers_y)
            self.time_text.set_text(
                'index = {}'.format(self.i % self.data.shape[self.dim])
            )
        return self.img, self.ant, self.time_text

    def generate_data(self):
        if self.dim == 0:
            data = self.data
        elif self.dim == 1:
            data = self.data.transpose(1, 0, 2)
        elif self.dim == 2:
            data = self.data.transpose(2, 0, 1)

        self.i = -1
        while self.i < data.shape[0] - 1:
            if not self.pause:
                self.i += 1
            yield data[self.i]

    def run(self, fps=15):
        plt.tight_layout()
        animate = animation.FuncAnimation(
            self.fig, self.update, self.generate_data,
            interval=50, blit=False, repeat=True,
            save_count=self.data.shape[self.dim]
        )
        if self.save is not None:
            animate.save(
                self.save, writer='imagemagick', fps=fps
            )
        else:
            plt.show()
        return animate
