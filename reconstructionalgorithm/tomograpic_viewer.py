import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
import numpy as np
#Class for showing reconstruted 3D data slices


class Tomograpic_viewer(object):
    """Short summary.

    Parameters
    ----------
    X : array
        3D data array.
    c_log : bollena
        If True norm=LogNorm().
    max_val : type
        Description of parameter `max_val`.

    Attributes
    ----------
    slices_x : type
        Description of attribute `slices_x`.
    slices_y : type
        Description of attribute `slices_y`.
    slices_z : type
        Description of attribute `slices_z`.
    ind_x : type
        Description of attribute `ind_x`.
    ind_y : type
        Description of attribute `ind_y`.
    ind_z : type
        Description of attribute `ind_z`.
    fig : type
        Description of attribute `fig`.
    ax1 : type
        Description of attribute `ax1`.
    ax2 : type
        Description of attribute `ax2`.
    ax3 : type
        Description of attribute `ax3`.
    im_x : type
        Description of attribute `im_x`.
    line_y1 : type
        Description of attribute `line_y1`.
    line_z1 : type
        Description of attribute `line_z1`.
    im_y : type
        Description of attribute `im_y`.
    line_x2 : type
        Description of attribute `line_x2`.
    line_z2 : type
        Description of attribute `line_z2`.
    im_z : type
        Description of attribute `im_z`.
    line_x3 : type
        Description of attribute `line_x3`.
    line_y3 : type
        Description of attribute `line_y3`.
    cbar : type
        Description of attribute `cbar`.
    axmin : type
        Description of attribute `axmin`.
    axmax : type
        Description of attribute `axmax`.
    smin : type
        Description of attribute `smin`.
    smax : type
        Description of attribute `smax`.
    onscroll : type
        Description of attribute `onscroll`.
    onclick : type
        Description of attribute `onclick`.
    X
    c_log
    max_val

    """

    #Initialization of class
    def __init__(self,X,c_log,max_val):

        #3D data matrix
        self.X = X
        self.c_log=c_log
        self.max_val=max_val

        #Dimensions of 3D data matrix
        self.slices_x=X.shape[2]
        self.slices_y=X.shape[1]
        self.slices_z=X.shape[0]



        #dimension of 3d data matrix
        self.ind_x=self.slices_x//2
        self.ind_y=self.slices_y//2
        self.ind_z=0




        #Figure with 3 subplots
        self.fig, (self.ax1,self.ax2,self.ax3) = plt.subplots(1, 3,figsize=(10,4))

        #Subplot wich shows x slice of 3D Volume
        self.ax1.set_title('x-Slice {}/{}'.format(self.ind_x,self.slices_x-1))
        if self.c_log==True:
            self.im_x = self.ax1.imshow(self.X[:,:,self.ind_x],cmap='jet',aspect="equal",norm=LogNorm(),origin='lower')#change to aspect =3.24 if rcf data are shown
        else:
            self.im_x = self.ax1.imshow(self.X[:,:,self.ind_x],cmap='jet',aspect="equal",origin='lower')#change to aspect =3.24 if rcf data are shown
        self.line_y1=self.ax1.axvline(x=self.ind_y,color='red')
        self.line_z1=self.ax1.axhline(y=self.ind_z,color='red')
        self.ax1.set_xlabel('y in pixel')
        self.ax1.set_ylabel('z in pixel')

        #Subplot wich shows y slice of 3D Volume
        self.ax2.set_title('y-Slice {}/{}'.format(self.ind_y,self.slices_y-1))
        if self.c_log==True:
            self.im_y = self.ax2.imshow(self.X[:,self.ind_y,:],cmap='jet',aspect="equal",norm=LogNorm(),origin='lower') #change to aspect =3.24 if rcf data are shown
        else:
            self.im_y = self.ax2.imshow(self.X[:,self.ind_y,:],cmap='jet',aspect="equal",origin='lower')#change to aspect =3.24 if rcf data are shown

        self.line_x2=self.ax2.axvline(x=self.ind_x,color='red')
        self.line_z2=self.ax2.axhline(y=self.ind_z,color='red')
        self.ax2.set_xlabel('x in pixel')
        self.ax2.set_ylabel('z in pixel')

        #Subplot wich shows z slice of 3D Volume
        self.ax3.set_title('z-Slice {}/{}'.format(self.ind_z,self.slices_z-1))
        if self.c_log==True:
            self.im_z = self.ax3.imshow(self.X[self.ind_z,:,:],cmap='jet',aspect="equal",origin='lower',norm=LogNorm())
        else:
            self.im_z = self.ax3.imshow(self.X[self.ind_z,:,:],cmap='jet',aspect="equal",origin='lower')
        self.line_x3=self.ax3.axvline(x=self.ind_x,color='red')
        self.line_y3=self.ax3.axhline(y=self.ind_y,color='red')
        self.ax3.set_xlabel('x in pixel')
        self.ax3.set_ylabel('y in pixel')
        #self.ax3.axhline(y=10,color='black')



        #Colormap
        self.cbar=plt.colorbar(self.im_z,label='Relative Intensity')
        plt.tight_layout()

        #Axes and sliders for changing window for linear colormap
        axcolor = 'lightgoldenrodyellow'
        self.axmin = self.fig.add_axes([0.3, 0.01, 0.10, 0.03], facecolor=axcolor)
        self.axmax = self.fig.add_axes([0.7, 0.01, 0.10, 0.03], facecolor=axcolor)
        if self.c_log==True:
            self.smin = Slider(self.axmin, 'Min', 1,max_val, valinit=0)
            self.smax = Slider(self.axmax, 'Max', 1,max_val, valinit=max_val)
        else:
            self.smin = Slider(self.axmin, 'Min', 0, max_val, valinit=0)
            self.smax = Slider(self.axmax, 'Max', 0, max_val, valinit=max_val)

        #Set window for linear colormap
        self.im_x.set_clim([self.smin.val,self.smax.val])
        self.im_y.set_clim([self.smin.val,self.smax.val])
        self.im_z.set_clim([self.smin.val,self.smax.val])

        #Connect different events to functions
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)


        plt.show()

    #Function for handling click events
    def onclick(self,event):

        #Check if zoom etc is selected
        mode = plt.get_current_fig_manager().toolbar.mode
        if mode == '':

            #Selecting slices by clicking into subplots
            if event.inaxes is self.ax1:
                self.ind_y = int(event.xdata)
                self.ind_z = int(event.ydata)
                self.update()
            elif event.inaxes is self.ax2:
                self.ind_x = int(event.xdata)
                self.ind_z = int(event.ydata)
                self.update()
            elif event.inaxes is self.ax3:
                self.ind_x = int(event.xdata)
                self.ind_y = int(event.ydata)
                self.update()

            ##Selecting min and max values for linear colormap
            elif event.inaxes is self.axmin:
                self.update()
            elif event.inaxes is self.axmax:
                self.update()

    #Function for handling scroll events
    def onscroll(self, event,):

        #Selecting x-slice by scrolling in subplot
        if event.inaxes is self.ax1:
            if event.button == 'up':
                self.ind_x = (self.ind_x + 1) % self.slices_x
            else:
                self.ind_x = (self.ind_x - 1) % self.slices_x
            self.update()

        #Selecting y-slice by scrolling in subplot
        elif event.inaxes is self.ax2:
            if event.button == 'up':
                self.ind_y = (self.ind_y + 1) % self.slices_y
            else:
                self.ind_y = (self.ind_y - 1) % self.slices_y
            self.update()

        #Selecting z-slice by scrolling in subplot
        elif event.inaxes is self.ax3:
            if event.button == 'up':
                self.ind_z = (self.ind_z + 1) % self.slices_z
            else:
                self.ind_z = (self.ind_z - 1) % self.slices_z
            self.update()


    def update(self):

        #Update x-slice subplot
        self.ax1.set_title('x-Slice {}/{}'.format(self.ind_x,self.slices_x-1))
        self.im_x.set_data(self.X[:,:,self.ind_x])
        self.im_x.set_clim([self.smin.val,self.smax.val])
        self.line_y1.set_xdata(self.ind_y)
        self.line_z1.set_ydata(self.ind_z)

        #Update y-slice subplot
        self.ax2.set_title('y-Slice {}/{}'.format(self.ind_y,self.slices_y-1))
        self.im_y.set_data(self.X[:,self.ind_y,:])
        self.im_y.set_clim([self.smin.val,self.smax.val])
        self.line_x2.set_xdata(self.ind_x)
        self.line_z2.set_ydata(self.ind_z)

        #Update z-slice subplot
        self.ax3.set_title('z-Slice {}/{}'.format(self.ind_z,self.slices_z-1))
        self.im_z.set_data(self.X[self.ind_z,:,:])
        self.im_z.set_clim([self.smin.val,self.smax.val])
        self.line_x3.set_xdata(self.ind_x)
        self.line_y3.set_ydata(self.ind_y)

        #Redraw x-slice subplot
        self.im_x.axes.figure.canvas.draw()
        self.line_y1.axes.figure.canvas.draw()
        self.line_z1.axes.figure.canvas.draw()

        #Redraw y-slice subplot
        self.im_y.axes.figure.canvas.draw()
        self.line_x2.axes.figure.canvas.draw()
        self.line_z2.axes.figure.canvas.draw()

        #Redraw z-slice subplot
        self.im_z.axes.figure.canvas.draw()
        self.line_x3.axes.figure.canvas.draw()
        self.line_y3.axes.figure.canvas.draw()
