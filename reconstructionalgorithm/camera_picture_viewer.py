import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm

#Class for showing reconstruted 3D data slices


class Camera_picture_viewer(object):

    #Initialization of class
    def __init__(self,cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,c_log,bits):

        #Figure with 4 subplots
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(10,4))

        self.c_log=c_log
        self.bits=bits


        #Subplot which shows top camera picture
        ax1.set_title('Top camera')
        if self.c_log==True:
            self.im_1=ax1.imshow(cam_pic_top,origin='upper',aspect='equal',cmap='jet',norm=LogNorm())
        else:
            self.im_1=ax1.imshow(cam_pic_top,origin='upper',aspect='equal',cmap='jet')
        ax1.set_xlabel('x in pixel')
        ax1.set_ylabel('z in pixel')

        #Subplot which shows 120 camera picture
        ax2.set_title('120° camera ')
        if self.c_log==True:
            self.im_2=ax2.imshow(cam_pic_120,origin='upper',aspect='equal',cmap='jet',norm=LogNorm())
        else:
            self.im_2=ax2.imshow(cam_pic_120,origin='upper',aspect='equal',cmap='jet')
        ax2.set_xlabel('x_120_deg in pixel')
        ax2.set_ylabel('z in pixel')

        #Subplot which shows 240 camera picture
        ax3.set_title('240° camera')
        if self.c_log==True:
            self.im_3=ax3.imshow(cam_pic_240,origin='upper',aspect='equal',cmap='jet',norm=LogNorm())
        else:
            self.im_3=ax3.imshow(cam_pic_240,origin='upper',aspect='equal',cmap='jet')
        ax3.set_xlabel('x_240_deg in pixel')
        ax3.set_ylabel('z in pixel')

        #Subplot which shows front camera picture
        ax4.set_title('Front camera')
        if self.c_log==True:
            self.im_4=ax4.imshow(cam_pic_front,origin='lower',aspect='equal',cmap='jet',norm=LogNorm())
        else:
            self.im_4=ax4.imshow(cam_pic_front,origin='lower',aspect='equal',cmap='jet')
        ax4.set_xlabel('x in pixel')
        ax4.set_ylabel('y in pixel')

        plt.colorbar(self.im_4,label='Intensity')


        #Axes and sliders for changing window for linear colormap
        axcolor = 'lightgoldenrodyellow'
        self.axmin = fig.add_axes([0.3, 0.01, 0.10, 0.03], facecolor=axcolor)
        self.axmax = fig.add_axes([0.7, 0.01, 0.10, 0.03], facecolor=axcolor)
        if self.c_log==True:
            self.smin = Slider(self.axmin, 'Min', 1, 2**self.bits, valinit=0)
            self.smax = Slider(self.axmax, 'Max', 1, 2**self.bits, valinit=2**self.bits)
        else:
            self.smin = Slider(self.axmin, 'Min', 1, 2**self.bits, valinit=0)
            self.smax = Slider(self.axmax, 'Max', 1, 2**self.bits, valinit=2**self.bits)

        #Set window for linear colormap
        self.im_1.set_clim([self.smin.val,self.smax.val])
        self.im_2.set_clim([self.smin.val,self.smax.val])
        self.im_3.set_clim([self.smin.val,self.smax.val])
        self.im_4.set_clim([self.smin.val,self.smax.val])

        #Connect different events to functions
        fig.canvas.mpl_connect('button_press_event', self.onclick)


        plt.tight_layout()
        plt.show()

        self.sminval=self.smin.val #the minimun value is the one you choose before closing the plot



    #Function for handling click events
    def onclick(self,event):

        #Check if zoom etc is selected
        mode = plt.get_current_fig_manager().toolbar.mode
        if mode == '':

            #Selecting min and max values for linear colormap
            if event.inaxes is self.axmin:
                self.update()
            elif event.inaxes is self.axmax:
                self.update()


    def update(self):

        #Update front camera picture
        self.im_1.set_clim([self.smin.val,self.smax.val])

        #Update top camera picture
        self.im_2.set_clim([self.smin.val,self.smax.val])

        #Update 120 camera picture
        self.im_3.set_clim([self.smin.val,self.smax.val])

        #Update 240 camera picture
        self.im_4.set_clim([self.smin.val,self.smax.val])



        #Redraw front camera picture
        self.im_1.axes.figure.canvas.draw()

        #Redraw top camera picture
        self.im_2.axes.figure.canvas.draw()

        #Redraw 120 camera picture
        self.im_3.axes.figure.canvas.draw()

        #Redraw 240 camera picture
        self.im_4.axes.figure.canvas.draw()
