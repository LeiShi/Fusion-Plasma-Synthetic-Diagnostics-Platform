"""
Short script providing simple animation creation functions
"""

import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt

class MovieMaker:
    """create 2D movie from data
    Inputs:
        data: 3-dimension ndarray, shape (NT,NY,NX), NT is the time steps, NY,NX the vertical and horizontal pixel number

    """
    def __init__(self,data,interval=50,**im_para):

        self.data = data
        self.interval = interval
        self.frame = 0
        self.fig = plt.figure()
        self.im = plt.imshow(data[0],**im_para)
        self.t_tag = self.fig.text(0.9,0.9,'t=0',ha = 'right',va='top')


    def updatefig(self,t):
        self.im.set_array(self.data[t])
        self.t_tag.set_text('t={0}'.format(t))

    def showmovie(self):
        ani = anim.FuncAnimation(self.fig,self.updatefig,interval = self.interval)
        plt.show()

        
