# coding: utf-8

from visdom import Visdom
import numpy as np


###=========example========================
# vis =Visdom(env='expression_wind')
# # vis.text('Hello, world!')
# tr_loss = list(range(100))
# vis.line(Y = np.array(tr_loss),opts=dict(showlegend=True))
# ts_loss = list(range(10,110))
# vis.line(Y=np.column_stack((np.array(tr_loss),np.array(ts_loss))), opts=dict(showlegend=True))
##========================================



viz = Visdom(env='my_wind')
x,y,z=0,0,10
win = viz.line(
    X=np.array([x]),
    Y=np.column_stack((np.array([y]),np.array([z]))),
    opts=dict(title='two_lines', legend=['train', 'loss'], showlegend=True))
for i in range(100):
    x+=i
    y+=i
    z+=i*10
    viz.line(
        X=np.array([x]),
        Y=np.column_stack((np.array([y]),np.array([z]))),
        opts=dict(title='two_lines', legend=['train','loss'],showlegend=True),
        win=win,#win要保持一致
        update='append')