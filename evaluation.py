import numpy as np

def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt( ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c) )
