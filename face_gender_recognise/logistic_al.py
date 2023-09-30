import numpy as np
import imageio
path_to_ar = "/home/lamnv/Documents/my carier/Marchine learning/supervise learning/data/AR"
img_ids_train = np.arange(1,26)
img_ids_test = np.arange(26,50)
view_ids = np.hstack((np.arange(1,8),np.arange(14,21)))


def build_list_fn(pre,img_ids, view_ids):
    """
    input :
         pre means "M for man , W for woman"
         img_ids  : array range from 1 to number of imagines
         view_id : 01-07 and 14-20 for visibility imagine , without facemark or grass
    output :
         list of file names [pre-xxx-yy.bmp] where xxx = img_id , yy = view_id
    """
    filename = []
    for img_id in img_ids:
        for view_id in view_ids:
            filename.append(path_to_ar+pre+str(img_id).zfill(3)+str(view_id).zfill(2)+'.bmp')
    return filename
# reduce picture dimension from 3*165*120 to  165*120
def rgb2grey(rgb,siz):
    grey=0
    for i in range(siz):
        grey += rgb[:,:,i]
    return grey
def vector_normal(filename,expected_shape):
    imagine = imageio.imread(filename)
    grey = rgb2grey(imagine)
    return grey.reshape(1,expected_shape)
def build_data_matrix(img_ids,view_ids,projection_matrix,pre_expected_shape=165*120):
    """
    img_ids : the 1-D array
    view_ids : the 1-D array
    pre_expected_shape : the number of dimension has been reduce
    projection_matrix : ( pre_expected_shape,final_size)
    """
    total_imgine = len(img_ids)*len(view_ids)*2
    x_full = np.zeros((total_imgine,pre_expected_shape))
    list_man_name = build_list_fn('M-',img_ids,view_ids)
    list_woman_name = build_list_fn('W-',img_ids,view_ids)
    list_fn = list_man_name+list_woman_name
    for i, name in enumerate(list_fn):
        x_full[i,:] = vector_normal(name)
    x_reduce = np.dot(x_full,projection_matrix)
    return x_reduce
def feature_engineering(x,x_mean,x_var):
    """
    using standardization to standard data  using mean and var
    x shape : (number of picture, feature)
    x_mean,x_var with axis =0
    """
    return (x-x_mean)/x_var


    


   


