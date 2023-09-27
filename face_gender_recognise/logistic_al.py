import numpy as np
import imageio
path_to_ar = "/home/lamnv/Documents/my carier/Marchine learning/supervise learning/data/AR"
img_ids_train = np.arange(1,26)
img_ids_test = np.arange(26,50)
view_ids = np.hstack((np.arange(1,8),np.arange(14,21)))
print(type(img_ids_train))

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
def build_data_matrix(img_ids,view_ids,pre_expected_shape,projection_matrix):
    """
    img_ids : the list of 
    """
    total_imgine = len(img_ids)*len(view_ids)
    x_full = np.zeros((total_imgine,pre_expected_shape))
    


   


