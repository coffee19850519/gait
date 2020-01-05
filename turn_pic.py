import scipy 
from scipy import misc
from scipy import ndimage


for i in range(1,46):
    im = misc.imread('./image/wy/进门/'+str(i)+'.jpg')
    img_90_right = ndimage.rotate(im,-90)
    scipy.misc.imsave('./image/wy/进门_旋转/'+str(i)+'_90_right.jpg',img_90_right)

    