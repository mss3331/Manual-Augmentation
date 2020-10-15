import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io, data
from PIL import Image


def modif_elastic(image, kernel_length, sigma):
    print("sigma is :",sigma)
    #location of the effect in the image
    x_y_shift = (270,70)

    shape = image.shape

    dx = np.zeros(shape)
    dy = np.zeros(shape)

    shperical_gaus_kernel_y_axis, shperical_gaus_kernel_x_axis = getKern(l=kernel_length, sig=sigma)


    gaussina_kernel_3D_x_xis = np.repeat(shperical_gaus_kernel_x_axis[:, :, np.newaxis], 3, axis=2)


    gaussina_kernel_3D_y_xis = np.repeat(shperical_gaus_kernel_y_axis[:, :, np.newaxis], 3, axis=2)

    dx[x_y_shift[1]:x_y_shift[1]+kernel_length,x_y_shift[0]:+x_y_shift[0]+kernel_length,:] = gaussina_kernel_3D_x_xis
    dy[x_y_shift[1]:x_y_shift[1]+kernel_length,x_y_shift[0]:+x_y_shift[0]+kernel_length,:] = gaussina_kernel_3D_y_xis
    dx=gaussian_filter((dx), sigma*3)
    dy=gaussian_filter((dy), sigma*3)
    print(dy.shape,dx.shape)
    show_3d(dy[50:300,50:450,:],dx[50:300,50:450,:])
    y, x, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)
def getKern_perfectImlementation(l=5, sig=1):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    kernel /= np.sum(kernel)
    '''End of basic gaussian kernel'''
    kernel =np.max(kernel)- kernel #flip the gaussian
    return kernel
def getKern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    kernel /= np.sum(kernel)
    '''End of basic gaussian kernel'''
    #--------------------------------------------------------------------------------------------------------
    '''We will try to use gaussian distribution with a sum of l**2'''
    # kernel*=l**2/np.sqrt(2) # this normalization to make the displacment of both dx and dy equal to l**2
    #
    # print("The diplacement sum should be",np.sum(kernel))
    # show_3d(kernel,kernel)
    # return kernel
    #---------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------
    ##kernel /= np.max(kernel) #normalize the guassian matrix to manipulate the highest diplacement
    ##highest_diplacement = 30
    ##total_displacment = highest_diplacement/np.sqrt(2)

    # kernel *= (total_displacement/sig)

    ##kernel = np.multiply(total_displacment,kernel)
    # show_3d(kernel,kernel)

    #kernel *= kernel * total_displacement  # always the sum of the displacements will be half the kernel width


    #kernel *= l**2
    ##max = np.max(kernel)
    #kernel /= max #to normalize the kernel
    # kernel = 1- kernel #to flip the gaussian
    #kernel *= int(l/2) #to make the farthest displacement equal to the quarter of the kernel width
    #print(kernel)

    ## kernel = max-kernel
    #------------------------------------------------------------------------------------------------------
    '''we want to minimize the diplacement as we approach the middle of the kenrel
        therefore, we are going to use distance matrix to always achieve this condition'''
    # distance_mat = createDistanceMatrix(l, l)
    # distance_mat = distance_mat ** (5 / sig)
    # distance_mat *= 30 / np.sqrt(2)
    # return distance_mat
    # kernel *= distance_mat**(5/sig)
    # show_3d(distance_mat**(5/sig),distance_mat**(5/sig))
    # print(distance_mat[:3, :3])
    # print(kernel[:3, :3])
    # kernel *= distance_mat
    # print(kernel[:3, :3])
    #kernel *= -1 #minimizing glass effect

    #cosine_diplacement =  sig/np.sqrt(2) - np.cos(distance_mat*(np.pi*2.5))*(sig/np.sqrt(2))
    # return kernel
    #show_3d(cosine_diplacement,cosine_diplacement)
    #return cosine_diplacement
    #--------------------------------------------------------------
    # #this one is good with gaussian (first candidate)
    distance_mat = createDistanceMatrix(l, l)

    distance_mat = 1-np.abs(np.cos(distance_mat*np.pi)**2)

    distance_mat *= (l/2) / np.sqrt(2)
    #distance_mat = -distance_mat #inward
    distance_mat = distance_mat #outward
    dx_kernel = sphere_kernel(distance_mat)
    dy_kernel = sphere_kernel(distance_mat,False)

    return -dy_kernel, -dx_kernel
    #--------------------------------------------------------------
    #-------------------------------------------------------------
    # This one is a second candidate
    # distance_mat = createDistanceMatrix(l, l)**0.2
    # distance_mat *=(l/2)/np.sqrt(2)
    # return distance_mat
    #----------------------------------------------------------------
    # This one is a deviation from the second candidate
    # distance_mat = createDistanceMatrix(l, l)
    # distance_mat *=(l/2)/np.sqrt(2)
    # return distance_mat
    #----------------------------------------------------------------

def sphere_kernel(kernel_orig,x_axis=True):
    '''This function accept guassian kernel and modify it to make it spherical'''
    kernel = copy.deepcopy(kernel_orig)

    y,x = kernel.shape
    # print("Before x_axis",x_axis)
    # print(kernel)
    # print(y,x)
    y_start = int(y/2)
    x_start = int(x/2)
    if x_axis:
        kernel[:, x_start:] = -kernel[:, x_start:]
    else:
        kernel[y_start:,:] = -kernel[y_start:,:]

    # # print("x_axis",x_axis,"\n",kernel[48:52,48:52])
    # print("x_axis",x_axis,"\n",kernel)
    return kernel
def show_3d(dy,dx):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    # X = np.arange(-np.pi, np.pi, 0.1)
    # Y = np.arange(-np.pi, np.pi, 0.1)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X ** 2 + Y ** 2)
    # Z = (np.cos(R)+1)/2
    if np.min(dx.shape) == 3:
        dx = dx[:,:,0]
        dy = dy[:,:,0]
    #print("strange:",dx.shape)
    y_length, x_length, = dx.shape

    X = np.arange(-np.floor(x_length/2),np.floor(x_length/2))
    Y = np.arange(-np.floor(y_length/2),np.floor(y_length/2))
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(dx ** 2 + dy ** 2)
    # print(X.shape,Y.shape,dx[:,:,0].shape)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(np.min(R),np.max(R))
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    return

def createDistanceMatrix(x_size,y_size):
    #x_size, y_size = 5, 5
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (int(x_size/2),int(y_size/2))
    #cell = (2,2)
    dists = np.sqrt((x_arr - cell[0]) ** 2 + (y_arr - cell[1]) ** 2)
    dists = np.divide(dists,np.max(dists)) #normalize distanc
    return dists
if __name__=="__main__":
    img = data.chelsea()
    img =Image.fromarray(img)
    for i in range(1,20):
        i=Image.fromarray(modif_elastic(np.asarray(img),kernel_length=100,sigma=i))
        i.show()

