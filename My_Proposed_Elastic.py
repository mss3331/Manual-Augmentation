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

def iso_elastic(image,outward,y_x_location, sigma,kernel_length=100, show_Mag_surf=False):
    '''sigma should be from 20 to 45'''
    # location of the effect in the image
    #x_y_shift = (270, 70)
    y_x_shift = list(y_x_location) # location of the elasticity
    #y_x_shift = [70, 270]
    shape = image.shape

    # check if the location of elastic within the image borders
    if (y_x_shift[0] + kernel_length) >= shape[0]:
        y_x_shift[0] = shape[0] - kernel_length
    if (y_x_shift[1] + kernel_length) >= shape[1]:
        y_x_shift[1] = shape[1] - kernel_length

    dx = np.zeros(shape)
    dy = np.zeros(shape)

    shperical_gaus_kernel_y_axis, shperical_gaus_kernel_x_axis = getKern(l=kernel_length, sig=sigma)

    gaussina_kernel_3D_x_xis = np.repeat(shperical_gaus_kernel_x_axis[:, :, np.newaxis], 3, axis=2)

    gaussina_kernel_3D_y_xis = np.repeat(shperical_gaus_kernel_y_axis[:, :, np.newaxis], 3, axis=2)

    if not outward:
        gaussina_kernel_3D_x_xis *= -1
        gaussina_kernel_3D_y_xis *= -1

    dx[y_x_shift[0]:y_x_shift[0] + kernel_length, y_x_shift[1]:+y_x_shift[1] + kernel_length,:] = gaussina_kernel_3D_x_xis
    dy[y_x_shift[0]:y_x_shift[0] + kernel_length, y_x_shift[1]:+y_x_shift[1] + kernel_length,:] = gaussina_kernel_3D_y_xis
    dx = gaussian_filter((dx), sigma)
    dy = gaussian_filter((dy), sigma)

    if show_Mag_surf:#show the magnitude graph surface
        show_3d(dy[50:300, 50:450, :], dx[50:300, 50:450, :])

    y, x, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def modif_elastic(image, kernel_length, sigma):
    print("sigma is :",sigma*3)
    #location of the effect in the image
    x_y_shift = (270,70)
    x_y_shift = (400,70)

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
    kernel *= l**2 #make the largest diplacement is the kernel size
    kernel /= np.sum(kernel)
    print(np.sum(kernel))
    dx_kernel = sphere_kernel(kernel)
    dy_kernel = sphere_kernel(kernel, False)

    return dy_kernel, dx_kernel
def getKern(l=5, sig=1.):
    #--------------------------------------------------------------
    # #this one is good with gaussian (first candidate)
    distance_mat = createDistanceMatrix(l, l)

    distance_mat = 1-np.abs(np.cos(distance_mat*np.pi)**2)

    distance_mat *= (l/2) / np.sqrt(2)
    #distance_mat = -distance_mat #inward
    distance_mat = distance_mat #outward
    dx_kernel = sphere_kernel(distance_mat)
    dy_kernel = sphere_kernel(distance_mat,False)

    return dy_kernel, dx_kernel


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
    ax.set_zlim(0,25)
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
    for i in range(7,16):
        print(i,)
        i=Image.fromarray(iso_elastic(np.asarray(img),outward=False,y_x_location=(0,400),sigma=i))
        i.show()

