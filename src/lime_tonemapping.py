import numpy as np
import cv2
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import bicgstab
from scipy.ndimage import zoom
from scipy.signal import convolve2d


def compute_weights(image, sigma, sharpness):
    '''
    Computes texture weights for an image based on horizontal and vertical differences.

    This function calculates texture weights for an image by analyzing horizontal and vertical
    gradients and applying a smoothing filter. The weights are used in applications such as
    low-light image enhancement to preserve texture and edges while reducing noise. The method
    is based on the LIME (Low-Light Image Enhancement via Illumination Map Estimation) framework.

    Parameters:
    -----------
    image : numpy.ndarray
        A 2D numpy array representing the grayscale image for which weights are computed.

    sigma : int
        The size of the smoothing filter kernel. A larger sigma results in stronger smoothing.

    sharpness : float
        A small constant added to the denominator to avoid division by zero and control the
        sensitivity of the weights to texture details. Lower values increase sensitivity.

    Returns:
    --------
    W_h : numpy.ndarray
        A 2D numpy array representing the horizontal texture weights.

    W_v : numpy.ndarray
        A 2D numpy array representing the vertical texture weights.

    References:
    -----------
    X. Guo, Y. Li and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation,"
    in IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, Feb. 2017,
    doi: 10.1109/TIP.2016.2639450.
    '''

    # Compute vertical and horizontal differences
    dt0_v = np.roll(image, -1, axis=0) - image
    dt0_v[-1, :] = image[0, :] - image[-1, :]

    dt0_h = np.roll(image, -1, axis=1) - image
    dt0_h[:, -1] = image[:, 0] - image[:, -1]

    # Create a filter kernel
    kernel_h = np.ones((1, sigma)) / sigma
    kernel_v = np.ones((sigma, 1)) / sigma

    # Apply the filter
    gauker_h = convolve2d(dt0_h, kernel_h, mode='same')
    gauker_v = convolve2d(dt0_v, kernel_v, mode='same')

    # Compute texture weights
    W_h = 1 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)

    return W_h, W_v


def solve_linear_equation(image, wx, wy, lambda_val):
    '''
    Solves a linear equation for low-light image enhancement.

    This function solves a linear equation to estimate the illumination map of a low-light image.
    The equation is based on the camera response model and is used to enhance the image by adjusting
    the illumination map. The method is inspired by the LIME (Low-Light Image Enhancement via Illumination
    Map Estimation) framework and the camera response model-based approach.

    Parameters:
    -----------
    image : numpy.ndarray
        A 2D numpy array representing the grayscale image to be enhanced.

    wx : numpy.ndarray
        A 2D numpy array representing the horizontal gradient of the image.

    wy : numpy.ndarray
        A 2D numpy array representing the vertical gradient of the image.

    lambda_val : float
        A regularization parameter that controls the strength of the smoothing term in the equation.

    Returns:
    --------
    image_out : numpy.ndarray
        A 2D numpy array representing the enhanced image.

    References:
    -----------
    X. Guo, Y. Li and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation,"
    in IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, Feb. 2017,
    doi: 10.1109/TIP.2016.2639450.

    Z. Ying, G. Li, Y. Ren, R. Wang and W. Wang, "A New Low-Light Image Enhancement Algorithm Using
    Camera Response Model," 2017 IEEE International Conference on Computer Vision Workshops (ICCVW),
    Venice, Italy, 2017, pp. 3015-3022, doi: 10.1109/ICCVW.2017.356.
    '''

    height, width = image.shape
    k = height * width

    dx = -lambda_val * wx.flatten()
    dy = -lambda_val * wy.flatten()

    tempx = np.concatenate([wx[:, -1:], wx[:, :-1]], axis=1)
    tempy = np.concatenate([wy[-1:, :], wy[:-1, :]], axis=0)
    dxa = -lambda_val * tempx.flatten()
    dya = -lambda_val * tempy.flatten()

    tempx = np.concatenate([wx[:, -1:], np.zeros((height, width - 1))], axis=1)
    tempy = np.concatenate([wy[-1:, :], np.zeros((height - 1, width))], axis=0)
    dxd1 = -lambda_val * tempx.flatten()
    dyd1 = -lambda_val * tempy.flatten()

    dxd2 = -lambda_val * wx.flatten()
    dyd2 = -lambda_val * wy.flatten()

    data = np.array([dxd1, dxd2, dyd1, dyd2, 1 - (dx + dy + dxa + dya)])
    diags = np.array([-k + height, -height, -height + 1, -1, 0])
    A = csc_matrix(spdiags(data, diags, k, k))

    i = [0]
    f = lambda x: (i.append(i[-1] + 1), i.pop(0), print(i[-1], x))

    tin = image.flatten()
    tout, _ = bicgstab(A, tin, atol=1e-5, callback=f)
    image_out = tout.reshape((height, width))

    return image_out


def LECARM(image, camera_model='sigmoid', downsampling=0.5, scaling=1):
    '''
    Enhances a low-light image using the LECARM (Low-Light Image Enhancement Using the Camera Response Model) algorithm.

    This function takes a low-light image as input and applies the LECARM algorithm to enhance its brightness and visibility.
    The algorithm uses a camera response model to estimate the illumination map of the image and then applies a brightness
    transfer function to adjust the image's brightness. The method is based on the LECARM framework and can be used with
    different camera models.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    camera_model : str, optional
        The camera response model to use for enhancement. Can be one of 'sigmoid', 'gamma', 'betagamma', or 'preferred'.
        Default is 'sigmoid'.

    downsampling : float, optional
        The downsampling ratio for the illumination map estimation. Default is 0.5.

    scaling : float, optional
        The scaling factor for the brightness transfer function. Default is 1.

    Returns:
    --------
    image_out : numpy.ndarray
        A 3D numpy array representing the enhanced image in BGR format.

    References:
    -----------
    Y. Ren, Z. Ying, T. H. Li and G. Li, "LECARM: Low-Light Image Enhancement Using the Camera Response Model,"
    in IEEE Transactions on Circuits and Systems for Video Technology, vol. 29, no. 4, pp. 968-981, April 2019,
    doi: 10.1109/TCSVT.2018.2828141.
    '''

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    illumination = hsv_image[:, :, 2].astype(np.float32) / 255

    ratio_max = 7
    lambda_val = 0.15
    sigma = 2
    sharpness = 0.001

    # Apply the LIME estimator
    T_downsampled = zoom(illumination, downsampling, order=0)
    wx, wy = compute_weights(T_downsampled, sigma, sharpness)
    T_estimated = solve_linear_equation(T_downsampled, wx, wy, lambda_val)
    T_upsampled = zoom(T_estimated, 1 / downsampling, order=0)

    # Compute the scaling factor K
    K = np.minimum(1.0 / T_upsampled, ratio_max)
    K = np.tile(K[:, :, np.newaxis], (1, 1, image.shape[2]))

    # Apply the brightness transfer function
    image_out = image.astype(np.float32) / 255

    if camera_model == 'sigmoid':
        n = 0.90
        sigma = 0.60
        image_out = ((sigma + sigma ** 2) * scaling ** n * image_out) / (scaling ** n * sigma * image_out + (1 + sigma - image_out) * sigma)
    elif camera_model == 'gamma':
        image_out = image_out * np.power(scaling, 0.8)
    elif camera_model == 'betagamma':
        beta = np.exp((1 - np.power(scaling, -0.3293)) * 1.1258)
        gamma = np.power(scaling, -0.3293)
        image_out = np.power(image_out, gamma) * beta
    else:  # preferred
        cf = np.power(image_out, 1 / 0.14)
        ka = np.power(scaling, 4.35)
        image_out = (cf * ka / (cf * (ka - 1) + 1)) ** 0.14

    return np.clip(255 * image_out, 0, 255).astype(np.uint8)
