import numpy as np
import cv2


def white_balance_gray_world(image):
    '''
    Applies white balance to an image using the Gray-World assumption.

    This function takes an image as input and applies the Gray-World assumption to estimate the color temperature
    of the scene and adjust the image's white balance accordingly. The method is based on the idea that the average
    color of the scene is gray, and uses a power transformation to adjust the color balance of the image.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    Returns:
    --------
    image_out : numpy.ndarray
        A 3D numpy array representing the output image with adjusted white balance in BGR format.

    References:
    -----------
    Vonikakis, Vassilios & Arapakis, & Andreadis,. (2011). Combining Gray-World assumption, White-Point correction 
    and power transformation for automatic white balance.
    '''

    image_out = image.astype(np.float32)

    mean_r = np.mean(image_out[:, :, 2])
    mean_g = np.mean(image_out[:, :, 1])
    mean_b = np.mean(image_out[:, :, 0])

    # Mean value of all channels
    gain = np.mean((mean_r, mean_g, mean_b)) / 255

    max_r = np.max(image_out[:, :, 2])
    max_g = np.max(image_out[:, :, 1])
    max_b = np.max(image_out[:, :, 0])

    # logarithm base to which each channel will be raised
    base_r = mean_r / max_r
    base_g = mean_g / max_g
    base_b = mean_b / max_b
    
    # the power to which each channel will be raised
    power_r = np.log(gain) / np.log(base_r)
    power_g = np.log(gain) / np.log(base_g)
    power_b = np.log(gain) / np.log(base_b)
    
    # separately applying different color correction powers to each channel
    image_out[:, :, 2] = (image_out[:, :, 2] / max_r) ** power_r
    image_out[:, :, 1] = (image_out[:, :, 1] / max_g) ** power_g
    image_out[:, :, 0] = (image_out[:, :, 0] / max_b) ** power_b

    return (255 * image_out).astype(np.uint8)


def white_balance_percentile(image, percentile=97.5):
    '''
    Applies white balance to an image using a percentile-based approach.

    This function takes an image as input and applies a white balance correction based on the percentile values
    of each color channel. The method calculates the gain required to bring the percentile value of each channel
    to the maximum possible value (255), and then applies this gain to each channel.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    percentile : float, optional
        The percentile value to use for calculating the gain. Default is 97.5.

    Returns:
    --------
    image_out : numpy.ndarray
        A 3D numpy array representing the output image with adjusted white balance in BGR format.
    '''

    image_out = image.astype(np.float32)

    # Calculate the percentile values for each channel
    gain = 255 / np.percentile(image, percentile, axis=(0, 1))

    # Apply the gain to each color channel
    image_out[:, :, 2] *= gain[2]
    image_out[:, :, 1] *= gain[1]
    image_out[:, :, 0] *= gain[0]

    return np.clip(image_out, 0, 255).astype(np.uint8)


def white_balance_lab(image):
    '''
    Applies white balance to an image using the LAB color space.

    This function takes an image as input, converts it to the LAB color space, and applies a white balance correction
    by normalizing the a* and b* components. The method is based on the idea of setting the mean of the a* and b*
    components to zero, which helps to remove any color casts and improve the overall color balance of the image.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    Returns:
    --------
    image_out : numpy.ndarray
        A 3D numpy array representing the output image with adjusted white balance in BGR format.

    References:
    -----------
    Bianco, G. and Muzzupappa, M. and Bruno, F. and Garcia, R. and Neumann, L. (2015). A NEW COLOR CORRECTION METHOD 
    FOR UNDERWATER IMAGING. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 
    XL-5/W5, 25-32. doi: 10.5194/isprsarchives-XL-5-W5-25-2015
    '''

    # Convert the image from BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Reset alpha and beta range from [0, 255] to [-128, 127]
    lab_image[:, :, 1:] -= 128

    # Ensure the values are within the valid range for LAB to BGR conversion
    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1] - np.mean(lab_image[:, :, 1]), -128, 127)  # a* component
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2] - np.mean(lab_image[:, :, 2]), -128, 127)  # b* component

    # Reset alpha and beta range from [-128, 127] to [0, 255]
    lab_image[:, :, 1:] += 128

    # Convert the modified LAB image back to BGR
    return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)