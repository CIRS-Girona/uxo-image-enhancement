import numpy as np
import cv2


def get_photometric_mask(illumination, smoothing):
    '''
    Computes a photometric mask for image enhancement using a non-linear filtering approach.

    This function generates a photometric mask by applying a series of non-linear operations
    to the input illumination channel. The mask is designed to enhance local contrast while
    preserving edges and minimizing artifacts such as halos and bleeding effects. The process
    involves creating Look-Up Tables (LUTs) based on smoothing parameters and applying them
    in multiple directions to the illumination channel.

    Parameters:
    -----------
    illumination : numpy.ndarray
        A 2D numpy array representing the illumination channel of an image.

    smoothing : float
        A smoothing parameter that controls the strength of the blurring effect. Higher values
        result in stronger blurring, which improves local contrast but may introduce more halo
        artifacts. Lower values preserve edges better but may cause more 'bleeding' effects.

    Returns:
    --------
    numpy.ndarray
        A 2D numpy array representing the photometric mask. This mask can be used to enhance
        the illumination channel of an image while preserving edges and minimizing artifacts.

    References:
    -----------
    Source: https://github.com/bbonik/image_enhancement/blob/master/source/image_enhancement.py
    '''

    # internal parameters
    thr = 255 * np.ones((256, 2))
    thr[:, 0] *= smoothing
    thr[:, 1] *= 0.04

    alpha = 0.12 * 255  # controls non-linearity degree
    beta = np.maximum(255 - thr, 0.001)

    i = np.array((np.arange(256), np.arange(256))).T
    i_comp = i - thr
    mask = i >= thr

    lut = np.zeros((256, 2))
    lut[mask] = (alpha + beta[mask]) * (i_comp[mask] / (i_comp[mask] + alpha)) * (0.5 / beta[mask]) + 0.5
    lut[~mask] = (alpha * i[~mask]) / (alpha - i_comp[~mask]) * (0.5 / thr[~mask])

    # get sigmoid LUTs
    lut_a = lut[:, 0]
    lut_b = lut[:, 1]

    # expand dimensions to 3D for code compatibility (filtering assumes a 3D image)
    image_ph_mask = np.expand_dims(illumination.copy(), axis=2)

    # Start, End, Step, Axis, Diff, LUT
    dirs = (
        (                         1, image_ph_mask.shape[0] - 1,  1, 0, -1, lut_a),  # Up    -> Down
        (                         1, image_ph_mask.shape[1] - 1,  1, 1, -1, lut_a),  # Left  -> Right
        (image_ph_mask.shape[0] - 2,                          1, -1, 0,  1, lut_a),  # Down  -> Up
        (image_ph_mask.shape[1] - 2,                          1, -1, 1,  1, lut_b),  # Right -> Left
        (                         1, image_ph_mask.shape[0] - 1,  1, 0, -1, lut_b),  # Up    -> Down
    )

    for start, end, step, axis, diff, lut in dirs:
        for idx in range(start, end, step):
            if axis == 0:
                d = np.abs(image_ph_mask[idx - 1, :, :] - image_ph_mask[idx + 1, :, :])
                d = lut[(d * len(lut) - 1).astype(int)]
                image_ph_mask[idx, :, :] = image_ph_mask[idx, :, :] * d + image_ph_mask[idx + diff, :, :] * (1 - d)
            else:
                d = np.abs(image_ph_mask[:, idx - 1, :] - image_ph_mask[:, idx + 1, :])
                d = lut[(d * len(lut) - 1).astype(int)]
                image_ph_mask[:, idx, :] = image_ph_mask[:, idx, :] * d + image_ph_mask[:, idx + diff, :] * (1 - d)

    # convert back to 2D and return
    return np.squeeze(image_ph_mask)


def spatial_tonemapping(image, smoothing=0.2, mid_tone=0.5, tonal_width=0.5, areas_dark=0.5, areas_bright=0.5, preserve_tones=True, eps=1 / 256):
    '''
    Applies spatial tonemapping to an image using a center-surround framework for image enhancement.

    This function enhances the tonal range of an image by separating it into dark and bright regions
    and applying non-linear adjustments to each region. The process is based on the center-surround
    framework described in the referenced paper. It preserves the overall tonal structure while
    improving local contrast and detail visibility.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    smoothing : float (default=0.2)
        Controls the strength of the photometric mask smoothing. Higher values result in stronger
        smoothing, which can reduce noise but may also reduce detail.

    mid_tone : float (default=0.5)
        The mid-tone value used to separate dark and bright regions. Values range from 0 to 1.

    tonal_width : float (default=0.5)
        Controls the width of the tonal range adjustment. Higher values result in a wider tonal range.

    areas_dark : float (default=0.5)
        Controls the adjustment strength for dark regions. Values range from 0 to 1.

    areas_bright : float (default=0.5)
        Controls the adjustment strength for bright regions. Values range from 0 to 1.

    preserve_tones : bool (default=True)
        If True, preserves the overall tonal structure of the image while enhancing local contrast.

    eps : float (default=1/256)
        A small constant to avoid division by zero and ensure numerical stability.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array representing the tonemapped image in BGR format.

    References:
    -----------
    Vassilios Vonikakis, Stefan Winkler, "A center-surround framework for spatial image processing"
    in Proc. IS&T Int'l. Symp. on Electronic Imaging: Retinex at 50, 2016, doi: 10.2352/ISSN.2470-1173.2016.6.RETINEX-020
    '''

    # adjust range and non-linear response of parameters
    mid_tone = np.clip(mid_tone, 0, 1)

    tonal_width = np.clip(tonal_width, 0, 1)
    tonal_width = tonal_width * 0.1 / (1.1 - tonal_width) * (1 - eps) + eps

    areas_dark = 1 - np.clip(areas_dark, 0, 1)
    areas_dark = 5 * areas_dark * 0.05 / (1.05 - areas_dark)

    areas_bright = 1 - np.clip(areas_bright, 0, 1)
    areas_bright = 5 * areas_bright * 0.05 / (1.05 - areas_bright)

    # Extract illumination to apply tonemapping
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    illumination = hsv_image[:, :, 2].astype(np.float32) / 255

    image_ph_mask = get_photometric_mask(illumination, smoothing=smoothing)
    image_ph_mask_inv = 1 - image_ph_mask

    image_tonemapped = np.array((illumination, illumination))

    # Image lower
    image_tonemapped[0][illumination >= mid_tone] = 0
    alpha = np.power(image_ph_mask, 2) / tonal_width
    tone_continuation_factor = mid_tone / np.maximum(mid_tone - image_ph_mask, eps)
    alpha = alpha * tone_continuation_factor + areas_dark
    image_tonemapped[0] = image_tonemapped[0] * (alpha + 1) / (alpha + image_tonemapped[0])

    # Image upper
    image_tonemapped[1][illumination < mid_tone] = 0
    alpha = np.power(image_ph_mask_inv, 2) / tonal_width
    tone_continuation_factor = mid_tone / np.maximum(1 - mid_tone - image_ph_mask_inv, eps)
    alpha = alpha * tone_continuation_factor + areas_bright 
    image_tonemapped[1] = (image_tonemapped[1] * alpha) / (1 + alpha - image_tonemapped[1])

    image_tonemapped = image_tonemapped[0] + image_tonemapped[1]
    if preserve_tones is True:
        preservation_degree = np.abs(0.5 - image_ph_mask) / 0.5
        image_tonemapped = preservation_degree * image_tonemapped + illumination * (1 - preservation_degree)

    hsv_image[:, :, 2] = np.clip(255 * image_tonemapped, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def local_contrast_enhancement(image, degree=1.5, smoothing=0.2):
    '''
    Enhances local contrast in an image using a multi-scale approach.

    This function improves the visibility of details in an image by amplifying local contrast
    while preserving the overall structure. It works by separating the illumination and detail
    components of the image, amplifying the details, and then recombining them. The process is
    based on the multi-scale contrast enhancement framework described in the referenced paper.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the input image in BGR format.

    degree : float (default=1.5)
        Controls the strength of the detail amplification. Higher values result in stronger
        contrast enhancement but may introduce artifacts if set too high.

    smoothing : float (default=0.2)
        Controls the strength of the photometric mask smoothing. Higher values result in stronger
        smoothing, which can reduce noise but may also reduce detail.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array representing the contrast-enhanced image in BGR format.

    References:
    -----------
    V. Vonikakis and I. Andreadis, "Multi-scale image contrast enhancement," 
    2008 10th International Conference on Control, Automation, Robotics and Vision, 
    Hanoi, Vietnam, 2008, pp. 856-861, doi: 10.1109/ICARCV.2008.4795629.
    '''

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    illumination = hsv_image[:, :, 2].astype(np.float32) / 255
    image_ph_mask = get_photometric_mask(illumination, smoothing=smoothing)

    image_details = illumination - image_ph_mask

    detail_amplification_local = 255 * image_ph_mask / 100
    detail_amplification_local[detail_amplification_local > 1] = 1
    detail_amplification_local = 1 + 0.2 * (1 - detail_amplification_local)  # [1, 1.2]

    image_details = (degree * image_details * detail_amplification_local)
    hsv_image[:, :, 2] = np.clip(255 * (image_ph_mask + image_details), 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
