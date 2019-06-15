import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, ball, binary_opening, binary_erosion


def create_otsu_mask(input_image, str_elem_size=3):
    """Create a binary mask using morphological operations

    Opening removes small objects in the foreground.

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = threshold_otsu(input_image, nbins=512)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    thr_image = binary_opening(input_image > thr, str_elem)
    mask = binary_fill_holes(thr_image)
    return mask


def get_unimodal_threshold(input_image):
    """Determines optimal unimodal threshold

    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding

    :param np.array input_image: generate mask for this image
    :return float best_threshold: optimal lower threshold for the foreground
     hist
    """

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, 'Error in unimodal thresholding'
    return best_threshold


def create_unimodal_mask(input_image, str_elem_size=3):
    """Create a mask with unimodal thresholding and morphological operations

    unimodal thresholding seems to oversegment, erode it by a fraction

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    thr_image = binary_opening(input_image > thr, str_elem)
    mask = binary_erosion(thr_image, str_elem)
    return mask


def get_unet_border_weight_map(binary_mask, w0=10, sigma=5):
    """
    Generate the weight map for borders as specified in the UNet paper for a binary mask.
    Parameters
    ----------
    mask: array-like
        A 2D array of shape (image_height, image_width) one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    """

    assert binary_mask.dtype == np.uint8
    # class balance weights w_c(x)
    unique_values = np.unique(binary_mask).tolist()
    weight_map = [0] * len(unique_values)
    for index, unique_value in enumerate(unique_values):
        mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float64)
        mask[mask == unique_value] = 1
        weight_map[index] = 1 / mask.sum()

    # this normalization is important - foreground pixels must have weight 1
    weight_map = weight_map / max(weight_map)

    wc = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float64)
    for index, unique_value in enumerate(unique_values):
        wc[mask == unique_value] = weight_map[index]

    # cells instances for distance computation
    labeled_array, _ = scipy.ndimage.measurements.label(binary_mask)

    # cells distance map
    border_loss_map = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float64)
    distance_maps = np.zeros((binary_mask.shape[0], binary_mask.shape[1], np.max(labeled_array)), dtype=np.float64)
    if np.max(labeled_array) >= 2:
        for index, label in enumerate(range(1, np.max(labeled_array) + 1)):
            mask = np.ones_like(labeled_array)
            mask[labeled_array == label] = 0
            distance_maps[:, :, index] = scipy.ndimage.distance_transform_edt(mask)

    distance_maps = np.sort(distance_maps, 2)
    d1 = distance_maps[:, :, 0]
    d2 = distance_maps[:, :, 1]
    border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

    zero_label = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float64)
    zero_label[labeled_array == 0] = 1
    border_loss_map = np.multiply(border_loss_map, zero_label)

    # unet weight map mask
    weight_map_mask = wc + border_loss_map

    return weight_map_mask
