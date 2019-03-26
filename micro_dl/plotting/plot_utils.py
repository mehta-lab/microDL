"""Utility functions for plotting"""
import cv2
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import natsort
import numpy as np
import os
from micro_dl.utils.normalize import hist_clipping


def save_predicted_images(input_batch, target_batch, pred_batch,
                          output_dir, batch_idx=None, output_fname=None,
                          tol=1, font_size=15):
    """Saves a batch predicted image to output dir

    Format: rows of [input, target, pred]

    :param np.ndarray input_batch: expected shape [batch_size, n_channels,
     x,y,z]
    :param np.ndarray target_batch: target with the same shape of input_batch
    :param np.ndarray pred_batch: output predicted by the model
    :param str output_dir: dir to store the output images/mosaics
    :param int batch_idx: current batch number/index
    :param str output_fname: fname for saving collage
    :param float tol: top and bottom % of intensity to saturate
    :param int font_size: font size of the image title
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    batch_size = len(input_batch)
    if batch_size == 1:
        assert output_fname is not None, 'need fname for saving image'
        fname = os.path.join(output_dir, '{}.jpg'.format(output_fname))

    # 3D images are better saved as movies/gif
    if batch_size != 1:
        assert len(input_batch.shape) == 4, 'saves 2D images only'

    for img_idx in range(batch_size):
        cur_input = input_batch[img_idx]
        cur_target = target_batch[img_idx]
        cur_prediction = pred_batch[img_idx]
        n_channels = cur_input.shape[0]
        fig, ax = plt.subplots(n_channels, 3)
        fig.set_size_inches((15, 5 * n_channels))
        axis_count = 0
        for channel_idx in range(n_channels):
            ax[axis_count].imshow(hist_clipping(cur_input[channel_idx],
                                                tol, 100 - tol), cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 0:
                ax[axis_count].set_title('Input', fontsize=font_size)
            axis_count += 1
            ax[axis_count].imshow(hist_clipping(cur_target[channel_idx],
                                                tol, 100 - tol), cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 1:
                ax[axis_count].set_title('Target', fontsize=font_size)
            axis_count += 1
            ax[axis_count].imshow(hist_clipping(cur_prediction[channel_idx],
                                                tol, 100 - tol), cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 2:
                ax[axis_count].set_title('Prediction', fontsize=font_size)
            axis_count += 1
        if batch_size != 1:
            fname = os.path.join(
                output_dir,
                '{}.jpg'.format(str(batch_idx * batch_size + img_idx))
            )
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_xyz(image_dir,
             pos_idx,
             fig_name,
             tol=1,
             font_size=15,
             margin=10,
             scale=10):
    """
    Takes a 3D volume, plots the center slice and the yz and xz center
     cross sections.

    :param str image_dir: Directory containing z-stacks
    :param int pos_idx: Which FOV to plot
    :param str fig_name: Full path to figure file name
    :param float tol: top and bottom % of intensity to saturate
    :param int font_size: font size of the image title
    """
    search_str = os.path.join(image_dir, "*p{:03d}*".format(pos_idx))
    slice_names = natsort.natsorted(glob.glob(search_str))

    im_stack = []
    for im_z in slice_names:
        im_stack.append(cv2.imread(im_z, cv2.IMREAD_ANYDEPTH))
    im_stack = np.stack(im_stack, axis=-1)

    im_norm = im_stack / im_stack.std() * IM_STD
    im_norm = im_norm - im_norm.mean() + IM_MEAN
    # cutoff at 0
    im_norm[im_norm < 0] = 0.
    im_norm = im_norm.astype(np.uint16)

    fig, ax = plt.subplots(111)

    center_slice = hist_clipping(
        im_norm[..., int(len(slice_names) // 2)],
        tol, 100 - tol,
    )
    im_shape = im_stack.shape
    canvas = IM_MEAN ** np.ones((im_shape[0] + im_shape[2] * scale + margin,
                      im_shape[1] + im_shape[2] * scale + margin))
    # Add center slice
    canvas[0:im_shape[0], 0:im_shape[1]] = center_slice
    # add yz
    center_slice = hist_clipping(
        np.squeeze(im_norm[:, int(im_shape[1] // 2), :]),
        tol, 100 - tol,
    )
    cshape = center_slice.shape
    resized_slice = cv2.resize(center_slice, (cshape[1] * scale, cshape[0]))

    canvas[0:im_shape[0], im_shape[1] + margin:] = resized_slice

    # add xy
    center_slice = hist_clipping(
        np.squeeze(im_norm[int(im_shape[1] // 2), :, :]),
        tol, 100 - tol,
    )
    cshape = center_slice.shape
    resized_slice = cv2.resize(center_slice, (cshape[1] * scale, cshape[0]))

    canvas[0:im_shape[0], im_shape[1] + margin:] = resized_slice

    plt.imshow(center_slice, cmap='gray')
    plt.axis('off')
    ax.set_title('Input', fontsize=font_size)

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_mask_overlay(input_image, mask, op_fname, alpha=0.7):
    """
    Plot and save a collage of input, mask, overlay

    :param np.array input_image: 2D input image
    :param np.array mask: 2D mask image
    :param str op_fname: fname will full path for saving the collage as a jpg
    :param int alpha: opacity/transparency for the mask overlay
    """

    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1'
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((15, 5))
    ax[0].imshow(input_image, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis('off')
    # Convert image to uint8 color, scale to 255, and overlay a color contour
    im_rgb = input_image / input_image.max() * 255
    im_rgb = im_rgb.astype(np.uint8)
    im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    _, contours, _ = cv2.findContours(mask.astype(np.uint8),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours in green with linewidth 2
    im_rgb = cv2.drawContours(im_rgb, contours, -1, (0, 255, 0), 2)
    ax[2].imshow(im_rgb)
    ax[2].axis('off')
    fig.savefig(op_fname, dpi=250)
    plt.close(fig)


def save_plot(x, y, fig_fname, fig_labels=None):
    """
    Plot values y = f(x) and save figure.

    :param list x: x values
    :param list y: y values (same length as x)
    :param str fig_fname: File name including full path
    :param list fig_labels: Labels for x and y axes, and title
    """
    assert len(x) == len(y),\
        "x ({}) and y ({}) must be equal length".format(len(x), len(y))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    if fig_labels is not None:
        assert len(fig_labels) >= 2, "At least x and y labels must be present"
        ax.set_xlabel(fig_labels[0])
        ax.set_ylabel(fig_labels[1])
        if len(fig_labels) == 3:
            ax.set_title(fig_labels[2])
    fig.savefig(fig_fname, dpi=250)
    plt.close(fig)
