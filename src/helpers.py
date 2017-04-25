from __future__ import division
from __future__ import print_function
import numpy as np
import os
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import zoom as sp_zoom
from keras.utils import to_categorical as k_tg
import matplotlib.pyplot as plt


def normalize_and_center(data):
    """Normalize input data.
    #Arguments:
        data: A numpy ndarray.
    #Returns
        The input data divided by (amax(data) - amin(data)) and (mean) centered
        at 0.0."""
    ret = data.astype('float')
    ret -= np.amin(ret)
    ret /= np.amax(ret)
    ret -= np.mean(ret)
    return ret


def to_categorical(data, num_classes=4):
    """Transform numerical to categorical representation.
    #Arguments:
        data: A numpy ndarray.
        num_classes: the number of classes.
    #Returns
        One hot coded input."""
    orig_shape = data.shape
    data = k_tg(data, num_classes=4)
    data.shape = orig_shape[:-1] + (num_classes,)
    return data


def get_rand_z_chunk(sample, nchunks, overlap):
    """Crop a random z-chunk (data slice of certain thickness in z-direction)
        from data.
    #Arguments:
        data: A numpy ndarray.
        nchunks: the number of chunks data is divided into.
        overlap: overlap of chunks.
    #Returns
        One randomly picked z-chunk of data."""
    data_scan = sample[0]
    data_seg = sample[1]
    rand_block = np.random.randint(nchunks)
    z_size = data_scan.shape[2]
    blocksize = int(z_size / nchunks)
    new_block_size = int(blocksize + (blocksize * overlap * 2))
    ret_scan = np.zeros((data_scan.shape[0], data_scan.shape[1], new_block_size, data_scan.shape[3]))
    ret_seg = np.zeros((data_seg.shape[0], data_seg.shape[1], new_block_size, data_seg.shape[3]))
    ret_scan[:, :, int(blocksize * overlap):int(blocksize * overlap + blocksize), :] = data_scan[:, :, int(rand_block * blocksize):int(rand_block * blocksize + blocksize), :]
    ret_seg[:, :, int(blocksize * overlap):int(blocksize * overlap + blocksize), :] = data_seg[:, :, int(rand_block * blocksize):int(rand_block * blocksize + blocksize), :]
    if rand_block < (nchunks - 1):
        ret_scan[:, :, int(blocksize * overlap + blocksize):, :] = data_scan[:, :, int(rand_block * blocksize + blocksize):int(rand_block * blocksize + blocksize + blocksize * overlap), :]
        ret_seg[:, :, int(blocksize * overlap + blocksize):, :] = data_seg[:, :, int(rand_block * blocksize + blocksize):int(rand_block * blocksize + blocksize + blocksize * overlap), :]
    if rand_block > 0:
        ret_scan[:, :, :int(blocksize * overlap) + 1, :] = data_scan[:, :, int(rand_block * blocksize - blocksize * overlap):int(rand_block * blocksize), :]
        ret_seg[:, :, :int(blocksize * overlap) + 1, :] = data_seg[:, :, int(rand_block * blocksize - blocksize * overlap):int(rand_block * blocksize), :]
    return ret_scan, ret_seg


def resize(x, shape, cval=0.):
    """Resize an image to fit a specific shape with keeping the aspect ratio.
    # Arguments
        x: Input tensor.
        shape: The 3 dimensional shape of the new image (without the channels).
    # Returns
        The resized Numpy image tensor with the centered volume.
    """
    x_resized = np.zeros(shape + (x.shape[3],))
    for channel in range(x.shape[3]):
        xi = np.zeros(x.shape[0:3])
        xi[:, :, :] = x[:, :, :, 0]
        fzoom = np.min(np.divide(shape, x.shape[0:3]))
        xi = sp_zoom(xi, fzoom)
        dx = xi.shape[0] - shape[0]
        dy = xi.shape[1] - shape[1]
        dz = xi.shape[2] - shape[2]
        if fzoom >= 1.0:
            xi = np.roll(xi, (-int(dx/2), -int(dy/2), -int(dz/2)),
                            axis=(0, 1, 2))
            xi_new = np.full(shape, cval)
            xi_new[:, :, :] = xi[:shape[0], :shape[1], :shape[2]]
        else:
            xi_new = np.full(shape, cval)
            xi_new[:xi.shape[0], :xi.shape[1], :xi.shape[2]] = xi[:, :, :]
            xi_new = np.roll(xi_new, (-int(dx/2), -int(dy/2), -int(dz/2)),
                                axis=(0, 1, 2))
        x_resized[:, :, :, channel] = xi_new[:, :, :]
    return x_resized

# from:
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py,
# modified for volume data

def rotation(x, x_rot, y_rot, z_rot, fill_mode='constant', cval=0.):
    """Performs a rotation of a Numpy volume tensor.
        Order must be (x, y, z, chn).
    # Arguments
        x: Input tensor.
        x_rot: Rotation on x-axis, in degrees.
        y_rot: Rotation on y-axis, in degrees.
        z_rot: Rotation on z-axis, in degrees.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    # rotate around x-axis
    theta = np.pi / 180 * -x_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[2], x.shape[1]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    for nslc in range(x.shape[2]):
        slc = x[nslc, :, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[nslc, :, :, :] = slc
    # rotate around y-axis
    theta = np.pi / 180 * -y_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[0], x.shape[2]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    for nslc in range(x.shape[1]):
        slc = x[:, nslc, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, nslc, :, :] = slc
    # rotate around z-axis
    theta = np.pi / 180 * -z_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    for nslc in range(x.shape[2]):
        slc = x[:, :, nslc, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, :, nslc, :] = slc
    return x


def shift(x, x_shift, y_shift, z_shift, fill_mode='constant', cval=0.):
    """Performs a spatial shift of a Numpy image tensor.
        Order must be (x, y, z, chn).
    # Arguments
        x: Input tensor. Must be 3D.
        x_shift: shift in x-direction, as a float fraction of the width.
        y_shift: shift in y-direction, as a float fraction of the height.
        z_shift: shift in z-direction, as a float fraction of the depth.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    xs = int(x_shift * x.shape[0])
    ys = int(y_shift * x.shape[1])
    zs = int(z_shift * x.shape[2])
    x = np.roll(x, (xs, ys, zs), axis=(0, 1, 2))
    if fill_mode == 'constant':
        if xs >= 0:
            x[:xs, :, :, :] = cval
        else:
            x[xs:, :, :, :] = cval
        if ys >= 0:
            x[:, :ys, :, :] = cval
        else:
            x[:, ys:, :, :] = cval
        if zs >= 0:
            x[:, :, :zs, :] = cval
        else:
            x[:, :, zs:, :] = cval
    return x


def shear(x, xyx_shear, xyy_shear, xzx_shear, xzz_shear, yzy_shear, yzz_shear,
            fill_mode='constant', cval=0.):
    """Performs a spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        x_shear: Transformation intensity in xy plane along y axis.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    x_axis = 0
    y_axis = 1
    z_axis = 2
    #shear in xy plane along y axis
    shear = xyy_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[2]):
        slc = x[:, :, nslc, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, :, nslc, :] = slc
    #shear in xy plane along x axis
    x = np.swapaxes(x, 0, 1)
    shear = xyx_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[2]):
        slc = x[:, :, nslc, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, :, nslc, :] = slc
    x = np.swapaxes(x, 0, 1)
    #shear in xz plane along z axis
    shear = xzz_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[0], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[1]):
        slc = x[:, nslc, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, nslc, :, :] = slc
    #shear in xz plane along x axis
    x = np.swapaxes(x, 0, 2)
    shear = xzx_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[0], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[1]):
        slc = x[:, nslc, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, nslc, :, :] = slc
    x = np.swapaxes(x, 0, 2)
    #shear in yz plane along z axis
    shear = yzz_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[1], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[0]):
        slc = x[nslc, :, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[nslc, :, :, :] = slc
    #shear in yz plane along y axis
    x = np.swapaxes(x, 1, 2)
    shear = yzy_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[1], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    for nslc in range(x.shape[0]):
        slc = x[nslc, :, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[nslc, :, :, :] = slc
    x = np.swapaxes(x, 1, 2)
    return x


def zoom(x, fzoom, cval=0.):
    """Performs zoom of a Numpy tensor.
    # Arguments
        x: Input tensor.
        fzoom: Zoom factor.
    # Returns
        Zoomed numpy tensor of same shape as input. Zoom is centered at the
        volume center.
    """
    for channel in range(x.shape[3]):
        xi = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
        xi[:, :, :] = x[:, :, :, channel]
        old_shape = xi.shape
        xi = sp_zoom(xi, fzoom, order=0)
        dx = xi.shape[0] - old_shape[0]
        dy = xi.shape[1] - old_shape[1]
        dz = xi.shape[2] - old_shape[2]
        if fzoom >= 1.0:
            xi = np.roll(xi, (-int(dx/2), -int(dy/2), -int(dz/2)), axis=(0, 1, 2))
            xi_new = np.full(old_shape, cval)
            xi_new[:, :, :] = xi[:old_shape[0], :old_shape[1], :old_shape[2]]
        else:
            xi_new = np.full(old_shape, cval)
            xi_new[:xi.shape[0], :xi.shape[1], :xi.shape[2]] = xi[:, :, :]
            xi_new = np.roll(xi_new, (-int(dx/2), -int(dy/2), -int(dz/2)), axis=(0, 1, 2))
        x[:, :, :, channel] = xi_new[:, :, :]
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def show_batch(x_batch, y_batch=None, slc=80):
    for i in range(len(x_batch)):
        img = x_batch[i,:,:,slc,0]
        img = np.swapaxes(x, 0, 1)
        plot = plt.imshow(img)
        plt.gray()
        plt.show()
        if y_batch != None:
            reconvert = np.zeros((y_batch.shape[1], y_batch.shape[2], 1))
            reconvert[:,:,0] = np.argmax(y_batch[i,:,:,slc,:], axis=2)
            reconvert.shape = reconvert.shape[0], reconvert.shape[1]
            reconvert = np.swapaxes(reconvert, 0, 1)
            plot = plt.imshow(reconvert)
            plt.gray()
            plt.show()


def show_sample(x, y=None, slc=None):
    if slc is None:
        slc = int(x.shape[2] / 2)
    img = np.swapaxes(x, 0, 1)
    plot = plt.imshow(img[:, :, slc, 0])
    plt.gray()
    plt.show()
    if y != None:
        reconvert = np.zeros((y.shape[0], y.shape[1], 1))
        reconvert[:,:,0] = np.argmax(y[:, :, slc, :], axis=2)
        reconvert.shape = reconvert.shape[0], reconvert.shape[1]
        reconvert = np.swapaxes(x, 0, 1)
        plot = plt.imshow(reconvert)
        plt.gray()
        plt.show()

