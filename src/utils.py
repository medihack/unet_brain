"""Multiple utility functions to handle volume data.
Some functions are derived from
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
for augmentation of volume data.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import zoom as sp_zoom
from keras.utils import to_categorical as k_tg
import matplotlib.pyplot as plt


def load_nifti(filepath):
    """Load a Nifti file.
    # Arguments
        filepath: The file path of the Nifti image to load.
    # Returns
        A four dimensional array where the last dimension are the color
        channels (rgb). If the Nifti only contains gray values then all
        channels have the same value for one voxel."""
    img = nib.load(filepath)
    img_data = img.get_data()
    # make sure volume orientation is LPS (DICOM default)
    orientation = ''.join(nib.aff2axcodes(img.affine))
    if not orientation == 'LPS':
        if orientation == 'LAS':
            img_data = np.flip(img_data, 1)
        elif orientation == 'RAS':
            img_data = np.flip(img_data, 0)
            img_data = np.flip(img_data, 1)
        else:
            raise ValueError('Unsupported orientation of Nifti file: ' +
                         orientation)
    # make it four dimensional (last dimension are channels)
    if len(img_data.shape) < 4:
        img_data.shape += (1,)
    return img_data


def save_nifti(img_data, filepath):
    """Save voxel data to a Nifti file. Cave, the image doesn't retain its
        world (e.g. MNI) space when saved.
    # Arguments
        img_data: The image (voxel) data of the volume.
        filepath: The file path of the Nifti image to save.
    """
    copy_data = np.copy(img_data)
    # remove channels (there must be only one channel when saving to a
    # nifti file)
    copy_data.shape = copy_data.shape[0:3]
    # convert orientation from LPS to RAS (default of nibabel)
    copy_data = np.flip(copy_data, 0)
    copy_data = np.flip(copy_data, 1)
    affine = np.eye(4)
    img = nib.Nifti1Image(copy_data, affine)
    nib.save(img, filepath)


def normalize_and_center(data):
    """Normalize input data.
    # Arguments
        data: A numpy ndarray.
    # Returns
        The input data divided by (amax(data) - amin(data)) and (mean) centered
        at 0.0."""
    ret = data.astype('float')
    ret -= np.amin(ret)
    ret /= np.amax(ret)
    ret -= np.mean(ret)
    return ret


def to_categorical(data, num_classes=4):
    """Transform numerical to categorical representation.
    # Arguments
        data: A numpy ndarray.
        num_classes: the number of classes.
    # Returns
        One hot coded input."""
    orig_shape = data.shape
    data = k_tg(data, num_classes=4)
    data.shape = orig_shape[:-1] + (num_classes,)
    return data


def to_numerical(data):
    """Transform categorical to numerical representation.
    # Arguments
        data: A numpy ndarray.
    # Returns
        Numerical representation of the input."""
    ret = np.zeros(data.shape[:-1])
    ret[:, :, :, 0] = np.argmax(axis=3)
    return data


# TODO make it work
def get_rand_z_chunk(sample, nchunks, overlap):
    """Crop a random z-chunk (data slice of certain thickness in z-direction)
        from data.
    # Arguments
        data: A numpy ndarray.
        nchunks: the number of chunks data is divided into.
        overlap: overlap of chunks.
    # Returns
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


def strip(x, threshold=0.):
    """Deletes any empty slices on the boundary of a volume.
    # Arguments
        x: Input tensor.
    # Returns:
        The stripped Numpy input tensor.
    """
    # delete empty slices to left and right
    slices_to_delete = []
    for i in range(x.shape[0]):
        if np.max(x[i, :, :, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    for i in range(x.shape[0] - 1, 0, -1):
        if np.max(x[i, :, :, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    x = np.delete(x, slices_to_delete, 0)

    # delete empty slice anterior and posterior
    slices_to_delete = []
    for i in range(x.shape[1]):
        if np.max(x[:, i, :, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    for i in range(x.shape[1] - 1, 0, -1):
        if np.max(x[:, i, :, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    x = np.delete(x, slices_to_delete, 1)

    # delete empty slice superior and inferior
    slices_to_delete = []
    for i in range(x.shape[2]):
        if np.max(x[:, :, i, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    for i in range(x.shape[2] - 1, 0, -1):
        if np.max(x[:, :, i, :]) > threshold:
            break;
        else:
            slices_to_delete.append(i)
    x = np.delete(x, slices_to_delete, 2)

    return x


def fill(x, left=0, right=0, front=0, back=0, top=0, bottom=0, cval=0.):
    """Add (empty) slices to the boundary of a volume.
    # Arguments:
        x: Input sensor.
        left: Number of slices to add to the left.
        right: Number of slices to add to the right.
        front: Number of slices to add to the front (anterior).
        bottom: Number of slices to add to the back (posterior).
        top: Number of slices to add to the top (superior).
        bottom: Number of slices to add to the bottom (inferior).
        cval: The value to use for those slices.
    # Returns:
        The filled up numpy tensor (that has a new shape).
    """
    z = np.copy(x)
    if left > 0:
        t = np.full((left,) + z.shape[1:4], cval)
        z = np.append(z, t, axis=0)
    if right > 0:
        t = np.full((right,) + z.shape[1:4], cval)
        z = np.append(t, z, axis=0)
    if front > 0:
        t = np.full((z.shape[0], front) + z.shape[2:4], cval)
        z = np.append(t, z, axis=1)
    if back > 0:
        t = np.full((z.shape[0], back) + z.shape[2:4], cval)
        z = np.append(z, t, axis=1)
    if top > 0:
        t = np.full((z.shape[0], z.shape[1], top) + (z.shape[3],), cval)
        z = np.append(z, t, axis=2)
    if bottom > 0:
        t = np.full((z.shape[0], z.shape[1], bottom) + (z.shape[3],), cval)
        z = np.append(t, z, axis=2)
    return z


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
    # shear in xy plane along y axis
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
    # shear in xy plane along x axis
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
    # shear in xz plane along z axis
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
    # shear in xz plane along x axis
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
    # shear in yz plane along z axis
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
            xi_new = np.zeros(old_shape)
            xi_new[:, :, :] = xi[:old_shape[0], :old_shape[1], :old_shape[2]]
        else:
            xi_new = np.zeros(old_shape) + cval
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


def show_batch(x_batch, y_batch=None, slc=None, channel=0, axis=2):
    for i in range(len(x_batch)):
        img = x_batch[i, :, :, slc, 0]
        show_sample(img, slc=slc, channel=channel, axis=axis)
        if y_batch != None:
            img = y_batch[i, :, :, :, :]
            show_sample(img, slc=slc)


def show_sample(x, slc=None, channel=0, axis=2):
    """Show a slice of an image volume.
    # Arguments:
        x: Input tensor (image volume with channels).
        slc: The slice number to show (in LPS orientation).
        channel: The channel of the input tensor to show (0 is the default).
        axis: The axis along which the slice is selected.
            0 - along x-axis (sagittal / yz plane)
            1 - along y-axis (coronal / xz plane)
            2 - along z-axis (axial / xy plane), default
    """
    if slc is None:
        slc = int(x.shape[axis] / 2)
    plot=None

    if axis == 0:
        img = np.swapaxes(x, 1, 2)
        img = np.flip(img, 1)
        plot = plt.imshow(img[slc, :, :, channel])
    elif axis == 1:
        img = np.transpose(x, (2, 1, 0, 3))
        img = np.flip(img, 0)
        plot = plt.imshow(img[:, slc, :, channel])
    elif axis == 2:
        img = np.swapaxes(x, 0, 1)
        plot = plt.imshow(img[:, :, slc, channel])
    else:
        raise ValueError('Invalid axis for selecting a slice to show.')
    plt.gray()
    plt.show()


class VolSegDataManager(object):
    """Class to manage volume and respective segmentation Data.
    # Arguments
        vol_folder: Path to volume (and segmentation, see below) data.
        seg_folder: Path to segmentation data. If not specified segmentation
            files are supposed to be in the same folder as volume files.
        vol_prefix: prefix for finding volume files. Files containing volume
            data should begin with a uniform prefix to separate from
            segmentation files.
        seg_prefix: prefix for finding segmentation files.
        test_fract: Fraction of the data used as test/validation data.
            Default = 0.1.
        verbose: Verbosity. 0: no printouts. 1: Print basic information
            (default).
        """
    def __init__(self, vol_folder, seg_folder='', vol_prefix='',
                 seg_prefix='', test_fract=0.1, verbose=1):
        if seg_folder == '':
            seg_folder = vol_folder
        self.vol_folder = vol_folder
        self.seg_folder = seg_folder
        self.data_dict = {}
        self.train_ind_list = []
        self.test_ind_list = []
        self.data_count = 0
        self.data_aug = False
        self.rotrg = 180 # range of random rotations when data_aug==True
        self.shearrg = 0.05 # range of random shear when data_aug==True
        self.zoomrg =  0.2 # range of random zoom when data_aug==True
        self.shiftrg = 0.05 # range of random shift when data_aug==True
        self.verbose = verbose
        # get filepaths
        for entry in os.listdir(self.vol_folder):
            if entry[:len(vol_prefix)] == vol_prefix:
                vol_filename = entry
                seg_filename = vol_filename[len(vol_prefix):]
                seg_filename = seg_prefix + seg_filename
                assert seg_filename in os.listdir(self.seg_folder), 'segmentation file ' + seg_filename + ' for ' + vol_filename + ' doesnt exist.'
                self.data_dict[self.data_count] = (vol_folder + '/' + vol_filename, seg_folder + '/' + seg_filename)
                self.data_count += 1
        assert (test_fract <= 1.) and (test_fract >= 0.), 'fraction of data used for testing must be between 0 and 1.'
        # split up in train data and test data
        self.ntest = int(test_fract * self.data_count)
        self.ntrain = self.data_count - self.ntest
        # --> take ntest random test examples
        for i in range(self.ntest):
            index = int(np.random.random() * (self.data_count - 1) + 0.5)
            self.test_ind_list.append(index)
        # --> use rest as training data
        for i in range(self.data_count):
            if not (i in self.test_ind_list):
                self.train_ind_list.append(i)
        if self.verbose == 1:
            print('Data Manager: got ' + str(self.data_count) + ' volume / segmentation file pairs')
            print('Data Manager: using ' + str(self.ntrain) + ' samples as training data')
            print('Data Manager: using ' + str(self.ntest) + ' samples as test/validation data')

    def _load(self, filepath):
        """Internal load fuction."""
        if filepath[-4:] == '.img' or filepath[-4:] == '.nii' or filepath[-7:] == '.nii.gz':
            data = load_nifti(filepath)
        elif filepath[-4:] == '.npy':
            data = np.load(filepath)
        return data

    def get_sample_i(self, index):
        if self.verbose == 1:
            print('Data Manager: loading sample with index ' + str(index) + '')
        x_path, y_path = self.data_dict[index]
        x = self._load(x_path)
        x = normalize_and_center(x)
        y = self._load(y_path)
        if self.data_aug:
            # zoom
            zm = (np.random.random() * self.zoomrg * 2.) - self.zoomrg
            x = zoom(x, 1. + zm)
            y = zoom(y, 1. + zm)
            # rotation
            xrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            yrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            zrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            x = rotation(x, xrot, yrot, zrot)
            y = rotation(y, xrot, yrot, zrot)
            # shift
            xshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            yshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            zshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            x = shift(x, xshift, yshift, zshift)
            y = shift(y, xshift, yshift, zshift)
            # shear
            xyxshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            xyyshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            xzxshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            xzzshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            yzyshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            yzzshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            x = shear(x, xyxshear, xyyshear, xzxshear, xzzshear, yzyshear, yzzshear)
            y = shear(x, xyxshear, xyyshear, xzxshear, xzzshear, yzyshear, yzzshear)
        y = to_categorical(y)
        return x, y

    def get_rand_sample(self):
        ind = int(np.random.random() * (self.data_count - 1) + 0.5)
        return self.get_sample_i(ind)

    def get_rand_train_sample(self):
        ind = int(np.random.random() * (self.ntrain - 1) + 0.5)
        index = self.train_ind_list[ind]
        return self.get_sample_i(index)

    def get_rand_test_sample(self):
        index = self.test_ind_list[int(np.random.random() * (self.ntest - 1) + 0.5)]
        return self.get_sample_i(index)

    def get_train_generator(self):
        while True:
            yield self.get_rand_train_sample()

    def get_test_generator(self):
        while True:
            yield self.get_rand_test_sample()

    def get_sample_generator(self):
        while True:
            yield self.get_rand_sample()

    def get_rand_batch(self, batch_size=1):
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = self.get_rand_sample()
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

    def get_rand_train_batch(self, batch_size=1):
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = self.get_rand_train_sample()
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

    def get_rand_test_batch(self, batch_size=1):
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = self.get_rand_test_sample()
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

    # not working correctly, needs restructuring
    def get_train_batch_generator_chunks(self, nchunks, overlap, batch_size=1):
        def get_croped_batch():
            batch = self.get_rand_train_batch(batch_size=batch_size)
            new_scan_batch = []
            new_seg_batch = []
            for i in range(batch_size):
                scan, seg = batch[0][i], batch[1][i]
                scan, seg = get_rand_z_chunk((scan, seg), nchunks, overlap)
                new_scan_batch.append(scan)
                new_seg_batch.append(seg)
                new_batch = np.array(new_scan_batch), np.array(new_seg_batch)
            return new_batch
        yield get_croped_batch()

    # not working correctly, needs restructuring
    def get_test_batch_generator_chunks(self, nchunks, overlap, batch_size=1):
        def get_croped_batch():
            batch = self.get_rand_test_batch(batch_size=batch_size)
            new_scan_batch = []
            new_seg_batch = []
            for i in range(batch_size):
                scan, seg = batch[0][i], batch[1][i]
                scan, seg = get_rand_z_chunk((scan, seg), nchunks, overlap)
                new_scan_batch.append(scan)
                new_seg_batch.append(seg)
                new_batch = np.array(new_scan_batch), np.array(new_seg_batch)
            return new_batch
        yield get_croped_batch()

    def get_batch_generator(self, batch_size=1):
        while True:
            yield self.get_rand_batch(batch_size=batch_size)

    def get_train_batch_generator(self, batch_size=1):
        while True:
            yield self.get_rand_train_batch(batch_size=batch_size)

    def get_test_batch_generator(self, batch_size=1):
        while True:
            yield self.get_rand_test_batch(batch_size=batch_size)

