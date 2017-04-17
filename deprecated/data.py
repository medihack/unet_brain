
from __future__ import division
import numpy as np
import os
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import zoom as sp_zoom
from keras.utils import to_categorical as k_tg


def normalize_and_center(data):
    ret = data.astype('float64')
    ret -= np.amin(ret)
    ret /= np.amax(ret)
    ret -= np.mean(ret)
    return ret

def to_categorical(data, num_classes=4):
    orig_shape = data.shape
    data = k_tg(data, num_classes=4)
    data.shape = orig_shape[:-1] + (num_classes,)
    return data

def get_rand_z_chunk(sample, nchunks, overlap):
    data_scan = sample[0]
    data_seg = sample[1]
    rand_block = np.random.randint(nchunks)
    rand_block = 0
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



# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py, modified for volume data

def rotation(x, x_rot, y_rot, z_rot, fill_mode='constant', cval=0.):
    """Performs a rotation of a Numpy volume tensor.
    Order must be (x, y, z, chn)
    # Arguments
        x: Input tensor. Must be 3D.
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
    x_axis = 0
    y_axis = 1
    z_axis = 2
    channel_axis=3

    if channel_axis > z_axis:
        channel_axis -= 1

    # rotate around z-axis
    theta = np.pi / 180 * z_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[x_axis], x.shape[y_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

    for nslc in range(x.shape[z_axis]):
        if z_axis == 0:
            slc = x[nslc, :, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[nslc, :, :, :] = slc
        elif z_axis == 1:
            slc = x[:, nslc, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, nslc, :, :] = slc
        elif z_axis == 2:
            slc = x[:, :, nslc, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, nslc, :] = slc
        elif z_axis == 3:
            slc = x[:, :, :, nslc]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, :, nslc] = slc

    # rotate around x-axis
    theta = np.pi / 180 * x_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    np.swapaxes(x, z_axis, x_axis)
    tmp = x_axis
    x_axis = z_axis
    z_axis = tmp
    h, w = x.shape[x_axis], x.shape[y_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

    for nslc in range(x.shape[z_axis]):
        if z_axis == 0:
            slc = x[nslc, :, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[nslc, :, :, :] = slc
        elif z_axis == 1:
            slc = x[:, nslc, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, nslc, :, :] = slc
        elif z_axis == 2:
            slc = x[:, :, nslc, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, nslc, :] = slc
        elif z_axis == 3:
            slc = x[:, :, :, nslc]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, :, nslc] = slc

    # rotate around y-axis
    theta = np.pi / 180 * y_rot
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    np.swapaxes(x, z_axis, y_axis)
    tmp = y_axis
    y_axis = z_axis
    z_axis = tmp
    h, w = x.shape[x_axis], x.shape[y_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

    for nslc in range(x.shape[z_axis]):
        if z_axis == 0:
            slc = x[nslc, :, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[nslc, :, :, :] = slc
        elif z_axis == 1:
            slc = x[:, nslc, :, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, nslc, :, :] = slc
        elif z_axis == 2:
            slc = x[:, :, nslc, :]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, nslc, :] = slc
        elif z_axis == 3:
            slc = x[:, :, :, nslc]
            slc = apply_transform(slc, transform_matrix, channel_axis, fill_mode, cval)
            x[:, :, :, nslc] = slc

    np.swapaxes(x, z_axis, x_axis)
    np.swapaxes(x, x_axis, y_axis)

    return x


def shift(x, x_shift, y_shift, z_shift, fill_mode='constant', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    Order must be (x, y, z, chn)
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


def shear(x, xy_shear, xz_shear, yz_shear, fill_mode='constant', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
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
    shear = xy_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)

    for nslc in range(x.shape[2]):
        slc = x[:, :, nslc, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, :, nslc, :] = slc

    #shear in xz plane along z axis
    shear = xz_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[0], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)

    for nslc in range(x.shape[1]):
        slc = x[:, nslc, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[:, nslc, :, :] = slc

    #shear in yz plane along z axis
    shear = yz_shear
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[1], x.shape[2]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)

    for nslc in range(x.shape[0]):
        slc = x[nslc, :, :, :]
        slc = apply_transform(slc, transform_matrix, 2, fill_mode, cval)
        x[nslc, :, :, :] = slc

    return x


def zoom(x, fzoom, cval=0.):

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


class VolDataManager(object):
    def __init__(self, vol_folder, seg_folder='', vol_prefix='vol_',
                 seg_prefix='seg_', index_length=3, test_fract=0.1):

        if seg_folder == '':
            seg_folder = vol_folder

        self.vol_folder = vol_folder
        self.seg_folder = seg_folder
        self.data_dict = {}
        self.train_ind_list = []
        self.test_ind_list = []
        self.data_count = 0
        self.data_aug = False
        self.rotrg = 180 #range of random rotations when data_aug==True
        self.shearrg = 0.05 #range of random shear when data_aug==True
        self.zoomrg =  0.2 #range of random zoom when data_aug==True
        self.shiftrg = 0.05 #range of random shift when data_aug==True

        for entry in os.listdir(self.vol_folder):
            if entry[:len(vol_prefix)] == vol_prefix:
                vol_filename = entry
                seg_filename = vol_filename[len(vol_prefix):]
                seg_filename = seg_prefix + seg_filename
                assert seg_filename in os.listdir(self.seg_folder), 'segmentation file ' + seg_filename + ' for ' + vol_filename + ' doesnt exist.'
                self.data_dict[self.data_count] = (vol_folder + '/' + vol_filename, seg_folder + '/' + seg_filename)
                self.data_count += 1

        assert (test_fract <= 1.) and (test_fract >= 0.), 'fraction of data used for testing mus be between 0 and 1.'

        self.ntest = int(test_fract * self.data_count)
        self.ntrain = self.data_count - self.ntest

        for i in range(self.ntest):
            index = int(np.random.random() * (self.data_count - 1) + 0.5)
            self.test_ind_list.append(index)

        for i in range(self.data_count):
            if not (i in self.test_ind_list):
                self.train_ind_list.append(i)

    def get_sample_i(self, index):
        print('getting sample from file with index ' + str(index))
        x_path, y_path = self.data_dict[index]
        x = np.load(x_path)
        x = normalize_and_center(x)
        y = np.load(y_path)
        if self.data_aug:
            xrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            yrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            zrot = (np.random.random() * self.rotrg * 2.) - self.rotrg
            xshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            yshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            zshear = (np.random.random() * self.shearrg * 2.) - self.shearrg
            xshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            yshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            zshift = (np.random.random() * self.shiftrg * 2.) - self.shiftrg
            zm = (np.random.random() * self.zoomrg * 2.) - self.zoomrg
            x = zoom(x, 1. + zm)
            y = zoom(y, 1. + zm)
            x = rotation(x, xrot, yrot, zrot)
            y = rotation(y, xrot, yrot, zrot)
            x = shift(x, xshift, yshift, zshift)
            y = shift(y, xshift, yshift, zshift)
            x = shear(x, xshear, yshear, zshear)
            y = shear(y, xshear, yshear, zshear)
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


def show_batch(x_batch, y_batch=None, slc=80):
    import matplotlib.pyplot as plt
    for i in range(len(x_batch)):
        plot = plt.imshow(x_batch[i,:,:,slc,0])
        plt.gray()
        plt.show()
        if y_batch != None:
            reconvert = np.zeros((y_batch.shape[1], y_batch.shape[2], 1))
            reconvert[:,:,0] = np.argmax(y_batch[i,:,:,slc,:], axis=2)
            reconvert.shape = reconvert.shape[0], reconvert.shape[1]
            plot = plt.imshow(reconvert)
            plt.gray()
            plt.show()

def show_sample(x, y=None, slc=80):
    import matplotlib.pyplot as plt
    plot = plt.imshow(x[:,:,slc,0])
    plt.gray()
    plt.show()
    if y != None:
        reconvert = np.zeros((y.shape[0], y.shape[1], 1))
        reconvert[:,:,0] = np.argmax(y[:,:,slc,:], axis=2)
        reconvert.shape = reconvert.shape[0], reconvert.shape[1]
        plot = plt.imshow(reconvert)
        plt.gray()
        plt.show()

data = VolDataManager('/Users/hagen/Code/Python/nn/unet_brain/data/scan',
                     '/Users/hagen/Code/Python/nn/unet_brain/data/seg',
                     vol_prefix='scan_', seg_prefix='seg_',
                     test_fract=0.34)

data.data_aug = True

gen = data.get_train_batch_generator_chunks(4, 0.1, batch_size=10)
batch = gen.next()
print batch[0].shape
print batch[1].shape
show_batch(*batch, slc=25)

# todo: implement resizing function, make chunking nicer.... for example implement it as an option like in data_aug
