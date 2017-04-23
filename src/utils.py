from __future__ import division
from __future__ import print_function
import numpy as np
import os
import nibabel as nib
from helpers import normalize_and_center
from helpers import to_categorical
from helpers import get_rand_z_chunk
from helpers import rotation
from helpers import shift
from helpers import shear
from helpers import zoom


def load_nifti(filepath):
    """Load a Nifti file.
    #Arguments:
        filepath: The file path of the Nifti image to load.
    #Returns
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
    img_data.shape += (1,)

    return img_data


class VolSegDataManager(object):
    """Class to manage volume and respective segmentation Data.
    #Arguments:
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
        assert (test_fract <= 1.) and (test_fract >= 0.), 'fraction of data used for testing mus be between 0 and 1.'
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
            print('Data Manager: got ' + str(self.data_count) + ' files')
            print('Data Manager: using ' + str(self.ntrain) + ' files as training data')
            print('Data Manager: using ' + str(self.ntest) + ' files as test/validation data')

    def _load(self, filepath):
        """Internal load fuction."""
        if filepath[-4:] == '.img' or filepath[-4:] == '.nii' or filepath[-7:] == '.nii.gz':
            data = load_nifti(filepath)
        elif filepath[-4:] == '.npy':
            data = np.load(filepath)
        return data

    def get_sample_i(self, index):
        if self.verbose == 1:
            print('Data Manager: getting sample from file with index ' + str(index) + '')
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

# todo: implement chunking as an option like data_aug
