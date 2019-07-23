import os
import numpy
import scipy.signal
import scipy.ndimage
import skimage
import astropy.io.fits
import matplotlib.pyplot

rawdata_dir = '/data/science/spitzer/I4/raw/obj/fits/'
npydata_dir = '/data/science/spitzer/I4/raw/obj/npy/'
outdata_dir = '/data/science/spitzer/I4/reduction/'

# utility class
# =============
class ReductionUtil(object):

   def mkdir(self, path, mode=0o755):
       if os.path.exists(path): return None
       return os.makedirs(path, mode)


class ImageProcessing(ReductionUtil):

    def __init__(self):
        self.mkdir(rawdata_dir)
        self.mkdir(npydata_dir)
        self.mkdir(outdata_dir)
        self.header = None
        self.original_image_size = 0
        self.original_image = numpy.array([])
        self.input_image = numpy.array([])
        self.output_image = numpy.array([])
        self.output_file = {'name': '', 'extention': ''}
        self.obj_name = ''
        pass

    def set_obj(self, obj_name):
        self._clear_obj()
        self.obj_name = obj_name
        self.rawdata_fits = os.path.join(rawdata_dir + obj_name + '.fits')
        self.rawdata_npy = os.path.join(npydata_dir + obj_name + '.npy')
        return

    def load_rawfits(self, save_format='npy'):
        self._load_fits(self.rawdata_fits)
        self.output_file['name'] = self.obj_name
        self.output_file['extention'] = '.' + save_format
        if self.original_image.shape[0] > self.original_image.shape[1]:
            self.original_image_size = self.original_image.shape[0]
        else: self.original_image_size = self.original_image.shape[1]
        return

    def load_npy(self, save_format='npy'):
        self._load_npy(self.rawdata_npy)
        self.output_file['name'] = self.obj_name
        self.output_file['extention'] = '.' + save_format
        if self.original_image.shape[0] > self.original_image.shape[1]:
            self.original_image_size = self.original_image.shape[0]
        else: self.original_image_size = self.original_image.shape[1]
        return

    def rotate(self, degree):
        self.input_image = scipy.ndimage.interpolation.rotate(
            input=self.input_image,
            angle=degree
        )
        self.output_file['name'] += '_rotate-' + str(degree)
        self.output_image = self.input_image
        return

    def refrect(self, direction):
        if direction == 'v':
            self.input_image = numpy.flipud(self.input_image)
        if direction == 'h':
            self.input_image = numpy.fliplr(self.input_image)
        self.output_file['name'] += '_refrect-' + direction
        self.output_image = self.input_image
        return

    def transform(self, offset):
        offset = skimage.transform.AffineTransform(translation=offset)
        self.input_image = skimage.transform.warp(im.output_image, matrix_trans)


    def remove_star(self, kernel_size):
        self.input_image = scipy.signal.medfilt2d(
            input=self.input_image,
            kernel_size=(kernel_size, kernel_size)
        )
        self.output_file['name'] += '_med-' + str(kernel_size)
        self.output_image = self.input_image
        return

    def resize(self, im_size):
        self.input_image = skimage.transform.resize(
            image=self.input_image,
            output_shape=(im_size, im_size)
        )
        self.output_file['name'] += '_resize-' + str(im_size)
        self.output_image = self.input_image
        return

    def normalize(self):
        min = self.input_image.min(axis=None, keepdims=True)
        max = self.input_image.max(axis=None, keepdims=True)
        self.input_image = (self.input_image-min)/(max-min)
        self.output_file['name'] += '_normalize'
        self.output_image = self.input_image
        return

    def znormalize(self):
        xmean = self.input_image.mean(axis=None, keepdims=True)
        xstd  = numpy.std(self.input_image, axis=None, keepdims=True)
        self.input_image = (self.input_image-xmean)/xstd
        self.output_file['name'] += '_znormalize'
        self.output_image = self.input_image
        return

    def nan2zero(self):
        self.input_image[numpy.isnan(self.input_image)] = 0
        self.output_image = self.input_image
        return

    def cut(self, factor):
        d = (self.original_image_size*factor)/2
        max_pix = int(self.input_image.shape[0]/2 + d)
        min_pix = int(self.input_image.shape[0]/2 - d)
        self.input_image = self.input_image[min_pix:max_pix, min_pix:max_pix]
        self.output_image = self.input_image
        return

    def plot(self, vmin, vmax):
        fig = matplotlib.pyplot.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        matplotlib.pyplot.imshow(
            X=self.output_image,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
            origin='lower'
        )
        matplotlib.pyplot.show()
        return


    def _set_member(self, image):
        self.input_image = image
        self.output_image = self.input_image
        return

    def _clear_obj(self):
        self.header = None
        self.original_image = numpy.array([])
        self.input_image = numpy.array([])
        self.output_image = numpy.array([])
        self.output_file = {'name': '', 'extention': ''}
        return

    def _load_fits(self, filepath):
        hdu = astropy.io.fits.open(path)
        self.original_image = hdu[0].data
        self.header = hdu[0].header
        self._set_member(self.original_image)
        return

    def _load_npy(self, filepath):
        self.original_image = numpy.load(self.rawdata_npy)
        self._set_member(self.original_image)
        return
