import os
import numpy
import astropy.io.fits

rawdata_dir = ''
npydata_dir = ''
outdata_dir = ''

# utility class
# =============
class ReductionUtil(object):

    def mkdir(self, path, mode=0770):
        if os.path.exists(path): return None
        return os.makedirs(path, mode)


class Cut(ReductionUtil):

    def __init__(self):
        self.file_name = ''
        pass

    def set_file(self, file_name):
        self.file_name = file_name
        return

    def load_fits(self):
        self._clear_file()
        self._load_fits
