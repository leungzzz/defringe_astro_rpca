import os
from astropy.io import fits as pf
import numpy as np

class ImageProcessor:
    """
    This class handles image reading, preprocessing, and mask creation.
    """

    def __init__(self, config):
        self.config = config
        self.names = self._create_image_names(config.rootDir, config.oriDir)
        self.images = self._read_images(self.names)
        self.ori_images = self.images.copy()
        self.elxnames = self._create_image_names(config.rootDir, config.elixirDataDir)
        self.elximages = self._read_images(self.elxnames)
        self.image_medians = np.median(self.images, axis=0)
        self.elximage_medians = np.median(self.elximages, axis=0)
        self.images -= self.image_medians[None, :]
        self.elximages -= self.elximage_medians[None, :]
        self.chipmask = self._read_chipmask()

    def _create_image_names(self, rootDir, subfolderDir):
        images_dir = []
        image_dir = os.path.join(rootDir, subfolderDir)
        image_files = os.listdir(image_dir)
        for file_name in image_files:
            full_dir = os.path.join(rootDir, subfolderDir, file_name)
            images_dir.append(full_dir)
        return images_dir

    def _read_images(self, image_names):
        images = []
        for name in image_names:
            image = pf.getdata(name).T  # Assuming split mode, image in primary hdu
            if self.config.cut_overscan:
                image = image[self.config.OVERSCAN_X_MIN:self.config.OVERSCAN_X_MAX, self.config.OVERSCAN_Y_MIN:self.config.OVERSCAN_Y_MAX]
            if self.config.small:
                image = image[:self.config.WINSMALL, :self.config.WINSMALL]
            images.append(image.flatten())
        return np.transpose(np.array(images))

    def _read_chipmask(self):
        with pf.open(self.config.chipmask_file) as f:
            chipmask = f[self.config.CCDNUM + 1].data.astype(float).T
            if self.config.cut_overscan:
                chipmask = chipmask[self.config.OVERSCAN_X_MIN:self.config.OVERSCAN_X_MAX, self.config.OVERSCAN_Y_MIN:self.config.OVERSCAN_Y_MAX]
            if self.config.small:
                chipmask = chipmask[:self.config.WINSMALL, :self.config.WINSMALL]
            return chipmask.flatten()

    def create_masks(self, images, chipmask, kappa=2.0, tol=1e-10, include_inputmask=True, robust=False):
        ndim_mask = chipmask.ndim
        imagemask_mode = False
        if ndim_mask > 1:
            imagemask_mode = True

        npix = images.shape[0]
        nobs = images.shape[1]
        masks = np.ndarray((npix, nobs), dtype=np.float64)  # create new null array
        for i in range(nobs):
            if imagemask_mode:
                masks[:, i] = self._mask_sources(images[:, i], chipmask[:, i], kappa, tol, include_inputmask, robust)
            else:
                masks[:, i] = self._mask_sources(images[:, i], chipmask, kappa, tol, include_inputmask, robust)
        return masks

    def _mask_sources(self, image, mask, kappa=2.0, tol=1e-10, include_inputmask=True, robust=True):
        msk = np.ones(mask.shape, dtype=np.float64)
        img = image * mask
        if not robust:
            rms = img.std()
            frac_err = 1e30
            while frac_err > tol:
                rms_old = rms
                clip = np.where((img - np.median(img)) / rms_old > kappa)
                msk[clip] = 0.0
                img *= msk
                rms = img.std()
                frac_err = np.abs(rms - rms_old) / rms_old
        else:
            rms = self._robust_sigma(img)
            clip = np.where((img - np.median(img)) / rms > kappa)
            msk[clip] = 0.0
            img *= msk
        return msk * mask if include_inputmask else msk

    def _robust_sigma(self, X):
        return 1.4826 * np.median(np.abs(X - np.median(X)))
