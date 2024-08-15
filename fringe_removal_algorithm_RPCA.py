# fringe_removal_algorithm_RPCA.py
import logging
import os
import numpy as np
from astropy.io import fits

from r_pca import R_pca


class FringeRemovalAlgorithm:
    """
    This class implements various algorithms for fringe removal from images.
    """

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def raw_defringe(self, images, mask=None, nxy=None, robust=False, keep_background=False):
        npix, nobs = images.shape
        if mask is not None:
            if images.shape[0] != mask.shape[0]:
                outStr = "images and mask must have the same shape"
                # print(outStr)
                logging.info(outStr)
                return
        if nxy is not None:
            if npix != nxy[0] * nxy[1]:
                outStr = "nxy values are incompatible with image size."
                # print(outStr)
                logging.info(outStr)
                return
        mfringe = self._raw_fringe(images, mask, nxy)
        fimages = images.copy()
        for i in range(nobs):
            outStr = f"Raw-defringing image number {i}"
            # print(outStr)
            logging.info(outStr)
            if mask.ndim > 1:
                coeffs = self._regress(images[:,i], mfringe, mask=mask[:,i], robust=robust)
            else:
                coeffs = self._regress(images[:,i], mfringe, mask=mask, robust=robust)

            if keep_background:
                fimages[:, i] = images[:, i] - coeffs[1] * mfringe
            else:
                fimages[:, i] = images[:, i] - coeffs[0] - coeffs[1] * mfringe

        return fimages


    def defringe_pcp(self, images, fringe_pattern, mask=None, nxy=None, robust=False,
                     keep_background=False):
        """
        * Function:
          原图 & Robust_PCA 计算得到的fringe pattern 两者相减
          减法进行前，需要先对fringe pattern进行regress，
          原因：这个fp是综合多张图像得到的，需要保证它与ori在coefficient上保持一致

        * Param:
            image: ori（列）
            fringe_pattern: 预估的 fringe pattern
            mask:
            nxy:
            robust:
            keep_background:

        * Return:
            fimages: 已经完成defringe后的图像

        """
        npix = images.shape[0]
        nobs = images.shape[1]
        if mask is not None:
            if images.shape[0] != mask.shape[0]:
                outStr = "images and mask have incompatible sizes.\n"
                # print(outStr)
                logging.info(outStr)
                return
        if nxy is not None:
            if npix != nxy[0] * nxy[1]:
                outStr = "nxy values are incompatible with image size.\n"
                # print(outStr)
                logging.info(outStr)
                return

        # Fringe pattern from r_pca
        # mfringe = fringe_pattern    # the dimension have been change
        mfringe = self._raw_fringe(fringe_pattern, mask=mask, nxy=nxy)
        fimages = images.copy()

        # Compute regression coefficients, and remove background and fringe pattern
        for i in range(nobs):
            outStr = f"Raw-defringing image number {i}"
            # print(outStr)
            logging.info(outStr)

            if mask is not None and mask.ndim > 1:
                coeffs = self._regress(images[:, i], mfringe, mask=mask[:, i], robust=robust)
            else:
                coeffs = self._regress(images[:, i], mfringe, mask=mask, robust=robust)

            if keep_background:
                # joe: only remove fringe pattern
                fimages[:, i] = images[:, i] - coeffs[1] * mfringe
            else:
                # joe: further remove background c[0] term
                fimages[:, i] = images[:, i] - coeffs[0] - coeffs[1] * mfringe
        return fimages

    def _raw_fringe(self, images, mask=None, nxy=None):
        fimages = images.copy()
        for i in range(images.shape[1]):
            fimages[:, i] -= np.median(images[:, i])
            if mask is not None:
                fimages[:, i] *= mask[:, i] if mask.ndim > 1 else mask
        return np.median(fimages, axis=1)


    def _regress(self, image, mfringe, mask=None, robust=False, niter=10):
        flat_image = image.copy()
        flat_mfringe = mfringe.copy()
        flat_back = np.ones(flat_image.size)
        if mask is not None:
            flat_back *= mask
            flat_mfringe *= mask
            flat_image *= mask
        eps = np.median(np.abs(flat_image) / 100.)
        X = np.array([flat_back, flat_mfringe])
        matrix = np.dot(X, X.T)
        coeff = np.dot(np.linalg.inv(matrix), np.dot(X, flat_image))
        if robust:
            for i in range(niter):
                res = flat_image - np.dot(X.T, coeff)
                W = 1. / np.sqrt(1.0 + (res / eps) ** 2)
                if mask is not None:
                    W *= mask
                matrix = np.dot(X, W[:, None] * X.T)
                coeff = np.dot(np.linalg.inv(matrix), np.dot(X, W * flat_image))
        return coeff

    def fringes_pcp(self, images, masks, sigmas=None, tol=None, unshrinking=False,
                    random=None, max_iter=2500, iter_print=100):
        npix, nobs = images.shape
        if sigmas is None:
            sigmas = np.array([self._robust_sigma((images * masks)[:, i]) for i in range(nobs)])
        scaled_images = images / sigmas[None, :]
        outStr = f"sigmas:{sigmas}"
        # print(outStr)
        logging.info(outStr)

        pcp_object = R_pca(scaled_images)
        lowrank_mt, sparse_mt = pcp_object.fit(tol=tol, max_iter=max_iter, iter_print=iter_print)

        if unshrinking:
            outStr = "Unshrinking..."
            # print(outStr)
            logging.info(outStr)

            lowrank_mt = self._unshrink(scaled_images, masks, lowrank_mt, random=random)

        return lowrank_mt * sigmas[None, :], sparse_mt

    def _unshrink(self, images, masks, Z, rank=None, random=(3, 0, 1)):
        npix, nobs = images.shape
        if rank is None:
            rank = 3

        U, D, VT = np.linalg.svd(Z, full_matrices=False)

        Znew = np.zeros(images.shape, dtype=np.float32)

        for i in range(nobs):
            B = U[:, :rank] * masks[:, i][:, None]
            BTB = np.dot(B.T, B)
            RHS = np.dot(B.T, images[:, i])
            coeff = np.dot(np.linalg.inv(BTB), RHS)
            Znew[:, i] = np.dot(U[:, :rank], coeff)

        return Znew

    def save_to_fits(self, ori, output_dir='ori', img_size=500):
        if len(ori.shape) != 2:
            raise ValueError("ori must be a 2D array")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i in range(ori.shape[1]):
            image_data = ori[:, i].reshape((img_size, img_size))
            hdu = fits.PrimaryHDU(image_data)
            hdul = fits.HDUList([hdu])
            fits_filename = os.path.join(output_dir, f"object_{i + 1}.fits")
            hdul.writeto(fits_filename, overwrite=True)
            outStr = f"Saved {fits_filename}"
            # print(outStr)
            logging.info(outStr)

        outStrTemp = "All FITS files have been saved."
        # print(outStrTemp)
        logging.info(outStrTemp)

    def save_all_result(self, savePath, ori, lowRankCoeff, sparseCoeff, pcaDefringeImg, width=500, height=500):
        output_dir = os.path.dirname(savePath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        savePath_ori = os.path.join(savePath, 'Ori')
        self.save_to_fits(ori, output_dir=savePath_ori, img_size=width)

        savePath_lowrank = os.path.join(savePath, 'LowRank')
        self.save_to_fits(lowRankCoeff, output_dir=savePath_lowrank, img_size=width)

        savePath_sparseCoeff = os.path.join(savePath, 'Sparse')
        self.save_to_fits(sparseCoeff, output_dir=savePath_sparseCoeff, img_size=width)

        savePath_pcaDefringe = os.path.join(savePath, 'Defringe')
        self.save_to_fits(pcaDefringeImg, output_dir=savePath_pcaDefringe, img_size=width)
