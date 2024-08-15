import numpy as nm
# try:
#     import cupy as cp
#     GPU = True
# except:
#     print("cupy not installed. This probably means there is no GPU on this host.")
#     print("Using numpy instead")
#     import numpy as cp
#     GPU = False
import numpy as cp
import os
from astropy.io import fits
GPU = False

from images import data
import randlin
from r_pca import R_pca  # import new
from config import Config

class fringe(data):
    def __init__(self, config):
        # Initialize parent data class
        super().__init__(config=config)
        return

    def raw_defringe(self, images, mask=None, nxy=None, robust=False, keep_background=False):
        """
          In this phase, we remove background (source) and a estimated fringe
          pattern from the original image.
        """

        npix = images.shape[0]
        nobs = images.shape[1]
        if mask is not None:
            if images.shape[0] != mask.shape[0]:
                print("images and mask have incompatible sizes.\n")
                return
        if nxy is not None:
            if npix != nxy[0] * nxy[1]:
                print("nxy values are incompatible with image size.\n")
                return

        # Compute median fringe pattern
        mfringe = self.raw_fringe(images, mask=mask, nxy=nxy)
        fimages = images.copy()

        # Compute regression coefficients, and remove background and fringe pattern
        for i in range(nobs):
            print('Raw-defringing image number %d' % i)
            if mask.ndim > 1:
                coeffs = self.regress(images[:, i], mfringe, mask=mask[:, i], robust=robust)
            else:
                coeffs = self.regress(images[:, i], mfringe, mask=mask, robust=robust)
            if keep_background:
                # joe:only remove fringe pattern
                fimages[:, i] = images[:, i] - coeffs[1] * mfringe
            else:
                # joe:further remove background c[0] term
                fimages[:, i] = images[:, i] - coeffs[0] - coeffs[1] * mfringe

        return fimages

    ## mine_defringe
    def mine_defringe(self, images, fringe_pattern, mask=None,
                      nxy=None, robust=False, keep_background=False):
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
                print("images and mask have incompatible sizes.\n")
                return
        if nxy is not None:
            if npix != nxy[0] * nxy[1]:
                print("nxy values are incompatible with image size.\n")
                return

        # Fringe pattern from r_pca
        # mfringe = fringe_pattern    # the dimension have been change
        mfringe = self.raw_fringe(fringe_pattern, mask=mask, nxy=nxy)
        fimages = images.copy()

        # Compute regression coefficients, and remove background and fringe pattern
        for i in range(nobs):
            print('Raw-defringing image number %d' % i)

            if mask is not None and mask.ndim > 1:
                coeffs = self.regress(images[:, i], mfringe, mask=mask[:, i], robust=robust)
            else:
                coeffs = self.regress(images[:, i], mfringe, mask=mask, robust=robust)

            if keep_background:
                fimages[:, i] = images[:, i] - coeffs[1] * mfringe  # joe: only remove fringe pattern
            else:
                fimages[:, i] = images[:, i] - coeffs[0] - coeffs[
                    1] * mfringe  # joe: further remove background c[0] term
        return fimages

    def raw_fringe(self, images, mask=None, nxy=None):
        # Creates a rough master fringe pattern from a set
        # of images from the same chip.

        # Flatten images (remove large scale modes)
        fimages = images.copy()
        nobs = images.shape[1]
        if nxy is None:
            # Assume square images
            nx = nm.int32(nm.sqrt(images.shape[0]))
            nxy = (nx, nx)
        for i in range(nobs):
            #fimages[:,i] = images[:,i]-large_scales(images[:,i],nxy)
            fimages[:, i] = images[:, i] - cp.median(images[:, i])  # joe: remove img itself median
            if (mask is not None):
                if (mask.ndim > 1):
                    fimages[:, i] *= mask[:, i]
                else:
                    fimages[:, i] *= mask

        # Take the median
        return cp.median(fimages, axis=1)  # joe: return median of same pixel in all imgs

    def regress(self, image, mfringe, mask=None, robust=False, niter=10):

        """
          Fit a master fringe and a (constant) background
          amplitude from the image.
          If mask is provided, then pixels where mask=0
          are not taken into account
        """

        flat_image = image.copy()
        flat_mfringe = mfringe.copy()
        if mask is not None:
            flat_mask = mask.copy()
            if flat_mask.size != flat_image.size:
                print("mask and image are of different sizes\n")
                return
        # Check that all images have the same size
        if flat_image.size != flat_mfringe.size:
            print("image and fringe pattern are of different sizes\n")
            return

        # Define constant mode for background
        flat_back = cp.ones(flat_image.size)

        # put all masked pixels to zero everywhere
        if mask is not None:
            flat_back *= flat_mask
            flat_mfringe *= flat_mask
            flat_image *= flat_mask

        eps = cp.median(nm.abs(flat_image) / 100.)
        # Compute the linear regression
        X = cp.asarray((flat_back, flat_mfringe))  # joe: 3-d array(2xmxn).X[0]=f_b, X[1]=f_m
        matrix = cp.dot(X, X.T)
        coeff = cp.dot(cp.linalg.inv(matrix), cp.dot(X, flat_image))
        print('coeff of OLS = ', coeff)
        if robust:
            #residual2 = (flat_image - nm.dot(coeff,X))
            #npix = residual2.size
            #indices = nm.argsort(residual2)
            #ou = nm.where(indices < npix/2.)
            #indices = indices[ou]
            #flat_image = flat_image[indices]
            #X = X[:,indices]
            #coeff = nm.dot(nm.linalg.inv(nm.dot(X,X.T)),nm.dot(X,flat_image))

            # Compute residual
            for i in range(niter):
                res = flat_image - cp.dot(X.T, coeff)
                W = 1. / (1.0 + (res / eps) ** 2) ** 0.5
                if (mask is not None):
                    W *= flat_mask
                # Compute WLS
                matrix = cp.dot(X, W[:, None] * X.T)
                coeff = cp.dot(cp.linalg.inv(matrix), cp.dot(X, W * flat_image))
                print('coeff of GLS = ', coeff)

        return coeff

    def fringes(self, images, masks, tol=1e-10, rank=None, unshrinking=True,
                sigmas=None, lfac=1.0, random=None):
        """
        # joe: 使用原始图像（已经去除了中值）进行defringe
        Starts from a collection of images and masks,
        computes the target regularization parameter lambda~\sqrt(npix)\sigma
        using a robust estimation of the image noise level, and
        makes a low-rank fit to the images in the following way:
        1) Computes the SOFT-INPUTE solution with warm start (boosting lambda first)
        2) For a given rank, fixing the fringe modes obtained above, recompute the low-rank coefficients
           with a linear regression (ML solution). This step is important to debias the result.

        """

        # Check if inputs are of the right kind
        if GPU:
            if not isinstance(images, cp.ndarray) or not isinstance(masks, cp.ndarray):
                print("Input arrays must be cupy.ndarray")
                return
        # Compute target lambda:
        npix = images.shape[0]
        nobs = images.shape[1]
        if sigmas is None:
            sigmas = cp.asarray([self.robust_sigma((images * masks)[:, i]) for i in range(nobs)])
        else:
            sigmas = cp.asarray(sigmas)
        scaled_images = images / sigmas[None, :]  # joe: why do this?
        print('sigmas:')
        print(sigmas)

        lamb = nm.sqrt(npix) * nm.sqrt(cp.sum(masks) / (npix * nobs * 1.)) * lfac  # joe: \mu in paper eq(2)
        print('lambda:', lamb)
        print('Computing low-rank solution, with regularization lambda = %.2f' % lamb)

        # Warm start, to speed up convergence. This is the SOFT-INPUTE STEP
        # joe: get what? a defringe image.
        # joe: 10倍lambda会快速筛掉很多特征值
        print('First iterations with lambda*10 (warm start)...')
        Z = self.svd_iterate(scaled_images, masks, lamb * 10., tol=tol, random=random)
        print('Second iterations with target lambda...')
        Z = self.svd_iterate(scaled_images, masks, lamb, Zold=Z, tol=tol, random=random)

        # Now un-shrink via ML solution on current singular vectors, for a given rank
        U, D, VT = cp.linalg.svd(Z, full_matrices=False)

        print('Singular values:')
        print(D[:10])
        #print ('Weights:')
        #print (VT[:10,:].T)
        if unshrinking:
            # joe: 为什么需要进行 unshrinking 操作？
            # 回答：你通过软阈值化操作去掉了噪声，但是图像的有用信息也被减弱了。比如，原本明亮的边缘变得不那么明显了。
            # 为了恢复图像的有用信息，你进行 unshrinking 操作，重新调整图像中的主要成分，使得它们恢复到接近原来的状态。
            # 这样，你不仅去除了噪声，还恢复了图像的清晰度和亮度。
            print('Unshrinking...')
            ##Z = unshrink(scaled_images,masks,Z,rank=rank)
            Z = self.regress_lsingular(scaled_images, masks, U, rank=rank)
            if random is None:
                U, D, VT = cp.linalg.svd(Z, full_matrices=False)
            else:
                U, D, VT = randlin.gpu_random_svd(Z, *random)
            print('Singular values after unshrinking:')
            print(D[:rank + 2])
            #print ('Weights after unshrinking:')
            #print VT[:rank,:].T
        return Z * sigmas[None, :]




    ## =====================================
    # r_pca for fringe pattern
    ## =====================================
    def fringes_pcp(self, images, masks, sigmas=None, tol=None, unshrinking=False,
                    random=None, max_iter=2500, iter_print=100):
        """
        joe: 使用 PCP 方法求 fringe pattern (final)
        """

        # Check if inputs are of the right kind
        if GPU:
            if not isinstance(images, cp.ndarray) or not isinstance(masks, cp.ndarray):
                print("Input arrays must be cupy.ndarray")
                return
        # Compute target lambda:
        npix = images.shape[0]
        nobs = images.shape[1]
        if sigmas is None:
            sigmas = cp.asarray([self.robust_sigma((images * masks)[:, i]) for i in range(nobs)])
        else:
            sigmas = cp.asarray(sigmas)
        scaled_images = images / sigmas[None, :]  # joe: why do this?
        print('sigmas:')
        print(sigmas)

        # using pcp method to do something cool!
        pcp_object = R_pca(scaled_images)
        lowrank_mt, sparse_mt = pcp_object.fit(tol=tol, max_iter=max_iter, iter_print=iter_print)

        if unshrinking:
            # joe: 为什么需要进行 unshrinking 操作？
            # 回答：你通过软阈值化操作去掉了噪声，但是图像的有用信息也被减弱了。比如，原本明亮的边缘变得不那么明显了。
            # 为了恢复图像的有用信息，你进行 unshrinking 操作，重新调整图像中的主要成分，使得它们恢复到接近原来的状态。
            # 这样，你不仅去除了噪声，还恢复了图像的清晰度和亮度。
            print('Unshrinking...')
            # Now un-shrink via ML solution on current singular vectors, for a given rank
            U_svd, D, VT = cp.linalg.svd(lowrank_mt, full_matrices=False)

            lowrank_mt = self.regress_lsingular(scaled_images, masks, U_svd)
            if random is None:
                U, D, VT = cp.linalg.svd(lowrank_mt, full_matrices=False)
            else:
                U, D, VT = randlin.gpu_random_svd(lowrank_mt, *random)
            print('Singular values after unshrinking:')
            rank = 3
            print(D[:rank + 2])
            #print ('Weights after unshrinking:')
            #print VT[:rank,:].T
        return lowrank_mt * sigmas[None, :], sparse_mt






    def svd_iterate(self, X, masks, lam=1., Zold=None, tol=1e-5, trunc=None, hard=False, rank=None, verbose=False,
                    random=None):

        # Solve the nuclear norm regularized problem with
        # partially observed data matrix X
        # X, masks, Zold are (npix,nobs) matrices
        # i.e. images have been flatten along columns
        # This is the "SOFT-INPUTE" algorithm of Mazumeder & Hastie

        # joe: return what? the fringe pattern image.

        npix, nobs = X.shape
        if Zold is None:
            Zold = cp.zeros((npix, nobs))

        frac_err = 1e30
        while frac_err > tol:
            if not hard:
                Znew = self.soft_svd(masks * X + (1.0 - masks) * Zold, lam, trunc=trunc, random=random)
            else:
                if rank is None:
                    rank = 3
                Znew = self.hard_svd(masks * X + (1.0 - masks) * Zold, rank, random=random)
            frac_err = self.frob2(Znew - Zold) / self.frob2(Znew)
            print("fractional error = %g" % frac_err)
            if verbose:
                # Computes chi2 and nuclear norm terms, and print them
                chi2 = 0.5 * self.frob2(masks * (Znew - X)) / (npix * nobs)
                if hard:
                    print("chi2 = %f" % chi2)
                else:
                    nuke = lam * self.nuclear(Znew) / (npix * nobs)
                    print("chi2 = %f, nuclear norm = %f, sum = %f" % (chi2, nuke, chi2 + nuke))

            Zold = Znew

        return Znew

    def soft_svd(self, X, lam, trunc=None, random=(3, 0, 1)):

        # Solves the nuclear norm regularized prob
        # argmin 1/2 ||X-Z||^2 + lam ||Z||*
        # (first norm is Frobenius, second is nuclear norm)
        # Solution is soft-thresholded SVD, ie replace singular
        # values of X (d1,...,dr) by ((d1-lam)+,...,(dr-lam)+)
        # where (t)+ = max(t,0)
        # See Mazumder, Hastie, Tibshirani 2012
        # In case random keyword is not None, computes randomized SVD,
        # inputs are assumed to be equal
        # to (k,s,q), where k is the target rank, s the oversampling, q the number of iterations.

        if random is None:
            U, D, VT = cp.linalg.svd(X, full_matrices=False)
        else:
            U, D, VT = randlin.gpu_random_svd(X, *random)
        rankmax = D.size
        vlam = cp.ones(rankmax) * lam
        if trunc is not None:
            vlam[0:trunc] = 0.
        DD = cp.fmax(D - vlam, cp.zeros(rankmax))  # joe: no less than 0

        return cp.dot(U, cp.dot(cp.diag(DD), VT))

    def hard_svd(self, X, rank, random=(3, 0, 1)):
        # SVD truncation
        # If random is not None, inputs are respectively
        # target rank of randomized SVD, oversampling and
        # number of power iterations.
        if random is None:
            U, D, VT = cp.linalg.svd(X, full_matrices=False)
        else:
            U, D, VT = randlin.gpu_random_svd(X, *random)
        D[rank:] = 0.
        return cp.dot(U, cp.dot(cp.diag(D), VT))

    def frob2(self, X):
        # Square of Frobenius norm
        return cp.sum(X ** 2)

    def L1(self, X):
        # Sum of absolute values of elements
        return cp.sum(cp.abs(X))

    def nuclear(self, X, trunc=None):
        # Nuclear norm, sum of singular values
        s = cp.linalg.svd(X, compute_uv=False, full_matrices=False)
        if trunc:
            return s[trunc:].sum()
        else:
            return s.sum()

    ## This routine is not used anymore, memory footprint too large. Use regress_lsingular instead
    def unshrink(self, images, masks, Z, rank=None, random=(3, 0, 1)):
        """ Computes an ML solution of the regression on the (masked) first n=rank left singular vectors of Z """
        if rank is None:
            rank = 3
        npix = images.shape[0]
        nobs = images.shape[1]
        if random is None:
            U, D, VT = cp.linalg.svd(Z, full_matrices=False)
        else:
            U, D, VT = randlin.gpu_random_svd(Z, *random)
        B = cp.zeros((npix * nobs, rank))
        for i in range(rank):
            B[:, i] = cp.reshape(cp.outer(U[:, i], VT[i, :]) * masks, (npix * nobs,))
        BTB = cp.dot(B.T, B)
        RHS = cp.dot(B.T, cp.reshape(images * masks, (npix * nobs,)))
        alpha = cp.dot(cp.linalg.inv(BTB), RHS)
        # print VT[:rank,:]
        return cp.dot(U[:, :rank], cp.dot(nm.diag(alpha), VT[:rank, :]))

    def regress_lsingular(self, images, masks, U, niter=1, rank=None):
        npix, nobs = images.shape
        if rank is None:
            rank = 3
        UU = U[:, :rank]
        X = images * masks
        Znew = cp.zeros(images.shape, dtype=nm.float32)

        for i in range(nobs):
            B = UU * cp.reshape(masks[:, i], (npix, 1))
            if niter > 1:
                eps = cp.median(nm.abs(X[:, i]) / 100.)
                W = cp.ones((npix, 1), dtype=nm.float32)
                for j in range(niter):
                    res = images[:, i] - Znew[:, i]
                    W = 1. / (1.0 + (res / eps) ** 2) ** 0.5
                    BTB = cp.dot(B.T, B * nm.reshape(W, (npix, 1)))
                    RHS = cp.dot(B.T, W * X[:, i])
                    coeff = cp.dot(cp.linalg.inv(BTB), RHS)
                    Znew[:, i] = cp.dot(UU, coeff)
            else:
                BTB = cp.dot(B.T, B)
                RHS = cp.dot(B.T, X[:, i])
                coeff = cp.dot(cp.linalg.inv(BTB), RHS)
                Znew[:, i] = cp.dot(UU, coeff)
        return (Znew)

    def save_to_fits(self, ori, output_dir='ori', img_size=500):
        """
        * Function: 将 ori 矩阵存储为多个 FITS 文件.

        * Params:
            ori (numpy.ndarray): 形状为 (n_pixel, n_object) 的矩阵，其中每列是一个 500x500 的像素图像
            output_dir (str): 输出 FITS 文件的目录，默认为 'ori'
        """
        # 确保 ori 是二维数组
        if len(ori.shape) != 2:
            raise ValueError("ori 必须是一个二维数组")

        # 计算单个图像的边长
        n_pixel = ori.shape[0]

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存 FITS 文件
        for i in range(ori.shape[1]):
            # 重塑为 img_size x img_size
            image_data = ori[:, i].reshape((img_size, img_size))

            # 创建 FITS Primary HDU
            hdu = fits.PrimaryHDU(image_data)
            # 创建 FITS HDUList
            hdul = fits.HDUList([hdu])
            # 生成 FITS 文件名
            fits_filename = os.path.join(output_dir, f"object_{i + 1}.fits")
            # 写入 FITS 文件
            hdul.writeto(fits_filename, overwrite=True)

            print(f"Saved {fits_filename}")

        print("All FITS files have been saved.")

    def save_all_result(self, savePath, ori, lowRankCoeff, sparseCoeff, pcaDefringeImg, width=500, height=500):
        # 检查目录是否存在，如果不存在则创建
        output_dir = os.path.dirname(savePath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取行数、列数
        npix, nobs = ori.shape

        # 根据行数，知道原图的尺寸
        selfWidth = width

        # 分别读取各个列，依次保存各个变量
        # 保存 ori (存放在 Ori 文件夹下)
        savePath_ori = os.path.join(savePath, 'Ori')
        self.save_to_fits(ori, output_dir=savePath_ori, img_size=selfWidth)
        # 保存 lowRankCoeff (存放在 LowRank 文件夹下)
        savePath_lowrank = os.path.join(savePath, 'LowRank')
        self.save_to_fits(lowRankCoeff, output_dir=savePath_lowrank, img_size=selfWidth)
        # 保存 sparseCoeff (存放在 Sparse 文件夹下)
        savePath_sparseCoeff = os.path.join(savePath, 'Sparse')
        self.save_to_fits(sparseCoeff, output_dir=savePath_sparseCoeff, img_size=selfWidth)
        # 保存 pcaDefringeImg (存放在 Defringe 文件夹下)
        savePath_pcaDefringe = os.path.join(savePath, 'Defringe')
        self.save_to_fits(pcaDefringeImg, output_dir=savePath_pcaDefringe, img_size=selfWidth)
        return

def doit(config):
    """
    Example driver routine
    """
    # First instantiate fringe class, reads and prepare all images, masks, etc.
    g = fringe(config)

    # raw fringe pattern
    fimages = g.raw_defringe(g.images, mask=g.chipmask, nxy=config.nxy,
                             robust=config.robust, keep_background=config.keep_background)

    # mask (include： source mask + bad pixel mask)
    masks = g.create_masks(fimages, g.chipmask, kappa=config.kappa, tol=config.tol,
                        include_inputmask=config.include_inputmask, robust=config.robust)

    # 计算每张图像的标准差（更准确，能去除异常值干扰）
    sigmas = cp.asarray([g.robust_sigma(fimages[:, i]) for i in range(fimages.shape[1])])

    # joe: 计算 exact fringe pattern.
    lowrank_mt, spare_mt = g.fringes_pcp(g.images, masks, sigmas=sigmas, tol=config.tol,
            unshrinking=config.unshrinking, max_iter=config.max_iter, iter_print=config.iter_print)

    # 获得最终版本的 defringe images
    pca_defringe = g.mine_defringe(g.images, lowrank_mt, mask=g.chipmask, robust=config.robust)  # defringe图像

    # 保存结果
    g.save_all_result(config.output_fits_path, g.images, lowrank_mt, spare_mt, pca_defringe)

    return

if __name__ == "__main__":
    config = Config()
    doit(config)