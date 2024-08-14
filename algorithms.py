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
GPU = False

from images import data
import randlin

# joe: import new package
from r_pca import R_pca

class fringe(data):

    def __init__(self, rootdir=data.rootDir, oridir=data.oriDataDir,
                 elixirdir=data.elixirDataDir, chipmask_file=data.CHIPMASK_FILE,
                 ccdnum=data.CCDNUM, small=True):

        # Initialize parent data class
        super().__init__(rootdir=rootdir, oridir=oridir,
                         elixirdir=elixirdir, chipmask_file=chipmask_file,
                         ccdnum=ccdnum, small=small)

        return

    def raw_defringe(self, images, mask=None, nxy=None, robust=False, keep_background=False):
        """ In this phase, we remove background (source) and a estimated fringe
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
        mfringe = self.raw_fringe(images, mask=mask, nxy=nxy)  # 估计出一个 median fringe pattern
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
    def mine_defringe(self, images, fringe_pattern, mask=None, nxy=None, robust=False, keep_background=False):
        """
        * Function:
          原图 / Robust_PCA 计算得到的fringe pattern 两者相减
          减法进行时，需要先对fringe pattern进行regress，
          原因：这个fp是综合多张图像得到的，需要保证它与ori在coefficient上保持一致

        * Param:
            image: ori
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
        mfringe = fringe_pattern
        fimages = images.copy()

        # Compute regression coefficients, and remove background and fringe pattern
        for i in range(nobs):
            print('Raw-defringing image number %d' % i)
            if mask.ndim > 1:
                coeffs = self.regress(images[:, i], mfringe, mask=mask[:, i], robust=robust)
            else:
                coeffs = self.regress(images[:, i], mfringe, mask=mask, robust=robust)
            if keep_background:
                fimages[:, i] = images[:, i] - coeffs[1] * mfringe  # joe:only remove fringe pattern
            else:
                fimages[:, i] = images[:, i] - coeffs[0] - coeffs[
                    1] * mfringe  # joe:further remove background c[0] term
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

    def fringes(self, images, masks, tol=1e-10, rank=None,
                unshrinking=True, sigmas=None, lfac=1.0, random=None):
        """
        # joe: 使用原始图像（已经去除了中值），求 fringe pattern (final)
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
        # joe: get what? a fringe image.
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


def doit(ccdnum=13, rank=4, small=True, unshrinking=True, lfac=1.0):
    """
    Example driver routine
    """
    # First instantiate fringe class, reads and prepare all images, masks, etc.
    g = fringe(ccdnum=ccdnum, small=small)

    # Compute roughly defringed images. Will be used to compute robust estimates of noise level
    # (otherwise noise level estimates dominated by fringe variations)
    # joe: remove background and median fringe pattern, get roughly defringed images.
    # joe: 结果为 fringe removal image. (减去背景、减去fringe pattern)(并非本文的最终结果)
    fimages = g.raw_defringe(g.images, mask=g.chipmask, robust=config.robust)

    # Create source mask from images and chipmask
    # masks 屏蔽了source和chip bad pixels.
    # 注意调整 kappa 值（会影响 mask 的准确度）
    masks = g.create_masks(fimages, g.chipmask, robust=config.robust)

    # Compute robust estimates of noise levels using roughly defringed images
    # 每列 fringe removal image 计算一个sigma. (可以理解为是标准差)
    sigmas = cp.asarray([g.robust_sigma(fimages[:, i]) for i in range(fimages.shape[1])])

    # Now compute fringes using multiple modes
    # If "unshrinking" is True, regress on the main left singular vectors
    # (held fixed) to unbias the singular values
    # joe: 计算 exact fringe pattern.
    Z = g.fringes(g.images, masks, rank=rank, tol=1e-12, sigmas=sigmas,
                  unshrinking=unshrinking, lfac=lfac)

    # # #%% do PCP(Robust PCA) method
    # # Params:
    # # Return:
    # r = R_pca(g.images)
    # L, S = r.fit(max_iter=2500, iter_print=100)
    # visually inspect results (requires matplotlib)
    # r.plot_fit()
    # plt.show()

    # pickle_rpca = 'defringed-ccd-%02d_rpca.npy' % ccdnum
    # # Store results in pickle file
    # with open(pickle_rpca, 'wb') as f:
    #     nm.save(f, L)
    #     nm.save(f, S)
    # f.close()
    #
    # pca_defringe = g.mine_defringe(g.images, L)  # 减值defringe图像
    #
    # # # %%
    # if unshrinking:
    #     pickle = 'defringed-ccd-%02d.npy' % ccdnum
    # else:
    #     pickle = 'defringed-nounshrinking-ccd-%02d.npy' % ccdnum
    #
    # # Store results in pickle file
    # with open(pickle, 'wb') as f:
    #     nm.save(f, g.images)
    #     nm.save(f, fimages)
    #     nm.save(f, g.elximages)
    #     nm.save(f, Z)
    #     nm.save(f, g.image_medians)
    #     nm.save(f, masks)
    #     nm.save(f, pca_defringe)
    # f.close()

    return

#
# if __name__ == "__main__":
#     ccdnum = 0
#     small = True
#     unshrinking = False
#     doit(ccdnum=ccdnum, small=small, unshrinking=unshrinking)
