from astropy.io import fits as pf
import numpy as nm

# try:
#     import cupy as cp
#     GPU = True
# except:
#     print("cupy not installed. This probably means there is no GPU on this host.")
#     print("Using numpy instead")
#     import numpy as cp
#
#     GPU = False

import numpy as cp
CPU = False

class data(object):
    """
    This class will be used to read images, create masks, etc.
    and form the data matrix to be processed for fringe removal
    """

    ODOMETER_FILE = "fringe_data_zzj\\load_img.txt"
    ROOTDIR = r"G:\我的云端硬盘\GoogleDrive_Codes\defringe_main_lzzh\\"
    CHIPMASK_FILE = "masks\\2003A.mask.0.36.02.fits"  # 这个文件是一个MEF文件（Multi Extend File), 经查看，一共有35个Extends.
    # THESE ARE SPECIFIC TO THE MEGACAM CHIPS, SHOULD BE ADAPTED TO YOUR OWN NEEDS

    CCDNUM = 13  # 分析多个不同拍摄时间、第几张CCD图
    OVERSCAN_X_MIN = 32  # ?
    OVERSCAN_X_MAX = 2080  # ?
    OVERSCAN_Y_MIN = 0  # ?
    OVERSCAN_Y_MAX = 4612  # ?
    WINSMALL = 500  # 窗口

    def __init__(self, odometer_file=ODOMETER_FILE, rootdir=ROOTDIR, chipmask_file=CHIPMASK_FILE, ccdnum=CCDNUM,
                 small=True):

        # First create list of image names. We will assume here that the images are in 'split' format,
        # i.e. one separate FITS file per CCD.
        # joe: 生成图像names的目录，以list形式
        # joe: 这里的 ori_image 图像到底是什么？没有经过defringe处理，包含fringe波纹的图像\
        self.names = self.create_image_names(rootdir + odometer_file, rootdir + 'nodefringe', [ccdnum])
        # Now read the images, and store them as column vectors of the output matrix. Thus, each column of the matrix
        # contains all pixels of a given image.
        self.images = self.read_images(self.names, small=small)
        self.ori_image = self.images

        # Now do the same for images corrected with a robust regression on a single median fringe template
        # joe: 这里的 elximages 图像是什么？ 经过标准流程处理，已经去除了fringe波纹的图像\
        self.elxnames = self.create_image_names(rootdir + odometer_file, rootdir + 'elixir-defringe', [ccdnum])
        elximages = self.read_images(self.elxnames, small=small)
        self.elximages = cp.asarray(elximages)

        # Remove median per image of elximages, for plotting purposes
        elximage_medians = nm.median(self.elximages, axis=0)  # column:axis=0 计算所有图像的每个像素位置的中值，列向量
        self.elximage_medians = cp.asarray(elximage_medians)
        # self.image_medians[None, :] 添加了一个新的维度，用于广播运算；列向量在水平方向广播
        # 背景（中值图像）被去除，剩下的差异（变化的信号）被保留在 self.images 中
        self.elximages -= self.elximage_medians[None, :]      # 每幅图像减去一幅中值图像

        # Now do the same for the original images
        self.image_medians = cp.median(self.images, axis=0) # 计算所有图像，每个相同位置像素的中值
        self.images -= self.image_medians[None, :]  # 相减，减掉了中值（这个中值是多幅图像综合下的中值）

        # Now read the corresponding mask for dead/hot pixels, etc.
        chipmask = self.read_chipmask(rootdir + chipmask_file, ccdnum, small=small)
        self.chipmask = cp.asarray(chipmask)

        return

    def create_image_names(self, odomfile, rootdir, ccdlist, mef=False):
        """
        * Function:
          Store filenames of all the images in a list

        * Input:
          odomfile - filename indices of all the images taken during a run
          rootdir  - where the .fits stored. example: xxx/xxx/nodefringe
          ccdlist  - 0 to 35
          mef(optional) - multi-extension fit file

        * Output: A list with all the filenames

        * Example:
        names=firstpass.create_image_names('../odomlist_z_14Am01',
                                            '../z_14Am01_nodefringe',
                                            [0])
        """
        odometers = nm.loadtxt(odomfile, dtype=nm.unicode_)
        ccdlist = ["%02d" % int(i) for i in ccdlist]
        image_names = []
        for odom in odometers:
            if mef:
                # image_names += [rootdir + '/' + odom + 'p.fits']
                image_names += [rootdir + '\\' + odom + 'p.fits']
            else:
                # image_names += [rootdir + '/' + odom + 'p/' + odom + 'p' + ccd + '.fits' for ccd in ccdlist]
                image_names += [rootdir + '\\' + odom + 'p\\' + odom + 'p' + ccd + '.fits' for ccd in ccdlist]
        return image_names

    def read_images(self, image_names, cut_overscan=True, small=False, mef=False, ccd=0):
        """
        * Function:
          Read data from magacam images. Take data from each image as a column vector,
        and store the column vectors in a large matrix.

        * Params:
          - cut_overscan: get data without the border details of the image
          - small: only get data of a small window size (500x500) from the original image
          - ccd: 0-35 are valid numbers

        * Return:
          matrix, each column is an image.

        """
        images = []
        for name in image_names:
            if mef:
                image = pf.getdata(name, ccd + 1)
            else:
                image = pf.getdata(name)  # Split mode, image in primary hdu
            image = image.T  # joe: transpose operation
            if cut_overscan:
                image = image[self.OVERSCAN_X_MIN:self.OVERSCAN_X_MAX, self.OVERSCAN_Y_MIN:self.OVERSCAN_Y_MAX]
            if small:
                image = image[:self.WINSMALL, :self.WINSMALL]
                images.append(cp.reshape(image, (self.WINSMALL * self.WINSMALL,)))
            else:
                images.append(cp.reshape(image, (
                    (self.OVERSCAN_X_MAX - self.OVERSCAN_X_MIN) * (self.OVERSCAN_Y_MAX - self.OVERSCAN_Y_MIN),)))
        return cp.transpose(cp.asarray(images))

    def read_chipmask(self, MEF, chipnum, cut_overscan=True, small=False):
        """
        Function:
            Return a mask (as column vector) for specified CCD. Input is a full FP pixel mask as a MEF file.

        Params:
            MEF: 具体的文件地址。mask文件里面有 35张 ext mask.
            chipnum: 0-35 are valid numbers
            cut_overscan: get data without the border details of the image
            small: only get data of a small window size (500x500) from the original image

        Return:
            chipmask: 返回的mask (第chipnum张mask)

        """

        f = pf.open(MEF)
        chipmask = cp.asarray(f[chipnum + 1].data, dtype=float)
        chipmask = chipmask.T * 1.0
        f.close()
        if cut_overscan:
            chipmask = chipmask[self.OVERSCAN_X_MIN:self.OVERSCAN_X_MAX, self.OVERSCAN_Y_MIN:self.OVERSCAN_Y_MAX]
        if small:
            chipmask = chipmask[:self.WINSMALL, :self.WINSMALL]
            return cp.reshape(chipmask, (self.WINSMALL * self.WINSMALL,))
        else:
            return cp.reshape(chipmask, (
                (self.OVERSCAN_X_MAX - self.OVERSCAN_X_MIN) * (self.OVERSCAN_Y_MAX - self.OVERSCAN_Y_MIN),))
        return chipmask

    # #%%

    # import astropy.io.fits as fits
    #
    # def check_hdu_types(file_path):
    #     """
    #     检查 FITS 文件中 HDU 的类型和名称
    #     """
    #     with fits.open(file_path) as hdul:
    #         for i, hdu in enumerate(hdul):
    #             print(f"HDU {i}: Type = {type(hdu)}, Name = {hdu.name}")
    #
    # # 使用示例
    # file_path = r'D:\Datasets\Defringe_data\masks\2003A.mask.0.36.02.fits'
    # check_hdu_types(file_path)
# # %%

    def create_masks(self, images, chipmask, kappa=2.0, tol=1e-10, include_inputmask=True, robust=False):
        """
        * Function:
          Apply mask_sources to all images to create the overall mask (source mask + bad pixel mask)
          Assumes images is (npix,nobs), and chipmask is (npix,) in chipmask_mode
          or (npix,nobs) in imagemask_mode

        * Parameters:
            images: 原图, 格式（npix，nobs）
            chipmask：mask for dead/hot pixels
            kappa: kappa.sigma中的kappa （这个值越小，越多点被视为source; 这个值越大，越少点被视为source.）
                         换一种说法：结果的mask点太多（除了real source，其他零散点很多1），应该怎么做？适当增加kappa值。
            tol: 迭代停止条件
            include_inputmask:

        * Return:
            masks： overall mask 格式(npix,nobs)

        """
        ndim_mask = chipmask.ndim
        imagemask_mode = False
        if ndim_mask > 1:
            imagemask_mode = True

        npix = images.shape[0]
        nobs = images.shape[1]
        masks = cp.ndarray((npix, nobs), dtype=nm.float64)  # create new null array
        for i in range(nobs):
            if imagemask_mode:
                masks[:, i] = self.mask_sources(images[:, i], chipmask[:, i], kappa, tol, include_inputmask, robust)
            else:
                masks[:, i] = self.mask_sources(images[:, i], chipmask, kappa, tol, include_inputmask, robust)
        return masks

    def mask_sources(self, image, mask, kappa=2.0, tol=1e-10, include_inputmask=True, robust=True):
        # Iterative kappa sigma-clipping to flag bright sources
        # Outputs a new mask based on the input mask
        # joe: 生成的mask, 同时遮罩bad pixels和sources（值为0）

        msk = cp.ones(cp.shape(mask), dtype=nm.float64)
        img = image.copy()
        img *= mask  # 先遮罩 bad pixels.

        if robust is not True:
            rms = img.std()
            frac_err = 1e30
            while frac_err > tol:
                rms_old = rms
                clip = cp.where((img - cp.median(img)) / rms_old > kappa)  # 大于 kappa, 视为背景点
                msk[clip] = 0.0
                img *= msk
                rms = img.std()
                frac_err = cp.abs(rms - rms_old) / rms_old
                print("fractional rms variation = %.3f" % frac_err)
        else:
            # robust=True, 走这个分支
            rms = self.robust_sigma(img)
            clip = cp.where((img - cp.median(img)) / rms > kappa)
            msk[clip] = 0.0
            img *= msk

        if include_inputmask:
            return msk * mask
        else:
            return msk

    def robust_sigma(self, X):
        # Computes estimate of rms via MAD
        # joe: 1.4826 是一个用于将 MAD 转换为与标准差等价值的比例因子，适用于正态分布数据。
        # MAD 是一种稳健的统计量，不受异常值影响，用于估计数据的集中趋势的变异性。
        # 这种转换因子在稳健统计中非常有用，特别是当数据集包含异常值时，通过 MAD 提供了对标准差的更稳健估计。
        return 1.4826 * cp.median(cp.abs(X - cp.median(X)))
