class Config:
    def __init__(self):
        """
        Configuration for fringe removal process.
        """

        # General paths and parameters
        self.rootDir = r'C:\Users\Administrator\Desktop'
        self.oriDir = r'part_data_just_for_test\ori_data'
        # self.elixirDataDir = r'elixir-defringe_data_lzzh\small_part_split_result\p00_1\inputs'
        self.chipmask_file = r'D:\Datasets\Defringe_data\masks\2003A.mask.0.36.02.fits'

        # 目标路径
        self.output_fits_path = r'D:\Datasets\output_image'

        # specific megacam chips, should be adaptive to your own needs

        self.CCDNUM = 0  # 分析多个不同拍摄时间、第几张CCD图 （和上面的要匹配）
        self.cut_overscan = True # get data without the border details of the image
        self.OVERSCAN_X_MIN = 0  #
        self.OVERSCAN_X_MAX = 1900  #
        self.OVERSCAN_Y_MIN = 0  #
        self.OVERSCAN_Y_MAX = 1900  #
        self.WINSMALL = 500  # 窗口
        self.mef = False   # chip mask is MEF or not, default is False.
        self.small = True
        self.unshrinking = False
        self.height = 1900
        self.width = 1900

        # source mask function
        self.kappa = 2.0
        self.tol = 1e-10
        self.include_inputmask = True
        self.robust = True

        # fringes_pcp function
        self.max_iter = 300  # 2500 的效果可行，已经试验过~ 5000 更佳
        self.iter_print = 50

        ## raw_defringe function (get raw fringe pattern)
        self.keep_background = False
        self.nxy = None

        ## regress
        self.niter = 10


