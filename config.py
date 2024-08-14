class Config:
    def __init__(self):
        # General paths and parameters
        self.rootDir = r'C:\Users\Administrator\Desktop'
        self.oriDir = r'nodefringe_data_lzzh\small_part_split_result\p00_1\inputs'
        self.elixirDataDir = r'elixir-defringe_data_lzzh\small_part_split_result\p00_1\inputs'
        self.chipmask_file = r'D:\Datasets\Defringe_data\masks\2003A.mask.0.36.02.fits'
        # self.output_base_dir = r''
        self.output_fits_path = r'D:\Datasets\output_image'  # �滻Ϊ���Ŀ��·��

        # specific megacam chips, should be adaptive to your own needs

        self.CCDNUM = 0  # ���������ͬ����ʱ�䡢�ڼ���CCDͼ ���������Ҫƥ�䣩
        self.cut_overscan = True # get data without the border details of the image
        self.OVERSCAN_X_MIN = 32  #
        self.OVERSCAN_X_MAX = 2080  #
        self.OVERSCAN_Y_MIN = 0  #
        self.OVERSCAN_Y_MAX = 4612  #
        self.WINSMALL = 500  # ����
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
        self.max_iter = 2500
        self.iter_print = 100

        ## raw_defringe function (get raw fringe pattern)
        self.keep_background = False
        self.nxy = None


