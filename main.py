# main.py
import sys

import numpy as np
import logging
from datetime import datetime

from config import Config
from image_processor import ImageProcessor
from fringe_removal_algorithm_RPCA import FringeRemovalAlgorithm

# 配置日志记录
log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 重定向标准输出和标准错误到日志文件
sys.stdout = open(log_filename, 'a')
# sys.stderr = open(log_filename, 'a')

def doit(config):
    """
    Example driver function to run the fringe removal process.

    Parameters:
        config (Config): The configuration object containing parameters for processing.
    """
    # do some init work (basic)
    image_processor = ImageProcessor(config)

    # prepare some core algorithms
    algorithm = FringeRemovalAlgorithm(image_processor)

    # 获得初步的fringe pattern
    fimages = algorithm.raw_defringe(image_processor.images, mask=image_processor.chipmask,
                                     nxy=config.nxy, robust=config.robust,
                                     keep_background=config.keep_background)

    # source mask + bad pixel mask
    masks = image_processor.create_masks(fimages, image_processor.chipmask, kappa=config.kappa,
                                         tol=config.tol, include_inputmask=config.include_inputmask,
                                         robust=config.robust)

    # get median (a robust way)
    sigmas = np.array([image_processor._robust_sigma(fimages[:, i]) for i in range(fimages.shape[1])])

    # core, using pcp to get low rank matrix, sparse matrix
    lowrank_mt, spare_mt = algorithm.fringes_pcp(image_processor.images, masks, sigmas=sigmas,
                                                 tol=config.tol, unshrinking=config.unshrinking,
                                                 max_iter=config.max_iter, iter_print=config.iter_print)

    # core, using low rank matrix to get defringe image
    pca_defringe = algorithm.defringe_pcp(image_processor.images, lowrank_mt,
                                          mask=image_processor.chipmask, nxy=config.nxy,
                                           robust=config.robust, keep_background=config.keep_background)

    # save result
    algorithm.save_all_result(config.output_fits_path, image_processor.images,
                              lowrank_mt, spare_mt, pca_defringe)

if __name__ == "__main__":
    config = Config()
    doit(config)
    # 关闭文件流
    sys.stdout.close()
    # sys.stderr.close()