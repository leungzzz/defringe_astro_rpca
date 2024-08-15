"""
this code download from https://github.com/dganguli/robust-pca
"""
from __future__ import division, print_function

import numpy as np

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass

class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape) # 初始化稀疏矩阵 S 为零矩阵
        self.Y = np.zeros(self.D.shape) # 初始化拉格朗日乘数矩阵 Y 为零矩阵

        # 设置更新步长 ( 若为 None，则根据输入数据的形状自动计算)
        if mu: self.mu = mu
        else: self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))
        self.mu_inv = 1 / self.mu

        # 正则化参数，用于控制稀疏性
        if lmbda: self.lmbda = lmbda
        else: self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    # 软阈值操作 >>> 使矩阵元素按一定阈值 tau 收缩。其目的是将接近零的元素归零，从而产生稀疏性
    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    # 1) 对矩阵进行奇异值分解; 2) 对奇异值进行软阈值收缩; 3) 重建恢复
    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    # RPCA 的核心算法 （将 D 分解为低秩矩阵 L 和 稀疏矩阵 S ）
    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S     # 稀疏矩阵
        Yk = self.Y     # 具体看原文，新引入的变量
        Lk = np.zeros(self.D.shape)     # 低秩矩阵

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)   # 迭代停止条件（之一）

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf

        while (err > _tol) and iter < max_iter:
            # this line implements step 3
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)

            # this line implements step 4
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)

            # this line implements step 5
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
