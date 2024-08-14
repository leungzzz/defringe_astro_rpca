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
        self.S = np.zeros(self.D.shape) # ��ʼ��ϡ����� S Ϊ�����
        self.Y = np.zeros(self.D.shape) # ��ʼ���������ճ������� Y Ϊ�����

        # ���ø��²��� ( ��Ϊ None��������������ݵ���״�Զ�����)
        if mu: self.mu = mu
        else: self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))
        self.mu_inv = 1 / self.mu

        # ���򻯲��������ڿ���ϡ����
        if lmbda: self.lmbda = lmbda
        else: self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    # ����ֵ���� >>> ʹ����Ԫ�ذ�һ����ֵ tau ��������Ŀ���ǽ��ӽ����Ԫ�ع��㣬�Ӷ�����ϡ����
    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    # 1) �Ծ����������ֵ�ֽ�; 2) ������ֵ��������ֵ����; 3) �ؽ��ָ�
    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    # RPCA �ĺ����㷨 ���� D �ֽ�Ϊ���Ⱦ��� L �� ϡ����� S ��
    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S     # ϡ�����
        Yk = self.Y     # ���忴ԭ�ģ�������ı���
        Lk = np.zeros(self.D.shape)     # ���Ⱦ���

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)   # ����ֹͣ������֮һ��

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
