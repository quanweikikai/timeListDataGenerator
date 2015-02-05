#!-*-coding:utf-8-*-

from ArtificialDataGenerator import ArtificialDataGenerator
import numpy as np
import random
from numpy import pi,exp
import warnings
import sys

#def calcMultivariateNormal(y, mean, covariance, stabilization_item=0):
##    return float(1/(np.linalg.det(2*pi*covariance))**0.5 * exp(-0.5* (y-mean).T * np.linalg.inv(covariance) * (y-mean)))
#    #大きい行列を入れると普通のdetはoverflowで落ちる
#    sign,logdet = np.linalg.slogdet(2*pi*covariance) #正定値行列のdetより常に正なのでsignは捨てる
#    det = np.exp(logdet - stabilization_item) #普通に計算するとeの2000乗とかになるので適当な項を引く このあと結局gamma=(割合)にするので等価
#
#    ####TODO ちゃんと確認する
#    return float((1/det)**0.5 * exp(-0.5* (y-mean).T * np.linalg.inv(covariance) * (y-mean)))


def calcMultivariateNormal(y, mean, covariance, covariance_inv, stabilization_item=0):
    #大きい行列を入れると普通のdetはoverflowで落ちる
    sign,logdet = np.linalg.slogdet(2*pi*covariance) #正定値行列のdetより常に正なのでsignは捨てる
    det = np.exp(logdet - stabilization_item) #普通に計算するとeの2000乗とかになるので適当な項を引く このあと結局gamma=(割合)にするので等価
    return float((1/det)**0.5 * exp(-0.5* (y-mean).T * covariance_inv * (y-mean)))

class PCCAEMalgorithmn(object):
    def __init__(self, data, K):
        self.data = data
        self.K = K
        dimension_1 = data[0][0].shape[0]
        dimension_2 = data[0][1].shape[0]
        dimension_x = data[0][2].shape[0]
        dimension_t = min(dimension_1, dimension_2)

        print "data dimension %d %d %d"%(dimension_1, dimension_2, dimension_x)

        self.gamma = np.zeros((len(data), self.K))

        self.pi = [np.random.rand() for i in range(self.K)]
        self.pi /= sum(np.array(self.pi))
        self.mu = [np.matrix(np.random.random((dimension_1 + dimension_2, 1))) for i in range(self.K)]
        self.Wx = [np.matrix(np.random.random((dimension_1 + dimension_2, dimension_x))) for i in range(self.K)]

        self.C = []
        self.Psi = []
        self.Wt_square = []
        for k in range(self.K):
            Wt_k = np.matrix(np.random.random((dimension_1 + dimension_2, dimension_t)))
            self.Wt_square.append(Wt_k * Wt_k.T)
            temp_matrix1_k = np.matrix(np.random.random((dimension_1, dimension_1)))
            temp_matrix2_k = np.matrix(np.random.random((dimension_2, dimension_2)))
            Psi1_k = temp_matrix1_k.T * temp_matrix1_k
            Psi2_k = temp_matrix2_k.T * temp_matrix2_k
            zero_matrix1 = np.matrix(np.zeros((dimension_1, dimension_2)))
            zero_matrix2 = np.matrix(np.zeros((dimension_2, dimension_1)))
            Psi_k = np.r_[np.c_[Psi1_k, zero_matrix1] , np.c_[zero_matrix2, Psi2_k]]
            self.Psi.append(Psi_k)
            C_k = Wt_k * Wt_k.T + Psi_k
            self.C.append(C_k)

    def calcOneStep(self):
        #Estep calc gamma
        _,logdet = np.linalg.slogdet(2*pi*self.C[0]) #TODO これ入れていても落ちる 対数のままで計算できる？ 計算が落ちないようにする安定化項
        C_inv = [np.linalg.inv(C_k) for C_k in self.C]

        for n,data_n in enumerate(self.data):
            sys.stdout.write("\r Estep - %d / num_data"%n)
            sys.stdout.flush()

            y1 = data_n[0]
            y2 = data_n[1]
            y = np.r_[y1,y2]
            x  = data_n[2]

            weighted_pdf = []
            for k in range(self.K):
                mean = self.Wx[k] * x + self.mu[k]
                covariance = self.C[k]
                covariance_inv = C_inv[k]
                weighted_pdf.append(self.pi[k] * calcMultivariateNormal(y,mean, covariance, covariance_inv, logdet)) #TODO pymc にmv_normal_cov_like という関数があるので，そっちに置き換えたほうがいいかも
                ##
            try: #TODO 尤度が小さくなりすぎて0divになることがあるので，雑な対応をした．他にどうすれば??対数尤度??
#                weighted_pdf.append(self.pi[k] * calcMultivariateNormal(y,mean, covariance))
                self.gamma[n] = weighted_pdf/sum(weighted_pdf)
            except RuntimeWarning:
                self.gamma[n] = [1.0/self.K for k in range(self.K)]
        #Mstep
        y = np.array([np.r_[y1,y2] for i,[y1,y2,_] in enumerate(self.data)])
        x = np.array([x_n for i,[y1,y2,x_n] in enumerate(self.data)])

        #calc mu
        sys.stdout.write("\r Mstep : calc mu         ")
        sys.stdout.flush()

        sum_gamma = np.sum(self.gamma, axis=0)
        self.mu = [sum([self.gamma[n][k] * (y[n] - self.Wx[k] * x[n]) for n,_ in enumerate(self.data)])/sum_gamma[k] for k in range(self.K)] #TODO 内包表記は悪 わかりやすく書きなおす

        #calc pi
        sys.stdout.write("\r Mstep : calc pi         ")
        sys.stdout.flush()

        self.pi = sum_gamma/len(self.data)

        #calc Wx
        sys.stdout.write("\r Mstep : calc Wx          ")
        sys.stdout.flush()
        for k in range(self.K):
            y_tilde = y - sum([self.gamma[n][k] * y[n] for n,_ in enumerate(self.data)])/sum_gamma[k]
            x_tilde = x - sum([self.gamma[n][k] * x[n] for n,_ in enumerate(self.data)])/sum_gamma[k]
            self.Wx[k] = np.matrix(sum([self.gamma[n][k] * y_tilde[n] * x_tilde[n].T for n,_ in enumerate(self.data)])) * np.matrix(sum([self.gamma[n][k] * x_tilde[n] * x_tilde[n].T for n,_ in enumerate(self.data)])).I #TODO 内包表記は悪
#        self.Wx[k] = [np.matrix(np.zeros((self.data[0][0].shape[0]+self.data[0][1].shape[0],self.data[0][2].shape[0]))) for i in range(self.K)] #Wxを常に0にしておけば正準相関分析と等価なはずなので比較に使える？

        #calc Wt
        sys.stdout.write("\r Mstep-calc Wt            ")
        sys.stdout.flush()

        S = []
        self.cluster_eigen_vecs = []
        self.cluster_eigen_vals = []
        for k in range(self.K):
            y_tilde = y - sum([self.gamma[n][k] * y[n] for n,_ in enumerate(self.data)])/sum_gamma[k] #TODO 速度が気になるならまとめる
            x_tilde = x - sum([self.gamma[n][k] * x[n] for n,_ in enumerate(self.data)])/sum_gamma[k]
            S_k = 0
            for n,gamma_n in enumerate(self.gamma):
                temp_matrix = y_tilde[n] - self.Wx[k] * x_tilde[n]
                S_k += gamma_n[k] * temp_matrix * temp_matrix.T
            S_k /= sum_gamma[k]

#            S_k = sum([self.gamma[n][k] * (y_tilde[n] - self.Wx[k] * x_tilde[n]) * (y_tilde[n] - self.Wx[k] * x_tilde[n]).T for n,_ in enumerate(self.data)])/sum_gamma[k]
            S.append(S_k)
            eigen_vals, eigen_vecs = np.linalg.eig(S_k)
            self.cluster_eigen_vals.append(eigen_vals)
            self.cluster_eigen_vecs.append(eigen_vecs)
            self.Wt_square[k] = eigen_vecs * (np.diag(eigen_vals) - self.Psi[k]) * eigen_vecs.T


        #calc Psi
        sys.stdout.write("\r Mstep-calc Psi          ")
        sys.stdout.flush()

        for k in range(self.K):
            self.Psi[k] = S[k] - self.Wt_square[k]
            self.C[k] = self.Wt_square[k] + self.Psi[k]
        sys.stdout.write("\r                         ")

        return self.gamma


    def calcUntilNoChangeClustering(self):
        warnings.filterwarnings('error') #0divを警告の段階でcatchして処理するため
        serial_results = []
        serial_results.append(self.calcOneStep().argmax(axis = 1))
        while(1):
            serial_results.append(self.calcOneStep().argmax(axis = 1))
            print ""
            print "%d step - updated %d labels"%(len(serial_results),(np.count_nonzero(serial_results[-1] - serial_results[-2])))
            if(np.array_equal(serial_results[-2], serial_results[-1])):
                warnings.resetwarnings() #警告の処理を元に戻しておかないとどうでもいい警告で落ちる
                return serial_results

    def calcUntilNoChangeGamma(self):
        serial_results = []
        serial_gamma = []
        serial_gamma.append(self.calcOneStep().copy())
        serial_results.append(serial_gamma[-1].argmax(axis = 1))
        while(1):
            serial_gamma.append(self.calcOneStep().copy())
            serial_results.append(serial_gamma[-1].argmax(axis = 1))
            print len(serial_results)
            print serial_results[-1]
            print np.count_nonzero(serial_gamma[-1] - serial_gamma[-2])
            print serial_gamma[-1] - serial_gamma[-2]


            if(np.array_equal(serial_gamma[-2], serial_gamma[-1])):
                return serial_results

    def getParamsDictionary(self):
        params_dic = {
            "gamma":self.gamma,
            "pi":self.pi,
            "mu":self.mu,
            "Wx":self.Wx,
            "C":self.C,
            "Psi":self.Psi }
        return params_dic

if __name__ == '__main__':
    generator = ArtificialDataGenerator([0.3,       0.5,       0.2],
                                        ["params1", "params2", "params3"])
    data =  generator.generate(1000)
    EM = PCCAEMalgorithmn(data, 3 )

#    kmeans_results = generator.clusteringDataWithKmeans(3)

    ######### 過去の方法でやるやつ そんなにいらない
#    from CausalPatternExtractor import CausalPatternExtractor
#    from random import randint
#    data = np.array(generator.get_data())
#    print data.shape
#    y1 = data[:,0:2]
#    y2 = data[:,2:5]
#    x  = data[:,5:9]
#    frames_clusterLabels = [randint(0,2) for i,_ in enumerate(x)]
#    clusterLabel_frames = []
#    for label_i in range(max(frames_clusterLabels)+1):
#        clusterLabel_frames.append([i for i,j in enumerate(frames_clusterLabels) if j==label_i])
#    extractor = CausalPatternExtractor(y1, y2, x, clusterLabel_frames)
#    step_sumErrors, frames_clusterLabels = extractor.extract(1000)
#    print frames_clusterLabels
   #########


    serial_results = EM.calcUntilNoChangeClustering()
#    serial_results = EM.calcUntilNoChangeGamma()
    correct_cluster_labels = generator.get_z()

#    from Plotter import plot3dData, matchClusterLabels #ファイル名とか考えなおす
#
#    correct_cluster_labels, serial_results = matchClusterLabels(correct_cluster_labels, serial_results,3)
#    from Plotter import plotErrorChange
#    plotErrorChange(correct_cluster_labels, serial_results)
#
#    plot3dData(generator.get_pcaedData(3),
#               np.array(correct_cluster_labels),
#               np.array(serial_results[-1]),
#               np.array(kmeans_results),#)
#               np.array(frames_clusterLabels))


    import cPickle as pickle
    pickle_file = open("pickle_data.txt","w")
    pickle.dump({"data":data, "serial_results":serial_results}, pickle_file)
    pickle_file.close()

    print EM.cluster_eigen_vals
    print EM.cluster_eigen_vecs

