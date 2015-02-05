#! -*-coding:utf-8-*-

from CausalityCalculator import CausalityCalculator
from random import shuffle
import numpy as np

class SurrogatingTester(object):
    def __init__(self,num_try, num_labels, labels_X_t, labels_X_t_k_m, labels_Y_t_k_m):
#        self.num_try = num_try #TODO rename

        self.labels_surrogateaverage = []
        self.labels_surrogatediv = []


        for label_i in range(num_labels):
            temp_causalities = []
            for i in range(num_try):
                shuffled_X_t = labels_X_t[label_i][:]
                shuffle(shuffled_X_t)

                shuffled_X_t_k_m = labels_X_t_k_m[label_i][:]
                shuffle(shuffled_X_t_k_m)

                shuffled_Y_t_k_m= labels_Y_t_k_m[label_i][:]
                shuffle(shuffled_Y_t_k_m)

                causality_calculatorYtoX = CausalityCalculator(shuffled_X_t, shuffled_X_t_k_m, shuffled_Y_t_k_m)
                temp_causalities.append(causality_calculatorYtoX.calcGrangerCausality()) #TODO TODO False が0として扱われているのでキケン

            self.labels_surrogateaverage.append(np.average(temp_causalities))
            self.labels_surrogatediv.append(np.std(temp_causalities))

    def get(self): #TODO rename
        return self.labels_surrogateaverage, self.labels_surrogatediv

    def compare(self, labels_causalities):
        labels_comparedCausalities = []
        for label_i in range(len(labels_causalities)):
            labels_comparedCausalities.append((labels_causalities[label_i] - self.labels_surrogateaverage[label_i]) / self.labels_surrogatediv[label_i])

        return labels_comparedCausalities
