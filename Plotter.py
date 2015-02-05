#!-*-coding:utf-8-*-
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#class Plotter(object):
#    def __init__(self):
#

def plot3dData(data,
               clustering_answer,
               clustering_EM_results,
               clustering_kmeans_results,
               clustering_oldmethod_results):
    plot_colors = ["r","g","b"]
    fig = plt.figure(figsize=(24,24))
    ax_answer = fig.add_subplot(221, projection="3d")
    temp_colors = [plot_colors[label] for label in clustering_answer]
    ax_answer.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
    ax_answer.set_title("answer")

    temp_colors = [plot_colors[label] for label in clustering_EM_results]
    ax_EM_results = fig.add_subplot(222, projection="3d")
    ax_EM_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
    ax_EM_results.set_title("MixtureOfPPCCA")

    temp_colors = [plot_colors[label] for label in clustering_kmeans_results]
    ax_kmeans_results = fig.add_subplot(223, projection="3d")
    ax_kmeans_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
    ax_kmeans_results.set_title("kmeans")

    temp_colors = [plot_colors[label] for label in clustering_oldmethod_results]
    ax_oldmethod_results = fig.add_subplot(224, projection="3d")
    ax_oldmethod_results.scatter(data[:,0], data[:,1], data[:,2], "o", color=temp_colors)
    ax_oldmethod_results.set_title("oldmethod")

    for ii in xrange(0,360,1):
        ax_answer.view_init(elev=ii, azim=ii)
        ax_EM_results.view_init(elev=ii, azim=ii)
        ax_kmeans_results.view_init(elev=ii, azim=ii)
        ax_oldmethod_results.view_init(elev=ii, azim=ii)
        plt.tight_layout()
        plt.savefig("movie/%09d"%ii+".png")

#    plt.show()

def plotErrorChange(correct_cluster_labels, serial_results, filename="./test.png"):
    serial_error_rate = []
    for time_t, results in enumerate(serial_results):
        serial_error_rate.append(np.count_nonzero(results - correct_cluster_labels))
    plt.plot(np.array(serial_error_rate)/float(len(correct_cluster_labels)))
    plt.savefig(filename)



def matchClusterLabels(correct_labels, serial_results_labels, num_clusters):
    correct_data_index_in_cluster = []
    correct_data_index_in_cluster = [[] for i in range(num_clusters)]
    for i, correct_label in enumerate(correct_labels):
        correct_data_index_in_cluster[correct_label].append(i)

    #最後の収束結果からクラスタ番号らしきものを逆算する
    cluster_index = []
    for cluster_i in range(num_clusters):
        result_need_in_a_cluster =  np.array(serial_results_labels[-1])[correct_data_index_in_cluster[cluster_i]]
        cluster_index.append(int(round(np.average(result_need_in_a_cluster))))
    correct_labels = [cluster_index[label] for label in correct_labels] #正解ラベルの順番をずらす 2->1 1->0 0->2みたいな
    return correct_labels, serial_results_labels


#        serial_error_rate = []
#        for time_t, results in enumerate(serial_results):
#            serial_error_rate.append(np.count_nonzero(results - correct_cluster_labels))
