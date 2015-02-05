#-*-coding:utf-8-*-

from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class MovieMaker(object):
    """
    # 始めにキャンバスの数を指定して、addXXX関数でcanvasを順に埋めていく
    # 最後に draw or savefigs or record でデータを出力する

    # 大量のデータを一気にムービーにしようとすると固まる(?)ので、savefigs を使って一旦画像に出力した後でムービーを作ったほうが安全
    """

    def __init__(self, num_canvas, num_canvas_horizontal=3):
        self.canvas_index = 1
        self.num_canvas_x = (num_canvas-1)/num_canvas_horizontal + 1
        self.num_canvas_y = num_canvas_horizontal

        self.figure = plt.figure(figsize=(6 * self.num_canvas_y, 6 * self.num_canvas_x), dpi=100)

        self.updateFunctions = []
        self.num_frameses = []


    def draw(self):
        animations = []
        for updateFunc,num_frames in izip(self.updateFunctions, self.num_frameses):
            animations.append(animation.FuncAnimation(self.figure, updateFunc, num_frames, interval=10, blit=False))
        plt.show()

    def savefigs(self, dirname, label_i):
        import os
        label_dir = dirname + "/%d"%label_i
        os.mkdir(label_dir)
        for time in range(self.num_frameses[0]): #本当は一番短いムービーに合わせたりすべきかも
            for updateFunc in self.updateFunctions:
                updateFunc(time)
            plt.savefig(label_dir + "/%09d"%(time + 1) + ".png") #ffmpegではファイルは1番から
    #        plt.savefig(label_dir + "/%09d"%(time + 1) + ".jpg")


    def record(self, filename, fps, dpi=100, bitrate=720):
        def updateFunctions(time):
            for updateFunc in self.updateFunctions:
                updateFunc(time)

        ani = animation.FuncAnimation(self.figure, updateFunctions, self.num_frameses[0], interval=10, blit=False)
        ani.save(filename,fps=fps, writer="ffmpeg" , dpi=dpi, bitrate=bitrate)

    """
    これ以降は設定した枠にどんなMovie埋めるかという関数
    """
    def addImageMovie(self, images, frames_in_label, embedded_time,  numSamplingFrames=1):
        axes = self.figure.add_subplot(self.num_canvas_x, self.num_canvas_y,self.canvas_index)

        frames_to_record = []
        for begin_frame in frames_in_label:
            for i in range(embedded_time):
                frames_to_record.append(begin_frame + i)

        def updateFunc(time):
            axes.cla()
            axes.imshow(images[frames_to_record[time]])

        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(len(frames_to_record))
        self.canvas_index += 1


    def addPCAMovie(self, frames_in_label, frames_clusterLabelsMat, frames_axes_pcaedDataMat, embedded_time,   numSamplingFrames=1, alpha=0.2):
        axes = self.figure.add_subplot(self.num_canvas_x, self.num_canvas_y,self.canvas_index)
        axes_frames_pcaedDataMat = frames_axes_pcaedDataMat.T
        axes.scatter(axes_frames_pcaedDataMat[0], axes_frames_pcaedDataMat[1],
                     c=frames_clusterLabelsMat, alpha=0.5, marker=".")
        plt.tight_layout() #いらないかも

        def updateFunc(time):
            axes.cla()
            axes.scatter(axes_frames_pcaedDataMat[0],axes_frames_pcaedDataMat[1],
                         c=frames_clusterLabelsMat, alpha=alpha,
                         linewidth='0', marker=".")

            axes.scatter(axes_frames_pcaedDataMat[0][frames_in_label[time/(embedded_time/numSamplingFrames)]],
                         axes_frames_pcaedDataMat[1][frames_in_label[time/(embedded_time/numSamplingFrames)]],
                         c="black", alpha=1, marker=".")

        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(len(frames_in_label) * embedded_time/numSamplingFrames)
        self.canvas_index += 1


    def addSomeSticksMovie(self, frames_jointsxyz_positions, limbs_jointsIndecies, embeddedTime, numSamplingFrames=1):
        axes = self.figure.add_subplot(self.num_canvas_x, self.num_canvas_y,self.canvas_index, projection="3d")
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        plots = []
        title = plt.title("")
        num_data = len(frames_jointsxyz_positions)
        for temp_jointsIndecies in limbs_jointsIndecies:
            jointsIndecies = np.array(temp_jointsIndecies)
            plots.append(axes.plot(
                frames_jointsxyz_positions[0,jointsIndecies*3+0],
                frames_jointsxyz_positions[0,jointsIndecies*3+1],
                frames_jointsxyz_positions[0,jointsIndecies*3+2],
                "-o")[0])
        plt.tight_layout()

        #####
        def updateFunc(time):
            for limb_i, temp_jointsIndecies in enumerate(limbs_jointsIndecies):
                jointsIndecies = np.array(temp_jointsIndecies)
                plots[limb_i].set_xdata(frames_jointsxyz_positions[time, jointsIndecies*3+0])
                plots[limb_i].set_ydata(frames_jointsxyz_positions[time, jointsIndecies*3+1])
                plots[limb_i].set_3d_properties(frames_jointsxyz_positions[time, jointsIndecies*3+2])
            title.set_text("data = %d/%d, time=%d/%d"%(time*numSamplingFrames/embeddedTime, num_data, time%(embeddedTime/numSamplingFrames), embeddedTime/numSamplingFrames))


        self.updateFunctions.append(updateFunc)
        self.num_frameses.append(len(frames_jointsxyz_positions))
        self.canvas_index += 1



def SplitData(data, frames_labels):
    numLabels = max(frames_labels)+1
    labels_frames_embeddedVecYtkm = [[] for i in range(numLabels)]
    labels_frames_embeddedVecXtkm = [[] for i in range(numLabels)]
    labels_frames_embeddedVecXt   = [[] for i in range(numLabels)]
    for frame_i, temp_data in enumerate(data):
        labels_frames_embeddedVecXtkm[frames_labels[frame_i]].append(np.array(temp_data[0]))
        labels_frames_embeddedVecYtkm[frames_labels[frame_i]].append(np.array(temp_data[1]))
        labels_frames_embeddedVecXt[frames_labels[frame_i]].append(np.array(temp_data[2]))
    return labels_frames_embeddedVecYtkm,labels_frames_embeddedVecXtkm,labels_frames_embeddedVecXt



if __name__ == '__main__':

    def get_pcaedData(data, n_components_):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = n_components_)
        inputs_data=[]
        for temp_data in data:
            temp_inputs = []
            for temp in temp_data:
                temp_inputs += np.array(temp.reshape(-1,)).tolist()[0]
            inputs_data.append(temp_inputs)
        return pca.fit_transform(np.array(inputs_data))


    import sys
    argvs = sys.argv
    if len(argvs) != 2:
        print "usage : python %s pickle_filename" % argvs[0]
        exit(-1)
    saveDir =  argvs[1].replace(".txt", "")
    import os
    os.mkdir(saveDir)

    import pickle
    pickle_data = pickle.load(open(argvs[1],"r"))

    labels_joints_data_x = pickle_data["data_for_movie_x"]
    labels_joints_data_y = pickle_data["data_for_movie_y"]

    data = pickle_data["data"]
    pcaed_data = get_pcaedData(data,2)
    serial_results = pickle_data["serial_results"]
    embeddedTime = pickle_data["embeddedTime"]
    numSamplingFrames = pickle_data["numSamplingFrames"]
    numCluster = pickle_data["numCluster"]
    eigen_vals = pickle_data["eigen_vals"]
    eigen_vecs = pickle_data["eigen_vecs"]

    #############

    frames_labels = serial_results[-1]
    labels_frames_embeddedVecXt,labels_frames_embeddedVecXtkm,labels_frames_embeddedVecYtkm = SplitData(data, frames_labels)


    labels_frames = [[] for i in range(numCluster)]
    for i, label in enumerate(serial_results[-1]):
        labels_frames[label].append(i)

    ##
    frames_vals = [eigen_vals[label_i][0] for frame_i,label_i in enumerate(frames_labels)] #それぞれのクラスタの最大の固有値で色付け
    ##

    def makeMovie(label_i):
        joints_data_x = labels_joints_data_x[label_i]
        joints_data_y = labels_joints_data_y[label_i]
        movie = MovieMaker(num_canvas = 3, num_canvas_horizontal = 3)
        movie.addSomeSticksMovie(joints_data_x,
                                 [[0,1,2,3],[3,6,5,4],[2,6]],embeddedTime, numSamplingFrames)
        movie.addSomeSticksMovie(joints_data_y,
                                 [[0,1,2,3],[3,6,5,4],[2,6]],embeddedTime, numSamplingFrames)
#        movie.addPCAMovie(labels_frames[label_i], serial_results[-1], pcaed_data, embeddedTime, numSamplingFrames)
        movie.addPCAMovie(labels_frames[label_i], frames_vals, pcaed_data, embeddedTime, numSamplingFrames)
#        movie.addPCAMovie(labels_frames[label_i], frames_causalities, pcaed_data, embeddedTime, numSamplingFrames)
        movie.savefigs(saveDir, label_i)

    import multiprocessing as mp
    processes = []
    for i in range(numCluster):
        p = mp.Process(target = makeMovie, args=(i,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
