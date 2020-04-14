from __future__ import division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from layerClass import Layer
from dataClass import Data
from trainClass import train
from pathlib import Path
from freezeGraphClass import freezeGraph
from modelclass import Model
import numpy as np
tf.compat.v1.disable_v2_behavior()
filePath = Path("C:/Users/soumi/Documents/Train-Images-All/").glob('*.jpg')
jsonPath = Path("C:/Users/soumi/Documents/Train-Images-All/trainData.txt")
savePath = "C:/Users/soumi/Documents/VAE-model/model/"
trainbatchSize = 16
valbatchSize = 16
epochs = 101
channelSize = 32
loss = 'ce_kldivergence'
learningRate = 0.0001
outputTensorName =  "Inference/Output"
imageHeight = 240
imageWidth = 320
trainingLossList = list()
validationLossList = list()
data = Data(filePath, jsonPath)
data.jsonData()
trainData = data.loadLabels()
print ("train data shape is", trainData.shape)

def getnumberofBatches(Datasize, batchSize):
    return int(Datasize/batchSize)

gpuInitialised = True
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    gpuInitialised = True
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
'''
if gpuInitialised:
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            tf.constant([imageHeight,imageWidth], dtype="float32",name = "imageSize")
            tf.constant([outputTensorName], name = "OutputTensorName")
            X = tf.compat.v1.placeholder(tf.float32, [None, imageHeight,imageWidth,1], name='Input')
            Y = tf.compat.v1.placeholder(tf.float32, [None, imageHeight,imageWidth,1])
            Z = tf.compat.v1.placeholder(tf.float32, [None, 300], name = 'InputLatentVector')
            isTrain = tf.compat.v1.placeholder(tf.bool, name="isTrain");

            model = Model(X,Y,Z,isTrain,learningRate)
            prediction = model.prediction()
            error = model.error()
            data.add_variable_summary(error, "Loss")
            optimizer = model.optimize()
            #train_dataset, val_dataset, test_dataset = data.createTensorflowDatasets(0.8,0.1,0.1)
            train_dataset, val_dataset = data.createTensorflowDatasets()
            merged_summary_operation = tf.compat.v1.summary.merge_all()
            modelname = "model_" + "learningRate_" + str(model.learningRate) + "epochs_" + str(epochs) + "_channelsize_" + str(channelSize) + loss
            print (modelname)
            train_summary_writer = tf.compat.v1.summary.FileWriter(savePath + modelname + '/tmp/' + modelname + "train")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(savePath + modelname + '/tmp/' + modelname+ "validation")
            init = tf.compat.v1.global_variables_initializer()
            init_l = tf.compat.v1.local_variables_initializer()
            sess.run(init)
            sess.run(init_l)
            print("Initialisation completed")
            flops = tf.compat.v1.profiler.profile(graph,options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
            print('FLOP = ', flops.total_float_ops)
            print("Initialisation completed")
            trainingClass = train(sess,data,optimizer,error,model,merged_summary_operation)
            for epoch in range(epochs):
                batches = getnumberofBatches(data.train_size, trainbatchSize)
                print ("Number of batches", batches)
                trainingError = trainingClass.run(epoch, train_dataset,data.train_size, trainbatchSize,batches,train_summary_writer)
                print ("Training error is ", trainingError)
                print ("Current epoch is",epoch)
                trainingLossList.append(trainingError)
                batches = getnumberofBatches(data.val_size, valbatchSize)
                validationError = trainingClass.validation(epoch,val_dataset,data.val_size, valbatchSize ,batches,validation_summary_writer)
                print ("Validation Error is ", validationError)
                validationLossList.append(validationError)
                if (epoch % 10 == 0):
                    saver = tf.compat.v1.train.Saver()
                    save_path = saver.save(sess, savePath + modelname + "/" + str(epoch) + "/model.ckpt")
                    print("Model saved in path: %s" % save_path)
                    plt.figure()
                    matplotlib.rcParams.update({'font.size':12})
                    plt.plot(trainingLossList, 'r')
                    plt.plot(validationLossList, 'b')
                    plt.xlabel("Number of iterations")
                    plt.ylabel("Error")
                    plt.gca().legend(('training Loss','validation Loss'))
                    plt.savefig(savePath + modelname + "/" + str(epoch) + '/Loss' + '.png')
                    #freezeGraph = freezeGraph(savePath + modelname + "/" + str(epoch) + "/", outputTensorName)
                    #freezeGraph.freeze_graph()
            saver = tf.compat.v1.train.Saver()
            save_path = saver.save(sess, savePath + modelname + "/" + "model.ckpt")
            print("Model saved in path: %s" % save_path)
    #freezeGraph = freezeGraph(savePath + modelname + "/" + str(epoch) + "/", outputTensorName)
    #freezeGraph.freeze_graph()
    plt.figure()
    matplotlib.rcParams.update({'font.size':12})
    plt.plot(trainingLossList, 'r')
    plt.plot(validationLossList, 'b')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.gca().legend(('Training Loss','Validation Loss'))
    plt.savefig(savePath + modelname + "/" + str(epoch) + "/" + "PupilCenterLoss" + ".png")
