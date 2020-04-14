import cv2
import numpy as np
import json
import imutils
import glob
import os
from collections import defaultdict
import tensorflow as tf
from pathlib import Path
from dataClass import Data
tf.compat.v1.disable_v2_behavior()
class train(Data):
	def __init__(self,sess,data, optimizer, error, model,merged_summary_operation):
		self.sess = sess
		self.data = data
		self.optimizer = optimizer
		self.error = error
		self.model = model
		self.merged_summary_operation = merged_summary_operation

	def run(self, epoch, dataset,size, sizeOfBatch,numberOfBatches,summary_writer):
		trainingErrorList = list()
		datasetIterator = self.data.createDatasetIterator(dataset,size, sizeOfBatch)
		batch = datasetIterator.get_next()
		self.sess.run(datasetIterator.initializer)
		for i in range(numberOfBatches):
			batch_data = self.sess.run(batch)
			self.data.getBatchData(batch_data)
			#print (np.array(self.data.inputData).shape)
			#print (np.array(self.data.labelData).shape)
			_, loss = self.sess.run([self.optimizer,self.error], feed_dict={self.model.isTrain:True, self.model.input:self.data.inputData, self.model.output: self.data.labelData})
			trainingErrorList.append(loss)
			print (loss)
			merged_summary = self.sess.run(self.merged_summary_operation,feed_dict={self.model.isTrain:False, self.model.input:self.data.inputData, self.model.output: self.data.labelData})
			summary_writer.add_summary(merged_summary,epoch)
		return np.average(trainingErrorList)

	def validation(self, epoch, dataset,size, sizeOfBatch,numberOfBatches,summary_writer):
		trainingErrorList = list()
		datasetIterator = self.data.createDatasetIterator(dataset,size, sizeOfBatch)
		batch = datasetIterator.get_next()
		self.sess.run(datasetIterator.initializer)
		for i in range(numberOfBatches):
			batch_data = self.sess.run(batch)
			self.data.getBatchData(batch_data)
			loss = self.sess.run(self.error, feed_dict={self.model.isTrain:False, self.model.input:self.data.inputData, self.model.output: self.data.labelData})
			trainingErrorList.append((loss))
			merged_summary = self.sess.run(self.merged_summary_operation,feed_dict={self.model.isTrain:False, self.model.input:self.data.inputData, self.model.output: self.data.labelData})
			summary_writer.add_summary(merged_summary,epoch)
		return np.average(trainingErrorList)
