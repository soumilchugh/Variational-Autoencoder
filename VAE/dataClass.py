import cv2
import numpy as np
import json
import imutils
import glob
import os
from collections import defaultdict
import tensorflow as tf
from pathlib import Path
import random
from scipy.stats import truncnorm
tf.compat.v1.disable_v2_behavior()
class Data(object):
	def __init__(self,filePath,jsonPath):
		self.filePath = filePath
		self.jsonPath = jsonPath
		self.trainPath = list()
		self.valPath = list()
		self.Dictdata = defaultdict(dict)
		self.train_dataset = None
		self.val_dataset = None
		self.train_size = None
		self.val_size = None
		self.inputData = list()
		self.labelData = list()

	def jsonData(self):
		with open(str(self.jsonPath), 'r') as f:
			self.Dictdata =  json.loads(f.read())

	def loadLabels(self):
		files = [file for file in self.filePath]
		testName = ["rose","tekken","eizenman"]
		validationName = ["fitsum","rose"]
		test = list()
		for file in files:
			filename, file_extension = os.path.splitext(str(file))
			name = str(os.path.basename(filename));
			my_path = file.absolute().as_posix()
			isValidation = False
			isTest = False
			if (name + ".jpg") in self.Dictdata:
				#if 'IrisBoundaryPoints' in self.Dictdata[name + ".jpg"]:
				for s in validationName:
					if s in name:
						self.valPath.append(str(my_path))
						isValidation = True
						break
				for s in testName:
					if s in name:
						isTest = True
						test.append(str(my_path))
						break

				if (isValidation == False) and (isTest == False):
					self.trainPath.append(str(my_path))
		print ("Test shape is", len(test))
		return np.array(self.trainPath)

	def createTensorflowDatasets(self):
		self.train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(self.trainPath)
		self.val_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(self.valPath)
		self.trainPath = np.array(self.trainPath)
		self.valPath = np.array(self.valPath)
		self.train_size = np.array(self.trainPath.shape[0])
		self.val_size = np.array(self.valPath.shape[0])
		print ("Training data size is ", self.train_size)
		print ("Validation data size is", self.val_size)
		return self.train_dataset, self.val_dataset

	def loadDatasetinTensorflow(self, dataset):
		#dataset = dataset.map(parse_function, num_parallel_calls=4)
		#dataset = dataset.map(lambda filename:tf.py_function(self.preprocess, [filename], [tf.float32,tf.float32]))
		print(dataset)
		return dataset

	def createDatasetIterator(self,dataset, datasetSize, batchSize):
		dataset = dataset.shuffle(datasetSize)
		dataset = dataset.batch(batchSize)
		dataset = dataset.prefetch(1)
		datasetIterator = dataset.make_initializable_iterator()
		return datasetIterator

	def flipImage(self,image):
		isFlipped = False
		if np.random.rand(1) < 0.5:
			image = cv2.flip(image,1)
			isFlipped = True
		return image,isFlipped

	def dataAugmentation(self,image):
		if (np.random.rand(1) < 0.2):
			sigma_value=np.random.randint(2, 7)
			image = cv2.GaussianBlur(image,(5,5),sigma_value)
		if (np.random.rand(1) < 0.2):
			kernel_v = np.zeros((5, 5))
			kernel_v[:, int((5 - 1)/2)] = np.ones(5)
			kernel_v /= 5
			image = cv2.filter2D(image, -1, kernel_v)
		return image

	def get_truncated_normal(self,mean=0, sd=1, low=0, upp=10):
		return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

	def randomNegCircle(self,image):
		num_reflections = np.random.randint(1, 5)
		counter = 0
		for i in np.arange(0, 100):
			if counter >= num_reflections:
				break
			mask = np.zeros([240,320]).astype(np.uint8)
			x1 = random.randint(0,320)
			y1 = random.randint(0,240)
			gauss_dist = self.get_truncated_normal(mean=200, sd=55, low=128, upp=255)
			size = np.random.randint(1, 5)
			cv2.circle(mask,(x1,y1),size,(255,255,255),-1) # Red
			mask_ind = np.argwhere(mask.astype(np.float)== 255)
			g = gauss_dist.rvs(len(mask_ind))
			image[mask_ind[:,0],mask_ind[:,1]] = g
			counter = counter + 1
		return image

	def randomCircle(self,image,pupilX, pupilY):
		num_reflections = np.random.randint(1, 10)
		counter = 0
		for i in np.arange(0, 100):
			if counter >= num_reflections:
				break
			mask = np.zeros([240,320]).astype(np.uint8)
			#x1 = random.randint(pupilX-50,pupilX + 50)
			#y1 = random.randint(pupilY-50,pupilY + 50)
			x1 = random.randint(0,320)
			y1 = random.randint(0,240)
			size = np.random.randint(1, 5)
			gauss_dist = self.get_truncated_normal(mean=200, sd=55, low=128, upp=255)
			cv2.circle(mask,(x1,y1),size,(255,255,255),-1) # Red
			mask_ind = np.argwhere(mask.astype(np.float)== 255)
			g = gauss_dist.rvs(len(mask_ind))
			image[mask_ind[:,0],mask_ind[:,1]] = g
			counter = counter + 1
		return image

	def getBatchData(self,batch):
		self.inputData = list()
		self.labelData = list()
		clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
		for image in (batch):
			newIrisBoundaryPoints = []
			image_reader = cv2.imread(image.decode("utf-8"),0)
			#clahe_image = image_reader
			#if (np.random.rand(1) < 0.5):
			#	clahe_image = clahe.apply(image_reader)
			#else:
			#	clahe_image = image_reader
			filename, file_extension = os.path.splitext(image.decode("utf-8"))
			name = str(os.path.basename(filename));
			#pupilX = int((self.Dictdata[name + ".jpg"]['PupilCenter']['PupilX'])*320)
			#pupilY = int((self.Dictdata[name + ".jpg"]['PupilCenter']['PupilY'])*240)
			'''
			if "pos" in name:
				if (np.random.rand(1) < 0.5):
					finalImage = self.randomCircle(clahe_image,pupilX,pupilY)
				else:
					finalImage = clahe_image
			else:
				if (np.random.rand(1) < 0.5):
					finalImage = self.randomNegCircle(clahe_image)
				else:
					finalImage = clahe_image
			'''
			'''
			irisBoundaryPoints = self.Dictdata[name + ".jpg"]['IrisBoundaryPoints']
			for item in irisBoundaryPoints:
				newIrisBoundaryPoints.append((int(item[0]*320),int(item[1]*240)))

			mask = np.zeros((240, 320))
			if len(newIrisBoundaryPoints) > 0:
				e = cv2.fitEllipse(np.array(newIrisBoundaryPoints))
				newcenterX = int(e[0][0])
				newcenterY = int(e[0][1])
				major_x = int((e[1][0])/2)
				major_y = int((e[1][1])/2)
				angle = int(e[2])
				cv2.ellipse(mask, (newcenterX,newcenterY), (major_x, major_y), angle, 0, 360, (255,255,255), -1)
				mask = mask.astype(np.float)/255.0
			'''
			mask = np.expand_dims(image_reader.astype(np.float)/255.0, axis=2)
			self.labelData.append((mask))
			self.inputData.append(mask)
			#finalImage = self.dataAugmentation(finalImage)
			#normalised_image = finalImage.astype(np.float)/255.0
			#normalised_image = np.expand_dims(normalised_image, axis=2)
			#self.inputData.append(normalised_image.astype(np.float))


	def add_variable_summary(self,tf_variable, summary_name):
		with tf.compat.v1.name_scope(summary_name + '_summary'):
			mean = tf.reduce_mean(tf_variable)
			tf.compat.v1.summary.scalar('Mean',mean)
			with tf.compat.v1.name_scope('standard_deviation'):
				standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
			tf.compat.v1.summary.scalar('StandardDeviation',standard_deviation)
			tf.compat.v1.summary.scalar('Maximum', tf.reduce_max(tf_variable))
			tf.compat.v1.summary.scalar('Minimum',tf.reduce_min(tf_variable))
			tf.compat.v1.summary.histogram('Histogram',tf_variable)
