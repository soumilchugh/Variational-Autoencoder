import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
class Layer:

    def __init__(self,initializer):
        self.weightsDict = {}
        self.biasDict = {}
        #if (initializer == "Xavier"):
        #    self.tf_initializer = tf.initializers.GlorotUniform()
        if (initializer == "Normal"):
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "He"):
            self.tf_initializer = tf.keras_initializers.he_normal(dtype=tf.float32)


    def conv2d(self,inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
        filter_shape = [filterSize, filterSize, inputSize, outputSize]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape,initializer=self.tf_initializer)
            self.biasDict[self.biasName] = tf.compat.v1.get_variable(self.biasName, shape = outputSize, initializer=self.tf_initializer)
        convOutput = tf.compat.v1.nn.conv2d(input = inputFeature, filter = self.weightsDict[self.weightName], strides=[1, strides, strides, 1], padding='SAME', name = name)
        finalOutput = tf.compat.v1.nn.bias_add(convOutput, self.biasDict[self.biasName])
        return finalOutput

    def avgpool2d(self,inputData):
        return tf.compat.v1.nn.avg_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    def downSamplingBlock(self,x,isTrain,input_channels,output_channels,down_size,name):
        if down_size != None:
            x = self.avgpool2d(x)
        print (x.shape)
        self.conv1 =  self.conv2d(x,3,input_channels,output_channels,name+"conv1")
        #print (self.conv1.shape)
        #x1 = tf.compat.v1.nn.leaky_relu(self.conv1)
        #x21 = tf.concat([x,x1],axis=3)
        #print (x21.shape)
        #self.conv21 = self.conv2d(x21,3, input_channels+output_channels,output_channels,name+"conv21")
        #self.conv22 = self.conv2d(self.conv21,3,output_channels,output_channels,name+"conv22")
        #print (self.conv22.shape)
        #x22 = tf.compat.v1.nn.leaky_relu(self.conv22)
        #x31 = tf.concat([x21,x22],axis=3)
        #print (x31.shape)
        #self.conv31 = self.conv2d(x31, 1, input_channels+2*output_channels,output_channels,name+"conv31")
        #self.conv32 = self.conv2d(self.conv31,3,output_channels,output_channels,name+"conv32")
        #print (self.conv32.shape)
        #out = tf.compat.v1.layers.batch_normalization(self.conv1,training=isTrain)
        return tf.compat.v1.nn.leaky_relu(self.conv1)

    def upSamplingBlock(self,currentInput,previousInput,isTrain,skip_channels,input_channels,output_channels,image_width, image_height,name):
        print ("Upsampling")
        x = tf.compat.v1.image.resize_images(images=currentInput,size=[image_width,image_height], method=tf.image.ResizeMethod.BILINEAR)
        print (x.shape)
        x_concat = tf.concat([x,previousInput],axis=3)
        print (x_concat.shape)
        self.conv11 = self.conv2d(x_concat, 3,skip_channels+input_channels,output_channels,name + "conv11")
        print (self.conv11.shape)
        #self.conv12 = self.conv2d(self.conv11,3,output_channels,output_channels,name + "conv12")
        #print (self.conv12.shape)
        #x1 = tf.compat.v1.nn.leaky_relu(self.conv12)
        #x21 = tf.concat([x,x1],axis=3)
        #print (x21.shape)
        #self.conv41 = self.conv2d(x21,1,skip_channels+input_channels,output_channels,name + "conv41")
        #print (self.conv41.shape)
        #self.conv42 = self.conv2d(self.conv41, 3, output_channels,output_channels,name + "conv42")
        #print (self.conv42.shape)
        #batchNorm1 = tf.compat.v1.layers.batch_normalization(self.conv11, training=isTrain)
        out = tf.compat.v1.nn.leaky_relu(self.conv11)

        return out

    def fullyConnected(self,currentInput, inputNeurons, outputNeurons, name):
        filter_shape = [inputNeurons, outputNeurons]
        print ("Filter shape is", filter_shape)
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape,initializer=self.tf_initializer)
            self.biasDict[self.biasName] = tf.compat.v1.get_variable(self.biasName, shape = outputNeurons, initializer=self.tf_initializer)
        fc1 = tf.add(tf.matmul(currentInput, self.weightsDict[self.weightName]),  self.biasDict[self.biasName], name = name + "FC")
        return fc1

    def generator(self, inputZ, isTrain, out_channels=1,channel_size=32):
        self.fc1 = self.fullyConnected(inputZ, 300, 9600, name = 'fc')
        self.fc1Activation = tf.nn.relu(self.fc1,name ='relu')
        self.x11 = tf.reshape(self.fc1Activation,[-1,15,20,32], name = "Reshaped")
        self.x6 = self.upSamplingBlock(self.x11, self.x4, isTrain, skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
        self.x7 = self.upSamplingBlock(self.x6, self.x3,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
        self.x8 = self.upSamplingBlock(self.x7, self.x2,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
        self.x9 = self.upSamplingBlock(self.x8, self.x1, isTrain, skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 240,image_height = 320, name = "UpBlock4")
        out_conv1 = self.conv2d(self.x9,1,channel_size,out_channels,name = "Output")
        self.out_conv1 = tf.clip_by_value(out_conv1, 1e-8, 1 - 1e-8)
        self.output = tf.reshape(self.out_conv1,[-1,240,320,1], name = "Inference/Output")
        return self.output

    def runBlock(self,inputData,isTrain,in_channels=1,out_channels=1,channel_size=32):
        self.x1 = self.downSamplingBlock(inputData,isTrain, input_channels=in_channels,output_channels=channel_size, down_size=None,name = "DownBlock1")
        self.x2 = self.downSamplingBlock(self.x1, isTrain, input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock2")
        self.x3 = self.downSamplingBlock(self.x2,isTrain, input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock3")
        self.x4 = self.downSamplingBlock(self.x3,isTrain, input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock4")
        self.x5 = self.downSamplingBlock(self.x4,isTrain ,input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock5")
        print (self.x5.shape)
        self.flatten = tf.compat.v1.layers.flatten(self.x5)
        print ("Flatten shape is", self.flatten.shape)
        self.z_mu = (self.fullyConnected(self.flatten,9600, 300, name='enc_fc4_mu'))
        z_log_sigma_sq = (self.fullyConnected(self.flatten,9600, 300, name='enc_fc4_sigma'))
        self.z_log_sigma_sq = 1e-6 + tf.nn.softplus(z_log_sigma_sq)
        eps = tf.compat.v1.random_normal(shape=tf.shape(self.z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
        #self.z = tf.add(self.z_mu , tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps, name = "Latent")
        self.z = tf.add(self.z_mu, self.z_log_sigma_sq*eps, name = "Latent")
        self.fc1 = self.fullyConnected(self.z, 300, 9600, name = 'fc')
        #self.batchNorm1 = tf.compat.v1.layers.batch_normalization(self.fc1, training=isTrain)
        self.fc1Activation = tf.nn.relu(self.fc1,name ='relu')
        self.x11 = tf.reshape(self.fc1Activation,[-1,15,20,32], name = "Reshaped")
        self.x6 = self.upSamplingBlock(self.x11, self.x4,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
        self.x7 = self.upSamplingBlock(self.x6, self.x3,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
        self.x8 = self.upSamplingBlock(self.x7, self.x2,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
        self.x9 = self.upSamplingBlock(self.x8, self.x1,isTrain,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 240,image_height = 320, name = "UpBlock4")
        out_conv1 = self.conv2d(self.x9,1,channel_size,out_channels,name = "Output")
        out_conv2 = tf.clip_by_value(out_conv1, 1e-8, 1 - 1e-8)
        self.out_conv1 = tf.sigmoid(out_conv2)
        self.output = tf.reshape(self.out_conv1,[-1,240,320,1], name = "Inference/Output")
        return self.out_conv1, self.z_mu, self.z_log_sigma_sq
