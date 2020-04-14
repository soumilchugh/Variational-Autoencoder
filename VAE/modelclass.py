import tensorflow as tf
from layerClass import Layer
tf.compat.v1.disable_v2_behavior()
class Model(Layer,):

    def __init__(self, inputPlaceholder, outputPlaceHolder, LatentPlaceHolder ,isTrainPlaceHolder,learningRate):
        print ("Initialisation")
        self.initializer = "Normal"
        Layer.__init__(self,self.initializer)
        self.input = inputPlaceholder
        self.output = outputPlaceHolder
        self.z = LatentPlaceHolder
        self.learningRate = learningRate
        self._prediction = None
        self._optimize = None
        self.total_loss = None
        self.z_log_sigma_sq = None
        self.z_mu = None
        self.isTrain = isTrainPlaceHolder

    def prediction(self):
        print ("Prediction")
        if not self._prediction:
            self._prediction, self.z_mu, self.z_log_sigma_sq = Layer.runBlock(self, inputData = self.input,isTrain = self.isTrain)
            self.outputImage = Layer.generator(self,inputZ = self.z,isTrain = self.isTrain)
        return self._prediction

    def error(self):
        with tf.name_scope('loss'):
            if not self.total_loss:
                self.marginal_likelihood = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels = self.output, predictions = self._prediction)
                # the latent distribution and N(0, 1)
                self.KL_divergence = -0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1)
                self.ELBO = tf.reduce_mean(self.marginal_likelihood + self.KL_divergence)
                self.total_loss = self.ELBO
                return self.total_loss


    def optimize(self):
        with tf.name_scope('optimiser'):
            if not self._optimizer:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learningRate)
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.compat.v1.control_dependencies(update_ops):
                    self._optimize = optimizer.minimize(self.total_loss)
            return self._optimize
