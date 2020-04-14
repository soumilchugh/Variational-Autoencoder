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
                epsilon = 1e-10
                #recon_loss = -tf.reduce_sum(self.output * tf.log(epsilon+self._prediction) + (1-self.output) * tf.log(epsilon+1-self._prediction), axis=1)
                #self.recon_loss = tf.reduce_mean(recon_loss)
                #self.marginal_likelihood = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.output, logits = self._prediction))
                self.marginal_likelihood = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels = self.output, predictions = self._prediction))
                # Latent loss
                # KL divergence: measure the difference between two distributions
                # Here we measure the divergence between
                # the latent distribution and N(0, 1)
                self.KL_divergence = -0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1)
                #kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
                #KL_divergence = -0.5 * tf.reduce_sum(kl_div_loss, 1)
                #self.latent_loss = tf.reduce_mean(latent_loss)

                #self.total_loss = self.recon_loss + self.latent_loss
                #marginal_likelihood = tf.reduce_sum(self.output * tf.math.log(tf.sigmoid(self._prediction)) + (1 - self.output) * tf.math.log(1 - tf.sigmoid(self.prediction)),[1, 2, 3])
                #KL_divergence = -0.5 * tf.reduce_sum(-tf.square(self.z_mu) - tf.square(self.z_log_sigma_sq) + tf.math.log(1e-8 + tf.square(self.z_log_sigma_sq)) + 1)
                #self.marginal_likelihood = tf.reduce_mean(marginal_likelihood)
                #self.KL_divergence = tf.reduce_mean(KL_divergence)
                self.ELBO = tf.reduce_mean(self.marginal_likelihood + self.KL_divergence)
                self.total_loss = self.ELBO
                return self.total_loss


    def optimize(self):
        with tf.name_scope('optimiser'):
            if not self._optimize:
                #optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learningRate, momentum = 0.9)
                #optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate = self.learningRate, momentum = 0.9, use_nesterov=False)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learningRate)
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.compat.v1.control_dependencies(update_ops):
                    self._optimize = optimizer.minimize(self.total_loss)
                #self.train_op = tf.group([self._optimize, update_ops])
            return self._optimize
