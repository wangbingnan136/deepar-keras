import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.compat.v1.set_random_seed(1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


from tensorflow.keras.regularizers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
import tensorflow_addons as tfa
import numpy as np
import gc
import tensorflow as tf

def smape_error(y_true, y_pred):
    return K.mean(K.clip(K.abs(y_pred - y_true),  0.0, 1.0), axis=-1)

    


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))



df = pd.read_csv('train_2.csv.zip')
# train_2.csv.zip is from kaggle web traffic ,you can download here https://www.kaggle.com/c/web-traffic-time-series-forecasting
train=df

data_start_date = df.columns[1]
data_end_date = df.columns[-1]
print('Data ranges from %s to %s' % (data_start_date, data_end_date))

reduce_mem_usage(train)

X_train=train[train.columns[1:train.shape[1]-64]]
y_train=train[train.columns[-64:]]

X_train=np.log1p(X_train)
y_train=np.log1p(y_train)


mins=X_train.min(axis=1)
maxs=X_train.max(axis=1)
print(mins.isnull().sum())
print(maxs.isnull().sum())

for col in X_train.columns:
  X_train[col]=(X_train[col]-mins)/((maxs-mins)+1)


for col in y_train.columns:
  y_train[col]=(y_train[col]-mins)/((maxs-mins)+1)

X_train=X_train.fillna(0).values.reshape(-1,739,1)
y_train.fillna(0,inplace=True)

X_test=X_train[116050:]
X_train=X_train[0:116050]

y_test=y_train[116050:]
y_train=y_train[0:116050]
y_train=y_train.values.astype(np.float32)
y_test=y_test.values.astype(np.float32)



X_train=X_train.reshape(-1,739,1).astype(np.float32)
X_test=X_test.reshape(-1,739,1).astype(np.float32)

print(X_train.dtype)
print(X_test.dtype)
print(y_train.dtype)
print(y_test.dtype)



from tensorflow import keras
def create_rnn_cells(rnn_units):
    cells = []
    for units in rnn_units:
        cells.append(keras.layers.LSTMCell(units))
    return cells

class GaussianLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.W_mu = None
        self.b_mu = None
        self.W_sigma = None
        self.b_sigma = None

    def build(self, input_shape):
        super(GaussianLayer, self).build(input_shape)
        
        dim = input_shape[-1]
        
        self.W_mu = self.add_weight(
            name='W_mu', 
            shape=(dim, 1), 
            initializer='glorot_normal', 
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name='b_mu', 
            shape=(1,), 
            initializer='zeros', 
            trainable=True,
        )
        
        self.W_sigma = self.add_weight(
            name='W_sigma',
            shape=(dim, 1),
            initializer='glorot_normal',
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(1,),
            initializer='zeros', 
            trainable=True,
        )        
        
    def call(self, inputs):
        mu = K.dot(inputs, self.W_mu)
        mu = K.bias_add(mu, self.b_mu, data_format='channels_last')
        
        sigma = K.dot(inputs, self.W_sigma)
        sigma = K.bias_add(sigma, self.b_sigma, data_format='channels_last')
        sigma = K.softplus(sigma) + K.epsilon()
        
        return tf.squeeze(mu, axis=-1), tf.squeeze(sigma, axis=-1) 
    

def gaussian_loss(y_true, mu, sigma):
    loss = (
         tf.math.log(sigma) 
        + 0.5 * tf.math.log(2 * np.pi) 
        + 0.5 * tf.square(tf.math.divide(y_true - mu, sigma))
    )
    return tf.reduce_mean(loss**2) # to avoid the loss became negative,it is quite uncomfortable to see that

    
def gaussian_sample(mu, sigma): 
    mu = tf.expand_dims(mu, axis=-1)
    sigma = tf.expand_dims(sigma, axis=-1)
    
    samples = tf.random.normal((1000,), mean=mu, stddev=sigma) ## 这里也可以使用tensorflow probability
    return tf.reduce_mean(samples, axis=-1)

    
class DeepAR(keras.Model): 
    def __init__(self, fw=62, rnn_units=[256]): 
      # fw is the number of itme steps you want to predict
      # rnn units is a list, for example, [256,256,256] means you use 3 layers of LSTM,each LSTM has a 256 units
        super(DeepAR, self).__init__()
        
        self.encoder_layer = keras.layers.RNN(create_rnn_cells(rnn_units))
        self.repeat_layer = keras.layers.RepeatVector(fw)
        self.decoder_layer = keras.layers.RNN(create_rnn_cells(rnn_units), return_sequences=True)
        self.gaussian_layer = GaussianLayer()

    def call(self, inputs): 
        outputs = self.encoder_layer(inputs)
        outputs = self.repeat_layer(outputs)
        outputs = self.decoder_layer(outputs)
        return self.gaussian_layer(outputs)

    def train_step(self, inputs):
        x, y = inputs
        
        with tf.GradientTape() as tape: 
            mu, sigma = self(x)
            loss_val = self.loss(y, mu, sigma)

        grads = tape.gradient(loss_val, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {'loss': loss_val}

    def test_step(self, inputs):
        x, y = inputs
        mu, sigma = self(x, training=False)
        c=[mu,sigma]

        #loss_val = self.loss(y, mu, sigma)
        y_pred=gaussian_sample(mu, sigma)
        return {'loss': smape_error(y,y_pred)}
    
    def predict_step(self, inputs): 
        x = inputs
        mu, sigma = self(x, training=False)
        return gaussian_sample(mu, sigma)



model=DeepAR()

model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=3e-4),loss=gaussian_loss)


model.fit(X_train,y_train,batch_size=512,epochs=500, \
          validation_data=((X_test,y_test)),shuffle=False,workers=4,use_multiprocessing=True)  



