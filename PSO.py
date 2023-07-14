from keras import backend as K
from keras.layers import Layer
import warnings

warnings.filterwarnings('ignore')
"""
压缩函数,使用0.5替代hinton论文中的1,如果是1，所有的向量的范数都将被缩小。
如果是0.5，小于0.5的范数将缩小，大于0.5的将被放大
"""
def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon() #||x||^2
    scale = K.sqrt(s_quared_norm) / (1 + s_quared_norm) #||x||/(0.5+||x||^2)
    result = scale * x
    return result

# 定义我们自己的softmax函数，而不是K.softmax.因为K.softmax不能指定轴
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result

# 定义边缘损失，输入y_true, p_pred，返回分数，传入fit即可，0.5，0.1
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    result = K.sum(y_true * K.square(K.relu(1 - margin -y_pred))
    + lamb * (1-y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    return result

class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)  # Capsule继承**kwargs参数
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)  # 得到激活函数

    # 定义权重
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            # 自定义权重
            self.kernel = self.add_weight( #[row,col,channel]->[1,input_dim_capsule,num_capsule*dim_capsule]
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight( 
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)  # 必须继承Layer的build方法

    # 层的功能逻辑(核心)
    def call(self, inputs):
        if self.share_weights: 
            #inputs: [batch, input_num_capsule, input_dim_capsule]
            #kernel: [1, input_dim_capsule, num_capsule*dim_capsule]
            #hat_inputs: [batch, input_num_capsule, num_capsule*dim_capsule]
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        #hat_inputs: [batch, input_num_capsule, num_capsule, dim_capsule]
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        #hat_inputs: [batch, num_capsule, input_num_capsule, dim_capsule]
        b = K.zeros_like(hat_inputs[:, :, :, 0]) 
        #b: [batch, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings-1:
                b += K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):  # 自动推断shape
        return (None, self.num_capsule, self.dim_capsule)


import warnings
import time
import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from keras.layers import Input, Conv2D, Reshape, Lambda, Dropout, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from CapsuleLayer import Capsule, margin_loss
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from math import sqrt

warnings.filterwarnings('ignore')







# 载入数据
def read_data(path):
    df = pd.read_csv(path, sep=",")
    '''df = df.drop('id', axis=1)'''
    #df = df.sample(frac=0.7, replace=False, random_state=1)#这个效果并不好，不应该分样本量，应该只改变训练集量，测试集不变

    all_but_target = list(set(df.columns.values) - set(["Diagnosis"]))
    X = df[all_but_target]
    y = df[["Diagnosis"]]

    print(pd.get_dummies(y))
    index1 = pd.get_dummies(y).columns
    print('index是', index1)
    return X.values, pd.get_dummies(y).values


# 模型


def MODEL():
    inputs = Input(shape=(5, 5, 1))
    n_class = 7
    print("input", inputs)
    x = Conv2D(36, (2, 2), padding='same', activation='sigmoid')(inputs)
    #x= MaxPooling2D(pool_size=(2,2))(x)
    #x = Dropout(0.1)(x)
    print(x.shape)
    x = Conv2D(86, (2, 2), padding='valid', activation='relu')(x)
    x = Dropout(0.2)(x)
    print(x.shape)
    x = Conv2D(36, (2, 2), padding='valid', activation='relu')(x)
    x = Dropout(0.2)(x)
    print(x.shape)
    # x= MaxPooling(pool_size=(3,3))(x)
    x = Reshape((-1, 36))(x)
    #print('x形状是++++++++++++++++++++++++++++++++++++++++++==', x.shape)
    x = Capsule(num_capsule=n_class, dim_capsule=36, routings=3)(x)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(x)  # 每个胶囊取模长
    model = Model(inputs=inputs, output=output)
    return model

def plot_matrix(y_true, y_pred,title_name):
    cm = confusion_matrix(y_true, y_pred)#混淆矩阵
    #annot = True 格上显示数字 ，fmt：显示数字的格式控制
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
    ax = sns.heatmap(cm, annot=True, fmt='g', xticklabels=['MT', 'LT', 'LED', 'PD',
       'NS', 'HT', 'HED'],yticklabels=['MT', 'LT', 'LED', 'PD',
       'NS', 'HT', 'HED'], cmap="Blues")
    #xticklabels、yticklabels指定横纵轴标签
    plt.tight_layout()
    ax.set_title(title_name) #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()


# 主函数
def main(train_x, train_y, test_x, test_y):

    model = MODEL()
    model.compile(loss= margin_loss, optimizer='adam', metrics=['accuracy'])#cosine_proximity 比 margin_loss 收敛更快，精度更好
    #model.compile(loss='cosine_proximity', optimizer='adamax', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_x, train_y, batch_size=200, nb_epoch=1000, verbose=2)
    pre = model.evaluate(test_x, test_y, batch_size=50, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])
    predict = model.predict(test_x)
    predict = np.argmax(predict, axis=1)
    lb_y = np.argmax(test_y, axis=1)
    # 计算 F1-score
    f1 = f1_score(lb_y, predict, average='weighted')

    # 计算 G-mean
    cm = confusion_matrix(lb_y, predict)
    sens = cm.diagonal() / cm.sum(axis=1)
    g_mean = sqrt(np.prod(sens))

    return predict, lb_y, history, f1, g_mean
    #return predict, lb_y, history


X, y = read_data(r'C:\Users\litong\Desktop\capsnet\vectorin\25.csv')
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
train_x = np.expand_dims(train_x, axis=-1)
train_x = np.expand_dims(train_x, axis=-1)
'''print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)'''

x_train = train_x.reshape(-1, 5, 5, 1)
x_test = test_x.reshape(-1, 5, 5, 1)
#predict, lb_y , history = main(x_train, train_y, x_test, test_y)
start_time = time.time()

predict, lb_y, history, f1, g_mean = main(x_train, train_y, x_test, test_y)
end_time = time.time()
total_time = end_time - start_time
print("Total training time: {:.2f} seconds".format(total_time))

print("F1-score: ", f1)
print("G-mean: ", g_mean)

#f1 = f1_score(lb_y, predict)
#gmean = geometric_mean_score(lb_y, predict)

plt.figure()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('acc'), label='accuracy')#categorical_accuracy
plt.legend()
plt.title('accuracy-loss')
plt.show()
plt.figure()

plot_matrix(lb_y, predict, 'example-confusion matrix')
plt.title('matrix')
plt.show()








