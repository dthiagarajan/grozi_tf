""" Deep Residual Network
References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
      learning applied to document recognition." Proceedings of the IEEE,
      86(11):2278-2324, November 1998.
Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
"""


# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 1])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 120, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, tensorboard_dir="./logs/", checkpoint_path='model_resnet_grozi',
                    max_checkpoints=10, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=100, validation_set=(testX, testY),
          show_metric=True, batch_size=256, run_id='resnet_grozi')