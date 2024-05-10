import keras.models
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


class Mnist10x2(nn.Sequential):
    def __init__(self):
        super(Mnist10x2, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class GTSRB10x2(nn.Sequential):
    def __init__(self):
        super(GTSRB10x2, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(144, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class GTSRBCNN(nn.Module):
    def __init__(self):
        super(GTSRBCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# https://github.com/keras-team/keras-io/blob/master/examples/vision/integrated_gradients.py#L186
def get_integrated_gradients(model, img_input, top_pred_idx, baseline=None, num_steps=50):
    """Computes Integrated Gradients for a predicted label.

    Args:
        img_input (ndarray): Original image in 3D
        top_pred_idx: Predicted label for the input image
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image.
    if baseline is None:
        baseline = np.zeros(img_input.shape).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    img_input = img_input.astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)

    # 2. Preprocess the interpolated images
    # from keras.applications import xception
    # interpolated_image = xception.preprocess_input(interpolated_image)

    # 3. Get the gradients
    # grads = []
    # for i, img in enumerate(interpolated_image):
    #     img = tf.expand_dims(img, axis=0)
    #     grad = get_gradients(model, img, top_pred_idx=top_pred_idx)
    #     grads.append(grad[0])
    # grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    grads = get_gradients(model, interpolated_image, top_pred_idx=top_pred_idx)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def get_gradients(model, img_input, top_pred_idx):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads


def save_figure(image, path, cmap=None):
    """
    To plot figures.
    :param image: the image array of shape (width, height, channel)
    :param path: figure name.
    :param cmap: 'gray' if to plot gray scale image.
    :return: an image saved to the designated path.
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


"""

class Mnist10x2(nn.Module):
    def __init__(self):
        super(Mnist10x2, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(28 * 28, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


pytorch_model = Mnist10x2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

keras_model = keras.models.load_model("models/mnist-10x2.h5")
# keras_model.save_weights("models/mnist-10x2-weights.h5")
# pytorch_model.load_state_dict(torch.load("models/mnist-10x2-weights.h5"))
weights = keras_model.get_weights()
pytorch_model.fc1.weight.data = torch.from_numpy(np.transpose(weights[0]))
pytorch_model.fc1.bias.data = torch.from_numpy(weights[1])
pytorch_model.fc2.weight.data = torch.from_numpy(np.transpose(weights[2]))
pytorch_model.fc2.bias.data = torch.from_numpy(weights[3])
pytorch_model.fc3.weight.data = torch.from_numpy(np.transpose(weights[4]))
pytorch_model.fc3.bias.data = torch.from_numpy(weights[5])

torch.save(pytorch_model, "models/mnist-10x2.pt")
pytorch_model = torch.load("models/mnist-10x2.pt")
pytorch_model.eval()



class GTSRBCNN(nn.Module):
    def __init__(self):
        super(GTSRBCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 15 * 15, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class GTSRBCNN(nn.Module):
    def __init__(self):
        super(GTSRBCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196, 20)
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x.permute(0,2,3,1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
        

pytorch_model = GTSRBCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

keras_model = keras.models.load_model("models/gtsrb-cnn.h5")
keras_weights = keras_model.get_weights()

pytorch_model.conv1.weight.data = torch.from_numpy(np.transpose(keras_weights[0], (3,2,0,1)))
pytorch_model.conv1.bias.data = torch.from_numpy(keras_weights[1])
pytorch_model.conv2.weight.data = torch.from_numpy(np.transpose(keras_weights[2], (3,2,0,1)))
pytorch_model.conv2.bias.data = torch.from_numpy(keras_weights[3])
pytorch_model.fc1.weight.data = torch.from_numpy(np.transpose(keras_weights[4]))
pytorch_model.fc1.bias.data = torch.from_numpy(keras_weights[5])
pytorch_model.fc2.weight.data = torch.from_numpy(np.transpose(keras_weights[6]))
pytorch_model.fc2.bias.data = torch.from_numpy(keras_weights[7])


x = x_test[0:1]
keras_model.predict(x)
x = np.moveaxis(x, -1, 1)
x = torch.from_numpy(x)
pytorch_model(x)

torch.save(pytorch_model, "models/gtsrb-cnn.pt")
pytorch_model = torch.load("models/gtsrb-cnn.pt")
pytorch_model.eval()

"""