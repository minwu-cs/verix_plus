import random
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from keras.models import load_model
# from tensorflow.image import rgb_to_grayscale
from skimage.color import label2rgb
from bound_propagation import BoundModelFactory, HyperRectangle
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import sys
sys.path.insert(0, "Marabou")
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou
from utils import *


class VeriX:
    """
    This is the VeriX class to take in an image and a neural network, and then output an explanation.
    """
    image = None
    keras_model = None
    mara_model = None
    traverse: str
    sensitivity = None
    dataset: str
    label: int
    inputVars = None
    outputVars = None
    epsilon: float
    directory: str
    unsat_set = []
    sat_set = []
    timeout_set = []
    marabou_time = None
    """
    Marabou options: 'timeoutInSeconds' is the timeout parameter. 
    """
    options = Marabou.createOptions(numWorkers=16,
                                    timeoutInSeconds=300,
                                    verbosity=0,
                                    solveWithMILP=True)

    def __init__(self,
                 dataset,
                 image,
                 model_path,
                 directory,
                 plot_original=True):
        """
        To initialize the VeriX class.
        :param dataset: 'MNIST' or 'GTSRB'.
        :param image: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        """
        self.dataset = dataset
        self.image = image
        self.directory = directory
        """
        Load the onnx model.
        """
        self.onnx_model = onnx.load(model_path + ".onnx")
        self.onnx_session = ort.InferenceSession(model_path + ".onnx")
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(image, axis=0)})
        # prediction = np.asarray(prediction[0])
        prediction = prediction[0][0]
        self.label = prediction.argmax()
        self.logit_rank = prediction.argsort()[::-1]
        """
        Load the onnx model into Marabou.
        Note: to ensure sound and complete analysis, load the model before the softmax activation function;
        if the model is trained from logits directly, then load the whole model. 
        """
        self.mara_model = Marabou.read_onnx(model_path + ".onnx")
        # if self.onnx_model.graph.node[-1].op_type == "Softmax":
        #     mara_model_output = self.onnx_model.graph.node[-1].input
        # else:
        #     mara_model_output = None
        # self.mara_model = Marabou.read_onnx(filename=model_path + ".onnx",
        #                                     outputNames=mara_model_output)
        self.inputVars = np.arange(image.shape[0] * image.shape[1])
        self.outputVars = self.mara_model.outputVars[0].flatten()
        """
        Load the keras model.
        """
        from keras.models import Sequential
        import tensorflow_ranking as tfr
        self.keras_model = Sequential()
        self.keras_model.compile(loss=tfr.keras.losses.SoftmaxLoss(),
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=['accuracy'])
        self.keras_model = load_model(model_path + ".h5")
        """
        Load the pytorch model.
        """
        self.torch_model = torch.load(model_path + ".pt")
        self.torch_model.eval()

        if plot_original:
            self.plot_original()


    def new_traversal(self,
                      epsilon,
                      traverse="bounds",
                      plot_sensitivity=True):
        self.traverse = traverse
        pixels = self.inputVars

        output_lower = []

        if self.traverse == "bounds":
            if self.keras_model.name.__contains__("10x2"):
                width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
                image = torch.from_numpy(self.image.flatten())
                lower = image.repeat(width * height, 1)
                upper = image.repeat(width * height, 1)
                if self.dataset == 'mnist':
                    for i in np.arange(width * height):
                        lower[i, i] = max(0, image[i] - epsilon)
                        upper[i, i] = min(1, image[i] + epsilon)
                elif self.dataset == 'gtsrb' or self.dataset == 'cifar10':
                    for i in np.arange(width * height):
                        lower[i, 3 * i] = max(0, image[3 * i] - epsilon)
                        upper[i, 3 * i] = min(1, image[3 * i] + epsilon)
                        lower[i, 3 * i + 1] = max(0, image[3 * i + 1] - epsilon)
                        upper[i, 3 * i + 1] = min(1, image[3 * i + 1] + epsilon)
                        lower[i, 3 * i + 2] = max(0, image[3 * i + 2] - epsilon)
                        upper[i, 3 * i + 2] = min(1, image[3 * i + 2] + epsilon)

                factory = BoundModelFactory()
                net = factory.build(self.torch_model)
                input_bounds = HyperRectangle(lower, upper)
                ibp_bounds = net.ibp(input_bounds)
                output_lower = ibp_bounds.lower[:, self.label].detach().numpy()
                sorted_index = output_lower.argsort()[::-1]
                self.inputVars = sorted_index
                self.sensitivity = output_lower.reshape(width, height)

            elif self.keras_model.name.__contains__("cnn"):
                # elif self.traverse == "auto_LiRPA":
                width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
                image = self.image.flatten()
                images = np.array([image] * width * height)
                lower = images.copy()
                upper = images.copy()
                # lower = image.repeat(width * height, 1)
                # upper = image.repeat(width * height, 1)
                if self.dataset == 'mnist':
                    for i in np.arange(width * height):
                        lower[i, i] = max(0, image[i] - epsilon)
                        upper[i, i] = min(1, image[i] + epsilon)
                elif self.dataset == 'gtsrb' or self.dataset == 'cifar10':
                    for i in np.arange(width * height):
                        lower[i, 3 * i] = max(0, image[3 * i] - epsilon)
                        upper[i, 3 * i] = min(1, image[3 * i] + epsilon)
                        lower[i, 3 * i + 1] = max(0, image[3 * i + 1] - epsilon)
                        upper[i, 3 * i + 1] = min(1, image[3 * i + 1] + epsilon)
                        lower[i, 3 * i + 2] = max(0, image[3 * i + 2] - epsilon)
                        upper[i, 3 * i + 2] = min(1, image[3 * i + 2] + epsilon)
                images = images.reshape(width * height, width, height, channel)
                images = torch.from_numpy(np.moveaxis(images, -1, 1))
                lower = lower.reshape(width * height, width, height, channel)
                lower = torch.from_numpy(np.moveaxis(lower, -1, 1))
                upper = upper.reshape(width * height, width, height, channel)
                upper = torch.from_numpy(np.moveaxis(upper, -1, 1))

                b_model = BoundedModule(self.torch_model, images)
                ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
                b_x = BoundedTensor(images, ptb)
                # lb, up = b_model.compute_bounds(x=b_x, method='IBP')
                lb, up = b_model.compute_bounds(x=b_x, method='alpha-CROWN')

                output_lower = lb[:, self.label].detach().numpy()
                sorted_index = output_lower.argsort()[::-1]
                self.inputVars = sorted_index
                self.sensitivity = output_lower.reshape(width, height)
            else:
                print('Need to indicate model structure in model name.')
        else:
            print("Unsupported traversal.")

        if plot_sensitivity:
            save_figure(image=self.sensitivity,
                        path=self.directory + f"sensitivity-{self.traverse}.png")

    def traversal_order(self,
                        traverse="heuristic",
                        plot_sensitivity=True,
                        seed=0):
        """
        To compute the traversal order of checking all the pixels in the image.
        :param traverse: 'heuristic' (by default) or 'random'.
        :param plot_sensitivity: if True, plot the sensitivity map.
        :param seed: if traverse by 'random', then set a random seed.
        :return: an updated inputVars that contains the traversal order.
        """
        self.traverse = traverse
        if self.traverse == "heuristic":
            width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
            temp = self.image.reshape(width * height, channel)
            image_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
            image_batch_manip = image_batch.copy()
            for i in range(width * height):
                """
                Different ways to compute sensitivity: use pixel reversal for MNIST and deletion for GTSRB.
                """
                if self.dataset == "mnist":
                    image_batch_manip[i][i][:] = 1 - image_batch_manip[i][i][:]
                elif self.dataset == "gtsrb" or self.dataset == "cifar10":
                    image_batch_manip[i][i][:] = 0
                else:
                    print("Dataset not supported: try 'mnist' or 'gtsrb'.")
            image_batch = image_batch.reshape((width * height, width, height, channel))
            predictions = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch})
            predictions = np.asarray(predictions[0])
            image_batch_manip = image_batch_manip.reshape((width * height, width, height, channel))
            predictions_manip = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch_manip})
            predictions_manip = np.asarray(predictions_manip[0])
            difference = predictions - predictions_manip
            features = difference[:, self.label]
            sorted_index = features.argsort()
            self.inputVars = sorted_index
            self.sensitivity = features.reshape(width, height)
            if plot_sensitivity:
                save_figure(image=self.sensitivity,
                            path=self.directory + f"sensitivity-{self.traverse}.png")

        elif self.traverse == "random":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def deploy_binary(self, epsilon):
        self.epsilon = epsilon
        x = self.inputVars
        self.binary_check_robust(x)
        self.plot_explanation()
        self.record_statistics()

    def binary_check_robust(self, x):
        if len(x) == 1:
            if self.check_robust(pixels=np.concatenate([self.unsat_set, x])):
                self.unsat_set.extend(x)
                return
            else:
                self.sat_set.extend(x)
                return
        a, b = np.array_split(np.asarray(x), 2)
        if self.check_robust(pixels=np.concatenate([self.unsat_set, a])):
            self.unsat_set.extend(a)
            # self.binary_check_robust(b)
            if self.check_robust(pixels=np.concatenate([self.unsat_set, b])):
                self.unsat_set.extend(b)
            else:
                if len(b) == 1:
                    self.sat_set.extend(b)
                else:
                    self.binary_check_robust(b)
        else:
            if len(a) == 1:
                self.sat_set.extend(a)
            else:
                self.binary_check_robust(a)
            self.binary_check_robust(b)

    def deploy_quickXplain(self, epsilon):
        self.epsilon = epsilon
        x = self.inputVars
        r_, i_ = self.quickXplain([], [], x)
        self.unsat_set = i_
        self.sat_set = r_
        self.plot_explanation()
        self.record_statistics()

    def quickXplain(self, i, r, x):
        if len(x) == 1:
            if self.check_robust(pixels=np.concatenate([i, x]),
                                 fixed_pixels=r):
                return [], np.concatenate([i, x])
            else:
                return x, i

        a, b = np.array_split(np.asarray(x), 2)
        if self.check_robust(pixels=np.concatenate([i, a]),
                             fixed_pixels=np.concatenate([r, b])):
            return self.quickXplain(np.concatenate([i, a]), r, b)
        elif self.check_robust(pixels=np.concatenate([i, b]),
                               fixed_pixels=np.concatenate([r, a])):
            return self.quickXplain(np.concatenate([i, b]), r, a)
        else:
            if len(a) == 1:
                a_, i_ = a, i
            else:
                a_, i_ = self.quickXplain(i, np.concatenate([b, r]), a)
            if len(b) == 1:
                b_, i__ = b, i_
            else:
                b_, i__ = self.quickXplain(i_, np.concatenate([a_, r]), b)

            # a_, i_ = self.quickXplain(i, np.concatenate([b, r]), a)
            # b_, i__ = self.quickXplain(i_, np.concatenate([a_, r]), b)
            return np.concatenate([a_, b_]), i__

    def check_robust(self, pixels=None, fixed_pixels=None, epsilon=None):
        epsilon = self.epsilon if not epsilon else epsilon
        pixels = self.inputVars if pixels is None else pixels.astype(int)
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)
        # for i in self.inputVars:
        for i in pixels:
            """
            Set allowable perturbations on the current pixel and the irrelevant pixels.
            """
            if self.dataset == "mnist":
                self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
            elif self.dataset == "gtsrb" or self.dataset == "cifar10":
                self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
            else:
                print("Dataset not supported: try 'mnist' or 'gtsrb'.")
        # fixed_pixels = list(set(self.inputVars) - set(pixels)) if fixed_pixels is None else fixed_pixels.astype(int)
        fixed_pixels = list(set(self.inputVars) - set(pixels))
        # for i in list(set(self.inputVars) - set(pixels)):
        for i in fixed_pixels:
            if self.dataset == "mnist":
                self.mara_model.setLowerBound(i, image[i][:])
                self.mara_model.setUpperBound(i, image[i][:])
            elif self.dataset == "gtsrb" or self.dataset == "cifar10":
                self.mara_model.setLowerBound(3 * i, image[i][0])
                self.mara_model.setUpperBound(3 * i, image[i][0])
                self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                self.mara_model.setUpperBound(3 * i + 2, image[i][2])
            else:
                print("Dataset not supported: try 'mnist' or 'gtsrb'.")
        # for j in range(len(self.outputVars)):
        for j in self.logit_rank:
            """
            Set constraints on the output variables.
            """
            if j != self.label:
                self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                              [1, -1], -1e-6,
                                              isProperty=True)
                exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                """
                additionalEquList.clear() is to clear the output constraints.
                """
                self.mara_model.additionalEquList.clear()
                if exit_code == 'sat' or exit_code == 'TIMEOUT':
                    break
                elif exit_code == 'unsat':
                    continue
        """
        clearProperty() is to clear both input and output constraints.
        """
        self.mara_model.clearProperty()
        if exit_code == "unsat":
            return True
        else:
            return False

    def compute_explanation(self,
                            epsilon,
                            approach="sequential",
                            deploy_binary_search=True,
                            plot_explanation=True,
                            plot_counterfactual=False,
                            plot_timeout=False,
                            record_statistics=True):
        self.epsilon = epsilon
        self.unsat_set = []
        self.sat_set = []
        if approach == "sequential":
            if deploy_binary_search:
                self.binary_check_robust(self.inputVars)
            else:
                for pixel in self.inputVars:
                    if self.check_robust(pixels=np.concatenate([[pixel], self.unsat_set]),
                                         epsilon=epsilon):
                        self.unsat_set.append(pixel)
                    else:
                        self.sat_set.append(pixel)
        elif approach == "quickXplain":
            self.sat_set, self.unsat_set = self.quickXplain([], [], self.inputVars)
        else:
            print("Unsupported approach; try 'sequential' or 'quickXplain'.")

        if plot_explanation:
            self.plot_explanation()
        if record_statistics:
            self.record_statistics()

    def get_explanation(self,
                        epsilon,
                        plot_explanation=True,
                        plot_counterfactual=False,
                        plot_timeout=False,
                        record_statistics=True):
        """
        To compute the explanation for the model and the neural network.
        :param record_statistics: record statistics if True
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :return: an explanation, and possible counterfactual(s).
        """
        self.epsilon = epsilon
        self.unsat_set = []
        self.sat_set = []
        self.timeout_set = []
        self.marabou_time = []
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)
        for pixel in self.inputVars:
            for i in self.inputVars:
                """
                Set constraints on the input variables.
                """
                if i == pixel or i in self.unsat_set:
                    """
                    Set allowable perturbations on the current pixel and the irrelevant pixels.
                    """
                    if self.dataset == "mnist":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                    elif self.dataset == "gtsrb" or self.dataset == "cifar10":
                        self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                        self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                    else:
                        print("Dataset not supported: try 'mnist' or 'gtsrb'.")
                else:
                    """
                    Make sure the other pixels are fixed.
                    """
                    if self.dataset == "mnist":
                        self.mara_model.setLowerBound(i, image[i][:])
                        self.mara_model.setUpperBound(i, image[i][:])
                    elif self.dataset == "gtsrb" or self.dataset == "cifar10":
                        self.mara_model.setLowerBound(3 * i, image[i][0])
                        self.mara_model.setUpperBound(3 * i, image[i][0])
                        self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                        self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                        self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                        self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                    else:
                        print("Dataset not supported: try 'mnist' or 'gtsrb'.")
            for j in range(len(self.outputVars)):
                """
                Set constraints on the output variables.
                """
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    if record_statistics:
                        marabou_tic = time.time()
                        exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                        marabou_toc = time.time()
                        self.marabou_time.append(marabou_toc - marabou_tic)
                    else:
                        exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    """
                    additionalEquList.clear() is to clear the output constraints.
                    """
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            """
            clearProperty() is to clear both input and output constraints.
            """
            self.mara_model.clearProperty()
            """
            If unsat, put the pixel into the irrelevant set; 
            if timeout, into the timeout set; 
            if sat, into the explanation.
            """
            if exit_code == 'unsat':
                self.unsat_set.append(pixel)
            elif exit_code == 'TIMEOUT':
                self.timeout_set.append(pixel)
            elif exit_code == 'sat':
                self.sat_set.append(pixel)
                if plot_counterfactual:
                    self.plot_counterfactual(vals, pixel)
        if plot_explanation:
            self.plot_explanation()
        if plot_timeout:
            self.plot_timeout()
        if record_statistics:
            self.record_statistics()
            marabou_time = np.asarray(self.marabou_time)
            self.marabou_time = np.mean(marabou_time)

    def plot_original(self):
        save_figure(image=self.image,
                    path=self.directory + f"original-predicted-as-{self.label}.png",
                    cmap="gray" if self.dataset == 'mnist' else None)

    def plot_explanation(self):
        mask = np.zeros(self.inputVars.shape).astype(bool)
        mask[self.sat_set] = True
        mask[self.timeout_set] = True
        plot_shape = self.image.shape[0:2] if self.dataset == "mnist" else self.image.shape
        save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                    self.image.reshape(plot_shape),
                                    colors=[[0, 1, 0]] if self.traverse != "random" else [[1, 0, 0]],
                                    bg_label=0,
                                    saturation=1),
                    path=self.directory + f"explanation-{len(self.sat_set) + len(self.timeout_set)}.png")

    def plot_timeout(self):
        mask = np.zeros(self.inputVars.shape).astype(bool)
        mask[self.timeout_set] = True
        plot_shape = self.image.shape[0:2] if self.dataset == "mnist" else self.image.shape
        save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                    self.image.reshape(plot_shape),
                                    colors=[[0, 1, 0]] if self.traverse != "random" else [[1, 0, 0]],
                                    bg_label=0,
                                    saturation=1),
                    path=self.directory + f"timeout-{len(self.timeout_set)}.png")

    def plot_counterfactual(self, vals, pixel):
        counterfactual = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
        counterfactual = np.asarray(counterfactual).reshape(self.image.shape)
        prediction = [vals.get(i) for i in self.outputVars]
        prediction = np.asarray(prediction).argmax()
        save_figure(image=counterfactual,
                    path=self.directory + f"counterfactual-at-pixel-{pixel}-predicted-as-{prediction}.png",
                    cmap="gray" if self.dataset == 'mnist' else None)

    def record_statistics(self):
        np.savetxt(self.directory + "unsat.txt", self.unsat_set, fmt="%d")
        np.savetxt(self.directory + "sat.txt", self.sat_set, fmt="%d")
        np.savetxt(self.directory + "timeout.txt", self.timeout_set, fmt="%d")

