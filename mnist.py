from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import tf2onnx
import argparse
import os
from VeriX import *

"""
download and process MNIST data.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

"""
show a simple example usage of VeriX. 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network', type=str, default='mnist-10x2')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--traverse', type=str, default='heuristic')
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon
traverse = args.traverse

result_dir = f"{dataset}-{index}-{model_name}-{traverse}-linf{epsilon}/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

verix_tick = time.time()
verix = VeriX(dataset=dataset,
              image=x_test[index],
              model_path="models/" + model_name,
              directory=result_dir,
              plot_original=True)
if verix.check_robust(epsilon=epsilon):
    print(f"Model is robust to {epsilon}-perturbation.")
    with open(f"{dataset}-{model_name}-{traverse}-linf{epsilon}-robust.txt", 'a') as f:
        f.write(str(index) + '\n')
    exit()


# verix+: bound-traversal + logit ranking
verix.new_traversal(epsilon=epsilon,
                    traverse="bounds")
verix.compute_explanation(epsilon=epsilon,
                          approach="sequential",
                          deploy_binary_search=False)

# verix+: bound-traversal + logit ranking + binary search
verix.new_traversal(epsilon=epsilon,
                    traverse="bounds")
verix.compute_explanation(epsilon=epsilon,
                          approach="sequential",
                          deploy_binary_search=True)

# verix+: bound-traversal + logit ranking + quickXplain
verix.new_traversal(epsilon=epsilon,
                    traverse="bounds")
verix.compute_explanation(epsilon=epsilon,
                          approach="quickXplain")

# verix baseline: heuristic + sequential, no logit ranking / binary search / quickXplain
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=epsilon)


verix_toc = time.time()
verix_time = verix_toc - verix_tick
verix_time_text = f"{dataset}-{model_name}-{traverse}-linf{epsilon}-time.txt"
with open(verix_time_text, 'a') as f:
    f.write(str(verix_time) + '\n')
verix_size_text = f"{dataset}-{model_name}-{traverse}-linf{epsilon}-size.txt"
with open(verix_size_text, 'a') as f:
    f.write(str(len(verix.sat_set)+len(verix.timeout_set)) + '\n')
marabou_time_text = f"{dataset}-{model_name}-{traverse}-linf{epsilon}-marabou.txt"
with open(marabou_time_text, 'a') as f:
    f.write(str(verix.marabou_time) + '\n')

exit()

"""
or you can train your own MNIST model.
Note: to obtain sound and complete explanations, train the model from logits directly.
"""
model_name = 'mnist-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# model.save('models/' + model_name + '.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='models/' + model_name + '.onnx')




