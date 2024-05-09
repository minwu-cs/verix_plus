import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tf2onnx
import argparse
import os
from VeriX import *

"""
load and process GTSRB data.
"""
gtsrb_path = 'models/gtsrb.pickle'
with open(gtsrb_path, 'rb') as handle:
    gtsrb = pickle.load(handle)
x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_valid = to_categorical(y_valid, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
gtsrb_labels = ['50 mph', '30 mph', 'yield', 'priority road',
                'keep right', 'no passing for large vechicles', '70 mph', '80 mph',
                'road work', 'no passing']

"""
show a simple example usage of VeriX. 
"""


# x = x_test[0:2]
# x = np.moveaxis(x, -1, 1)
# x = torch.from_numpy(x)
#
# model = torch.load("models/gtsrb-cnn.pt")
# model.eval()
# model(x)
#
# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
# b_model = BoundedModule(model, x)
# ptb = PerturbationLpNorm(eps=0.01, norm=np.inf, x_L=x, x_U=x)
# b_x = BoundedTensor(x, ptb)
# b_model(b_x)
# lb, up = b_model.compute_bounds(x=b_x, method='IBP')
#
#
#
# exit()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gtsrb')
parser.add_argument('--network', type=str, default='gtsrb-cnn')
parser.add_argument('--index', type=int, default=9)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--traverse', type=str, default='bounds')
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
or you can train your own GTSRB model.
Note: to obtain sound and complete explanations, train the model from logits directly.
 """
# model_name = 'gtsrb-10x2'
# model = Sequential(name=model_name)
# model.add(Flatten(input_shape=(32, 32, 3)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10))

model_name = 'gtsrb-cnn'
model = Sequential(name=model_name)
model.add(Conv2D(4, 3, (2, 2), input_shape=(32, 32, 3)))
model.add(Conv2D(4, 3, (2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10))

model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
datagen = ImageDataGenerator()
model.fit(datagen.flow(x=x_train, y=y_train, batch_size=64),
          steps_per_epoch=100,
          epochs=20,
          validation_data=(x_valid, y_valid),
          shuffle=1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save('models/' + model_name + '.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='models/' + model_name + '.onnx')

