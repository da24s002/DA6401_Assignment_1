from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import wandb

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="da24s002-indian-institute-of-technology-madras",
    # Set the wandb project where this run will be logged.
    project="DA6401_Assignment_1"
)

dataset = fashion_mnist.load_data()

X, y = dataset[0]
X_test, y_test = dataset[1]

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape, X_validation.shape, y_validation.shape, X_test.shape, y_test.shape)

#labels of fashion mnist
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

sample_images = [(X_train[list(y_train).index(i)], list(y_train).index(i)) for i in range(len(labels))]

wandb.run.name = "class_wise_images"
for image, index in sample_images:
    wandb.log({"sample images from each class": [wandb.Image(image, caption=labels[y_train[index]]) for image, index in sample_images]})

wandb.finish()