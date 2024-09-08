import pandas
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# TASK ONE

df = pandas.read_csv("SpotifyFeatures.csv") # load csv

# copy a filter of pop and classical genre
df_filtered = df[df['genre'].isin(["Pop", "Classical"])].copy()

# create a new column "label" which marks the values pop and classical
df_filtered['label'] = np.where(df_filtered['genre'] == 'Pop', 1, 0)

# number of samples for each genre
pop_count = df_filtered[df_filtered['label'] == 1].shape[0]
classical_count = df_filtered[df_filtered['label'] == 0].shape[0]

# use liveness and loudness columns for the logistic regression
df_features = df_filtered[['liveness', 'loudness']]

# use the label column (0 or 1) as target for logistic regression
labels = df_filtered['label']

x= df_features.to_numpy()
y = labels.to_numpy()

# split the dataset into training and test sets (0.2 for 80%/20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# TASK TWO

'''
    logistic function (sigmoid)
    [slide 24 pdf 4]
'''
def logistic_func(z):
    return 1 / (1 + math.exp(-z)) # overflow with np.exp()

'''
    loss function (cross entropy loss)
    [slide 29 pdf 4]
'''
def loss_func(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

'''
    accuracy function
'''
def accuracy_func(y, y_hat):
    y_hat_class = (y_hat >= 0.5).astype(int)
    accuracy = np.mean(y_hat_class == y)
    return accuracy

'''
    logistic regression using stochastic gradient descent
    [slide 36 pdf 4]
'''
def logistic_regression_sgd(x_train, y_train, learning_rate, epochs):
    num_samples, num_features = x_train.shape
    weights = np.zeros(num_features)
    bias = 0
    losses = []

    # training loop (based off earlier experience in INF-1600 and slide 36 pseudo)
    for epoch in range(epochs):
        for i in range(num_samples):
            z = np.dot(x_train[i], weights) + bias # instead of manual calculation
            y_hat = logistic_func(z)

            # gradient of the loss with respect to the weights and bias
            dw = (y_hat - y_train[i]) * x_train[i] # [slide 12 pdf 5]
            db = y_hat - y_train[i]

            # weights and bias (saw that most examples online calculate bias aswell)
            weights -= learning_rate * dw # [slide 8 pdf 5]
            bias -= learning_rate * db

            # loss function 
            loss = loss_func(y_train[i], y_hat)
            losses.append(loss)

    return weights, bias, losses

learning_rates = [0.001, 0.01, 0.1]
loss_dict = {}
for rates in learning_rates:
    weights, bias, losses = logistic_regression_sgd(x_train, y_train, learning_rate=rates, epochs=100)
    loss_dict[rates] = losses

    # calculate accuracy on the training set
    z = np.dot(x_train, weights) + bias
    y_hat_train = logistic_func(z)
    train_accuracy = accuracy_func(y_train, y_hat_train)
    print(f"[{rates}] Training Accuracy: {train_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 6))
for rates, losses in loss_dict.items():
    plt.plot(range(len(losses)), losses, label=f'LR = {rates}')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Error as a Function of Epochs')
plt.legend()
plt.show()

# testing set and print accuracy
z = np.dot(x_test, weights) + bias
y_hat = logistic_func(z)
test_accuracy = accuracy_func(y_test, y_hat)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# TASK THREE

# confusion matrix
y_hat_test = logistic_func(np.dot(x_test, weights) + bias)
y_hat_test_class = (y_hat_test >= 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_hat_test_class)

print("Confusion Matrix:")
print(f"\tTrue negatives {conf_matrix[0][0]}\n\tFalse positives {conf_matrix[0][1]}")
print(f"\tFalse negatives {conf_matrix[1][0]}\n\tTrue positives {conf_matrix[1][1]}")
