# MNIST using FCN
# 1. It is always the packages
import torch
from torch import nn, optim
# import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import FCN

# 2. Some global variables, like device, num_epochs, lr, all the hyperparameters and other task-specific variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 10e-3
NUM_EPOCHS = 2
BATCH_SIZE = 64
IN_DIM = 784
NUM_CLASSES = 10

# 3. Load the data
train_dataset = datasets.MNIST(root='dataset/', transform=transforms.ToTensor(), train=True, download=True)
# it saves the dataset in folder called 'dataset', performs the conversion into torch tensors and the rescaling of 255
# Also, set train=True, as it is the train_dataset, duh -_-
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# ignore the num_workers for simple tasks
# shuffle=True is to ensure that we don't have the same images in every epoch for every batch
test_dataset = datasets.MNIST(root='dataset/', transform=transforms.ToTensor(), train=False, download=True)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Load the model, from model.py
model = FCN(IN_DIM=IN_DIM, NUM_CLASSES=NUM_CLASSES).to(DEVICE)

# 5. Initialize the loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 6. Training Loop
# train_loader gives us the x, viz the features and y, viz the label; quite common to enumerate thru the train_loader
def train(loader, model):
    for epoch in range(NUM_EPOCHS):
        # print(epoch)
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(train_dataLoader)):
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            # print(data.shape) -> [64, 1, 28, 28];
            # 64 because we described the batch_size
            # so in this batch of the train_dataLoader, we are going to have 64 samples only
            # MNIST is black and white, so the channels is 1
            # we want to flatten 1,28,28 into 784
            data = data.reshape(data.shape[0], -1)

            # forward pass
            y_pred = model(data)
            loss = loss_fn(y_pred, target)

            # back-prop, using optimizer
            optimizer.zero_grad()  # intialize the grads to 0 for every batch, to prevent memory of previous gradients
            loss.backward()

            # gradient descent, viz the Adam
            optimizer.step()
    print('\nTraining is Done!!')

# 7. Evaluate
def test(loader, model):
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            data = data.reshape(data.shape[0], -1)  # [64,784]
            y_predictions = model(data)
            # y_predictions is [64,10], i.e, for every one of the 64 samples, there are 10 columns
            # corresponding to probability of the digit being 0 to 9

            # when dealing with tensors, better to use the below code to extract the argmax
            _, y_pred = y_predictions.max(1)  # along axis=1
            num_correct += (y_pred == target).sum()
            num_samples += data.shape[0]
    # model.train() -> this step is used in some other tasks, where training may have not been done before
    print(f'accuracy is {num_correct}/{num_samples} or {float(num_correct)/float(num_samples):.2f}')


if __name__ == '__main__':
    train(train_dataLoader, model)
    test(train_dataLoader, model)
    test(test_dataLoader, model)
