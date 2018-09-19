import  torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse


parser = argparse.ArgumentParser(description='image classifier')
parser.add_argument("--data_dir", help="Where the dataset exist." , default = "/home/workspace/aipnd-project/flowers")
parser.add_argument("--save_dir", help="Where you want to save your trained model as a checkpoint." , default = "/home/workspace/paind-project/")
parser.add_argument("--arch", help="Which are pretrained network architecture you want." , choices=["vgg16","vgg19"], default = "vgg16" )
parser.add_argument("--learning_rate", help="Which is the learning rate you want to use." , type=float ,default = 0.001)
parser.add_argument("--hidden_units1", help="First hidden unit number." , type=int ,default = 600)
parser.add_argument("--hidden_units2", help="Second hidden unit number." , type=int ,default = 400)
parser.add_argument("--hidden_units3", help="Third hidden unit number." , type=int ,default = 200)
parser.add_argument("--epochs", help="Epochs number which will use in traing the model." , type=int ,default = 5)
parser.add_argument("--gpu", help="Which is the traing process you want to use." , choices=["cpu","gpu"] ,default = "gpu")
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units1 = args.hidden_units1
hidden_units2 = args.hidden_units2
hidden_units3 = args.hidden_units3
epochs = args.epochs
gpu = args.gpu


def dataset (data_dir):
    # will use directory that will share by user so if data = followers, the folder will have train, vaild, and test folder.
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # torchvision transforms are used to augment the training data
    # The training, validation, and testing data is appropriately transformed

    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    # (train, validation, test) is loaded with torchvision's ImageFolder

    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=test_transforms)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=test_transforms)

    # data for each set is loaded with torchvision's DataLoader

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=50, shuffle=True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=25)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=25)

    return dataloaders, dataloaders_valid, dataloaders_test, image_datasets

def create_model(arch, learning_rate, hidden_units1, hidden_units2, hidden_units3):

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)

    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    input_size = 25088
    hidden_layers = [hidden_units1, hidden_units2, hidden_units3]
    output_size = 102
    drop_p = 0.5

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layers[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(drop_p)),
                          ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(drop_p*0.5)),
                          ('fc3', nn.Linear(hidden_layers[1], hidden_layers[2])),
                          ('relu3', nn.ReLU()),
                          ('fc4', nn.Linear(hidden_layers[2], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)

    return model , optimizer, criterion

def validation(model, dataloaders_valid, criterion, gpu):
    test_loss = 0
    accuracy = 0
    if gpu == 'gpu':
        model.to('cuda')
    elif gpu =='cpu':
        model.to('cpu')

    for data in dataloaders_valid:
        inputs, labels = data
        if gpu == 'gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        elif gpu == 'cpu':
            inputs, labels = inputs.to('cpu'), labels.to('cpu')


        outputs = model.forward(inputs)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)

        equality = (labels == predicted)
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def training(model, epochs, gpu, dataloaders, dataloaders_valid):

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 101
    if gpu == 'gpu':
        model.to('cuda')
    elif gpu =='cpu':
        model.to('cpu')

    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps += 1

            if gpu == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            elif gpu == 'cpu':
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss, accuracy = validation(model, dataloaders_valid, criterion, gpu)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders_valid)),
                    "Validation Accuracy: {:.1f} %".format(100*accuracy/len(dataloaders_valid)))

                running_loss = 0

                model.train()

    return model

def checkpoint_save(model,save_dir, arch, hidden_units1, hidden_units2, hidden_units3, image_datasets):
    state_dict = model.state_dict()
    optimizer_state = optimizer.state_dict()
    model.class_to_idx = image_datasets.class_to_idx
    input_size = 25088
    hidden_layers = [hidden_units1, hidden_units2, hidden_units3]
    output_size = 102
    drop_p = 0.5
    checkpoint = {'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': hidden_layers,
                'drop_p' : drop_p,
                'class_to_idx': model.class_to_idx,
                'arch': arch,
                'state_dict': state_dict,
                'optimizer_state': optimizer_state
                }
    save_dir = save_dir + 'checkpoint.pth'
    checkpoint = torch.save(checkpoint, save_dir)

    return checkpoint

def testing(model, gpu, dataloaders_test):
    correct = 0
    total = 0
    if gpu == 'gpu':
        model.to('cuda')
    elif gpu =='cpu':
        model.to('cpu')
    model.eval()
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(dataloaders_test):
            if gpu == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            elif gpu == 'cpu':
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy= 100 * correct / total
    return test_accuracy
if __name__ == "__main__":
    #First use data_dir to create the dataloaders.
    print("Dataloaders process will start now.\n\n")
    dataloaders, dataloaders_valid, dataloaders_test, image_datasets = dataset(data_dir)
    print("Dataloaders process has been completed. Now we will start with building the model.\n\n")
    model, optimizer, criterion = create_model(arch, learning_rate, hidden_units1, hidden_units2, hidden_units3)
    print("Model has been built. Now training and validation process will start.\n\n")
    model = training(model, epochs, gpu, dataloaders, dataloaders_valid)
    print("Model has been trained, testing process now starting.\n\n")
    test_accuracy = testing(model, gpu, dataloaders_test)
    print('Accuracy of the network on the test images: %d %%.\n\n' %test_accuracy)
    print("Now saving the model in a checkpoint.\n\n")
    checkpoint = checkpoint_save(model,save_dir, arch, hidden_units1, hidden_units2, hidden_units3, image_datasets)
    print('Checkpoint has been saved in your path.')