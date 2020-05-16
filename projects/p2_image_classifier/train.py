# Import Libraries

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
import collections

# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args

# Function train_transformer
def train_transformer(train_dir):
   # Define transformation
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the Data
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data

# Function test_transformer
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

# Function valid_transformer
def valid_transformer(valid_dir):
    # Define transformation
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data

# Function data_loader
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

"""Use GPU CUDA"""


# Function check_gpu(gpu_arg)
def check_gpu(gpu_arg):
    # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")

    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


"""Downloads model from torchvision"""


# primaryloader_model(architecture="vgg16") downloads model (primary) from torchvision
def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else:
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture

# Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model


"""Define initial classified model"""


def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None):
        hidden_units = 4096  # hyperparamters
        print("Number of Hidden Layers specificed as 4096.")

    # Find Input Layers
    input_features = model.classifier[0].in_features

    # Define Classifier
    classifier = nn.Sequential(collections.OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('droput1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1000, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier


"""Validates training against testloader to return loss and accuracy"""


# Function validation(model, testloader, criterion, device)
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


"""Training of the network model"""


def network_trainer(model, trainloader, testloader, validloader,device,
                    criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 2
        print("Number of Epochs specificed as 5.")

    print("Training process initializing .....\n")

    # Train Model
    for e in range(epochs):
        running_loss = 0
        model.train()  # Technically not necessary, setting this for good measure

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} | ".format(e + 1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss / print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss / len(testloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy / len(testloader)))

                running_loss = 0
                model.train()

    return model


"""Validate Model"""


# Function validate_model(model, testloader, device)
def validate_model(model, testloader, device):
    # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))


"""Function initial check point"""


# Function initial_checkpoint
def initial_checkpoint(model, save_dir, train_data):
    # Save model at checkpoint
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(save_dir):
            # Create `class_to_idx` attribute in model
            model.class_to_idx = train_data.class_to_idx

            # Create checkpoint dictionary
            checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict()}

            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else:
            print("Directory not found, model will not be saved.")


# Function main() is where all the above functions are called and executed
def main():
    # Get Keyword Args for Training
    args = arg_parser()

    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)

    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)

    # Load Model
    model = primaryloader_model(architecture=args.arch)

    # Build Classifier
    model.classifier = initial_classifier(model,
                                          hidden_units=args.hidden_units)

    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);

    # Send model to device
    model.to(device);

    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else:
        learning_rate = args.learning_rate

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Define deep learning method
    print_every = 30
    steps = 0

    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader,testloader,validloader,
                                    device, criterion, optimizer, args.epochs,
                                    print_every, steps)
    print("\nTraining process is now complete!!")

    # Quickly Validate the model
    validate_model(trained_model, testloader, device)

    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data)


# Run Program
if __name__ == '__main__': main()
