import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
# Set the device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the CIFAR-10 test dataset
full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

dataset = full_dataset
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

model_names = ['alexnet', 'vgg16', 'resnet34', 'resnet50']



def get_feature_extractor(model_name):
    model = getattr(models, model_name)(pretrained=True)
    if model_name == 'alexnet': 
      model = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((6, 6)), torch.nn.Flatten(), model.classifier[:-1])
    elif model_name == 'vgg16':
      model = torch.nn.Sequential(model.features,  model.avgpool, torch.nn.Flatten(), *model.classifier[:-1])  # Exclude the last fully connected layer
    else:
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model.to(device)

def print_model_to_file(model_name):
	model = get_feature_extractor(model_name)
	features = []
	for inputs, batch_labels in dataloader:
	  inputs = inputs.to(device)
	  with torch.no_grad():
	          outputs = model(inputs)
	          features.extend(outputs.view(outputs.size(0), -1).cpu().numpy())

	print(f"Done with {model_name}")

	save_path = "./cifar-data/features_test_train_" + str(model_name)

	features = np.array(features)

	# Save features_all_models to a .npz file
	np.save(save_path, features)

	print(f"Printed {model_name} to file with shape {features.shape}")


full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = full_dataset
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)


labels_list = []

for inputs, batch_labels in dataloader:
    inputs = inputs.to(device)
    with torch.no_grad():
        labels_list.extend(batch_labels.cpu().numpy())


labels = np.array(labels_list)

## 10-class labels 
labels_save_path = "labels_test_train_all_classes.npy"
np.save(labels_save_path, labels)

## Binary labels 
labels = (labels == 0) | (labels == 2) | (labels == 1) | (labels == 8) | (labels == 7) | (labels == 9)
labels_save_path = "labels_test_train.npy"
np.save(labels_save_path, labels)


