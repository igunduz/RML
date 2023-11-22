# %%
#### Homework Assignment 1: Adversarial Examples

## This jupyter notebook aims to get familiar with the untargted and targeted methods for generating adversarial examples
## We will try to attack a pretrained ImageNet ResNet50 model on a given ImageNet image .

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

# %%
## Load the sample ImageNet image (which is an image of ladybug) and visualize it 

from PIL import Image
from torchvision import transforms

# read the image, resize to 224 and convert to PyTorch Tensor
ladybug_img = Image.open("ladybug.jpg")

preprocess = transforms.Compose([
   transforms.Resize(224),
   transforms.ToTensor(),
])
ladybug_tensor = preprocess(ladybug_img)[None,:,:,:]

# plot image (note that numpy using HWC whereas Pytorch user CHW, so we need to convert)
#plt.imshow(ladybug_tensor[0].numpy().transpose(1,2,0))

# %%
## Prepare the ImageNet ResNet50 classification model
import torch
import torch.nn as nn
from torchvision.models import resnet50

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# load pre-trained ResNet50, and put into evaluation mode
model = resnet50(pretrained=True)
model.eval()

# %%
import json
with open("imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

# form the prediction on the clean ladybug image
pred_ladybug = model(norm(ladybug_tensor))
print(imagenet_classes[pred_ladybug.max(dim=1)[1].item()])

# %%
# 301 is the class index corresponding to the ImageNet class "ladybug"
# form the cross-entropy loss on the prediction according to the ground truth label
print(nn.CrossEntropyLoss()(model(norm(ladybug_tensor)),torch.LongTensor([301])).item())

# %%
## Your task 1: implement the simple version of untargeted PGD attack - make necessary changes to the following code
import torch.optim as optim

epsilon = 2./255

# delta stores the generated perturbation and updates its value iteratively
delta = torch.zeros_like(ladybug_tensor, requires_grad=True)

for t in range(30):
    pred = model(norm(ladybug_tensor + delta))
    loss = nn.CrossEntropyLoss()(pred, torch.LongTensor([301]))
     
    if t % 5 == 0:
        print(t, loss.item())

# pred stores the model predicted logits of the genreated adversarial example
print("True class probability:", nn.Softmax(dim=1)(pred)[0,301].item())

# %%
## check the predicted class of the generated adversarial example and its prediction probability
max_class = pred.max(dim=1)[1].item()
print("Predicted class: ", imagenet_classes[max_class])
print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())

# %%
# plot the original ladybug image
#plt.imshow(ladybug_tensor[0].detach().numpy().transpose(1,2,0))

# %%
# plot the adversarially perturbed ladybug image
#plt.imshow((ladybug_tensor + delta)[0].detach().numpy().transpose(1,2,0))

# %%
# plot the generated adversarial perturbation
#plt.imshow((50*delta+0.5)[0].detach().numpy().transpose(1,2,0))

# %%
## Your task 2: implement the basic version of targeted attack - make necessary changes to the following code
## Targeted label: zebra; Class index: 340

delta = torch.zeros_like(ladybug_tensor, requires_grad=True)

for t in range(100):
    pred = model(norm(ladybug_tensor + delta))
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([340]))

    if t % 10 == 0:
        print(t, loss.item())

print("True class probability:", nn.Softmax(dim=1)(pred)[0,301].item())

# %%
## check the predicted class of the generated adversarial example and its prediction probability
max_class = pred.max(dim=1)[1].item()
print("Predicted class: ", imagenet_classes[max_class])
print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())

# %%
# plot the original ladybug image
#plt.imshow(ladybug_tensor[0].detach().numpy().transpose(1,2,0))

# %%
# plot the adversarially perturbed ladybug image
#plt.imshow((ladybug_tensor + delta)[0].detach().numpy().transpose(1,2,0))

# %%
# plot the generated adversasrial perturbation
#plt.imshow((50*delta+0.5)[0].detach().numpy().transpose(1,2,0))


