import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import json
import argparse
from torchvision import models

parser = argparse.ArgumentParser(description='image classifier')
parser.add_argument("--image_path", help="Where the image file exist." , default = "/home/workspace/aipnd-project/flowers/test/76/image_02550.jpg")
parser.add_argument("--checkpoint_path", help="Where the checkpoint file exist." , default = "/home/workspace/paind-project/checkpoint.pth")
parser.add_argument("--json_path", help="Where the json file exist." , default = "/home/workspace/aipnd-project/cat_to_name.json" )
parser.add_argument("--top_k", help="How many most likely classes want to print." , type=int, default = 5)
parser.add_argument("--gpu", help="Which is the predicting processer you want to use." , choices=["cpu","gpu"] ,default = "gpu")
args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint_path
json_path = args.json_path
top_k = args.top_k
gpu = args.gpu



def catefgory_names (json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    width, height = image.size
    ratio_w = width/height
    ratio_h = height/width
    if width<height:
        size = 256, height*ratio_h
    if width>height:
        size = width*ratio_w, 256
    else :
        size = (256 , 256)
    image.thumbnail(size)
    xn = (image.size[0] - 224)/2
    yp = (image.size[1] - 224)/2
    xp = (image.size[0] + 224)/2
    yn = (image.size[1] + 224)/2
    image = image.crop((xn, yp, xp, yn))
    image = np.array(image)
    image = image/255
    image = image - np.array([0.485, 0.456, 0.406]) 
    image = image / np.array([0.229, 0.224, 0.225])
    image = image.transpose(2, 0, 1)    
    return image


def predict(image_path, model, top_k, gpu):
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    if gpu == 'gpu':
        model.to('cuda')
        image= image.cuda(async=True)
    elif gpu == 'cpu':
        model.to('cpu')
        image= image.to('cpu')
    
    model.eval()

    output = model(image)  
    probs, classes = torch.exp(output).topk(top_k)
    probs = probs.cpu()
    classes = classes.cpu()
    probs = probs.detach().numpy().tolist()[0]
    classes = classes.detach().numpy().tolist()[0]
    idx_new = {v:k for k,v in model.class_to_idx.items()}
    label = [idx_new[k] for k in classes]
    classes = [cat_to_name[idx_new[k]] for k in classes] 
    
    return probs , classes  


def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == "vgg19":
        model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    input_size = checkpoint['input_size']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    drop_p = checkpoint['drop_p']
    model.class_to_idx = checkpoint['class_to_idx']  
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
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state'])    
    return model 

if __name__ == "__main__": 
    cat_to_name = catefgory_names (json_path)
    model = load_checkpoint(checkpoint_path)
    probs , classes = predict(image_path, model, top_k, gpu)
    step = 1
    for step in range(top_k):
        print ("Your image is probablity to be by {:.1f} % ".format(probs[step] *100), "This type of followers =  ", classes[step], "\n" )
        step ++ 1