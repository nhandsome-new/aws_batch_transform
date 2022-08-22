import torch
import os, io, json
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

import logging
logger = logging.getLogger()

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

    
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(resnet20())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    image_tensor = ToTensor()(image).unsqueeze(0)    
    return image_tensor

# import numpy as np
# def input_fn(request_body, request_content_type):
#     # if set content_type as "image/jpg" or "application/x-npy", 
#     # the input is also a python bytearray
#     if request_content_type == "application/x-image": 
#         image_tensor = load_from_bytearray(request_body)
#     elif request_content_type == "application/json":
#         logger.info('asdfasdfasfd')
#         _json_data = json.loads(request_body)
#         logger.info(_json_data)
#         input = _json_data.pop("inputs", _json_data)
#         logger.info(input)
#         image_tensor = np.array(input)
#     else:
#         raise ValueError(f"not support this type yet: {request_body}")
#     return image_tensor

# Perform prediction on the deserialized object, with the loaded model
# def predict_fn(input_object, model):
#     output = model.forward(input_object)
#     pred = output.max(1, keepdim=True)[1]

#     return {"predictions": pred.item()}

# Serialize the prediction result into the desired response content type
# def output_fn(predictions, response_content_type):
#     return json.dumps(predictions)

def preprocess(input_data, content_type):
    if content_type == 'application/json':
        return json.loads(input_data)
    else:
        raise ValueError(f"not support this type yet: {content_type}")

def predict(data, model):
    # pop inputs for pipeline
    inputs = data.pop("inputs", data)

    output = model.forward(inputs)
    pred = output.max(1, keepdim=True)[1]

    return pred

def postprocess(predictions, accept):
    return {"predictions": predictions.item()}

def transform_fn(model, input_data, content_type, accept):
    import time
    logger.info('transform-han')
    # run pipeline
    start_time = time.time()
    processed_data = preprocess(input_data, content_type)
    preprocess_time = time.time() - start_time
    predictions = predict(processed_data, model)
    predict_time = time.time() - preprocess_time - start_time
    response = postprocess(predictions, accept)
    postprocess_time = time.time() - predict_time - preprocess_time - start_time

    logger.info(
        f"Preprocess time - {preprocess_time * 1000} ms\n"
        f"Predict time - {predict_time * 1000} ms\n"
        f"Postprocess time - {postprocess_time * 1000} ms"
    )

    return response