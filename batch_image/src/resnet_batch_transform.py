import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from PIL import Image
import base64, io
import numpy as np

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://colab.research.google.com/github/seyrankhademi/ResNet_CIFAR10/blob/master/CIFAR10_ResNet.ipynb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
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

BATCH_SIZE = 128

def model_load():
    '''
    どんなモデルを使いますか
    '''
    return resnet20()

def _decode_bytes(bytes):
    str_decode = bytes.encode('utf-8')
    bytes_decode = base64.b64decode(str_decode)
    return bytes_decode

def _decode_request(request_body):
    lines = request_body.decode("utf-8").rstrip(os.linesep).split(os.linesep)
    _data = []
    for line in lines:
        line = line.strip()
        input_data = json.loads(line)
        _img_bytes = input_data.pop('inputs',input_data)
        _img_bytes = _decode_bytes(_img_bytes)
        image_as_bytes = io.BytesIO(_img_bytes)
        image = Image.open(image_as_bytes)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        _data.append(image_tensor)
    tensor_data = torch.concat(_data)
    return tensor_data

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model_load())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def input_fn(request_body, content_type='application/jsonlines'):
    '''
    入力データの形によって、encode / decodeが変わる
    ここでは、Image - ENCODE - bytes - DECODE - Image
    Image以外には簡単ではないかな？
    '''
    if content_type == 'application/jsonlines':
        print("request received : application/jsonlines")
        # Warning: for some reason, when Sagemaker is doing batch transform,
        # it automatically adds a line break in the end, needs to strip the line break to avoid errors.
        # Sagemaker Endpoint doesn't have such issue.
        data = _decode_request(request_body)
        print(f'Num of data in a request: {len(data)}')
        return data
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_obj, model):
    '''
    Mini BatchのサイズがモデルのCapaより大きい場合、
        BATCH_SIZEに合わせて分けて処理 → concat
    '''
    print(f'Input Object Shape: {input_obj.shape}')
    pred = []
    if len(input_obj) <= BATCH_SIZE:
        print('Input Data Size <= BATCH_SIZE')
        output = model(input_obj)
        pred += torch.argmax(output, dim=1)
    else:
        print('Input Data Size > BATCH_SIZE')
        print(f'Split input data by BATCH_SIZE:{BATCH_SIZE}')
        batch_list = torch.split(input_obj, BATCH_SIZE, dim=0)
        for batch in batch_list:
            output = model(batch)
            pred += torch.argmax(output, dim=1)
    
    pred = np.array(pred).tolist()
   
    return {"predictions": pred}

def output_fn(predictions, response_content_type):
    '''
    Response 形を変えることができる
        Associate result with input で触るかも
    '''
    
    return json.dumps(predictions)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # Data and model checkpoints directories
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--backend', type=str, default=None,
#                         help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

#     # Container environment
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
#     parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # train(parser.parse_args())