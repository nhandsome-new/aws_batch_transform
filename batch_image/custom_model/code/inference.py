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

from src import get_multi_resnet

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://colab.research.google.com/github/seyrankhademi/ResNet_CIFAR10/blob/master/CIFAR10_ResNet.ipynb

def model_load():
    return get_multi_resnet(50)

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
        # _img_bytes = _decode_bytes(input_data)

        image_as_bytes = io.BytesIO(_img_bytes)
        image = Image.open(image_as_bytes)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        _data.append(image_tensor)
    tensor_data = torch.concat(_data)
    return tensor_data

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_load()
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
        BATCH_SIZEに合わせて分けて処理 → 合体
    '''
    print(f'Input Object Shape: {input_obj.shape}')
    # pred = []
    output = model(input_obj)[0]
    pred = torch.argmax(output, dim=1)
    print(f'PREDS SHAPE:{pred.shape}')
    pred = np.array(pred).tolist()
    # return {"predictions": pred}
    return pred

def output_fn(predictions, accept="application/jsonlines"):
    if accept == "application/jsonlines":
        lines = []
        for pred in predictions:
            lines.append({'predictions': pred})

        json_lines = [json.dumps(l) for l in lines]

        # Join lines and save to .jsonl file
        json_data = '\n'.join(json_lines)

        print(json_data)
        return json.dumps(json_data)
    raise Exception("{} accept type is not supported by this script.".format(accept))