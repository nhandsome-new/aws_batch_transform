import random, os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import json

def create_init_sample_data(save_dir, n):
    '''
    テスト用のデータセットの作成
        CIFAR10テストデータセットから、n個だけ抽出
    '''
    init_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    tmp_data = datasets.CIFAR10(root=save_dir, train=False, download=True, transform=init_transform)

    ids = random.sample(range(len(tmp_data)), n)
    ids = np.array(ids, dtype=np.int32)

    selected_images = []
    for idx in ids:
        img, _ = tmp_data[idx]
        selected_images.append(img.numpy())
        
    return selected_images

def convert_np_to_png(save_dir, np_images):
    for i, img in enumerate(np_images):
        img = (img * 255).astype(np.uint8)
        single_img_reshaped = np.transpose(img, (1,2,0))
        pngimg = Image.fromarray(single_img_reshaped)
        pngimg.save(os.path.join(save_dir, f"sample{i}.png"))
        
        
def create_inference_jsonlist(json_path, folder_dir, aws_dir):
    '''
    Create jsonlists file including the file(path) lists in a folder
    
    Args:
        json_path: path of json file
        folder_dir: directory for searching
        aws_dir: the dir path of files that are saved on S3
    '''
    with open(json_path, mode='w') as f:
        _dict = {}
        for inf_file in os.listdir(folder_dir):
            _dict["img_path"] = os.path.join(aws_dir, inf_file)
            json.dump(_dict, f)
            f.write('\n')