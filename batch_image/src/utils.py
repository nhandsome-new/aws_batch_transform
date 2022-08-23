import random, os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import json, base64

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

def _encode_image(image):
    data_encode_bytes = base64.b64encode(image)
    data_encode_str = data_encode_bytes.decode('utf-8')
    return data_encode_str

def _decode_bytes(bytes):
    str_decode = bytes.encode('utf-8')
    bytes_decode = base64.b64decode(str_decode)
    return bytes_decode

def image_to_bytes(json_path, data_dir, inference_dir, num_sample=100):
    # sample N images and convert into png
    # save in inference_dir
    sampled_imgs = create_init_sample_data(data_dir, num_sample)
    convert_np_to_png(inference_dir, sampled_imgs)

    with open(json_path, mode='w') as writer:
        _dict = {}
        for inf_file in os.listdir(inference_dir):
            img_path = os.path.join(inference_dir, inf_file)
            with open(img_path, "rb") as image:
                b = _encode_image(image.read())
            _dict['inputs'] = b
            json.dump(_dict, writer)
            writer.write('\n')

    