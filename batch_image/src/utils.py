import random, os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

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
        pngimg.save(os.path.join('./', f"sample1.png"))
        pngimg.save(os.path.join(save_dir, f"sample{i}.png"))