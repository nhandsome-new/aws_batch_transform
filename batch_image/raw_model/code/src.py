"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class MiddleClassifier(nn.Module):
    def __init__(self, block, in_size, out_size, kenel_size, num_classes=100):
        super().__init__()
        channel_in = in_size * block.expansion
        channel_out = out_size * block.expansion
        kernel_size = kenel_size
        middle_channel = channel_out//4
        
        self.kd_bottleneck =  nn.Sequential(
            nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            
            nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            
            nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc = nn.Linear(512 * block.expansion, num_classes)
    
    def forward(self, inputs):
        middle_output = self.kd_bottleneck(inputs)
        middle_output = self.avgpool(middle_output)
        middle_feature = middle_output
        middle_output = torch.flatten(middle_feature, 1)
        middle_output = self.middle_fc(middle_output)
        
        return middle_feature, middle_output

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
    
class MultiResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        self.middle_classfier_1 = MiddleClassifier(block, 64, 512, 8)
        self.middle_classfier_2 = MiddleClassifier(block, 128, 512, 4)
        self.middle_classfier_3 = MiddleClassifier(block, 256, 512, 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        middle_feature_1, middle_output_1 = self.middle_classfier_1(output)
        
        output = self.conv3_x(output)
        middle_feature_2, middle_output_2 = self.middle_classfier_2(output)
        
        output = self.conv4_x(output)
        middle_feature_3, middle_output_3 = self.middle_classfier_3(output)
        
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        final_feature = output
        
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output, middle_output_1, middle_output_2, middle_output_3, \
            final_feature, middle_feature_1, middle_feature_2, middle_feature_3

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

def get_resnet(resnet_version):
    resnets = {
        18: resnet18(),
        34: resnet34(),
        50: resnet50()
    }
        
    return resnets[resnet_version]

def multi_resnet18():
    return MultiResNet(BasicBlock, [2, 2, 2, 2])

def multi_resnet50():
    return MultiResNet(BasicBlock, [3,4,6,3])

def get_multi_resnet(resnet_version):
    resnets = {
        18: multi_resnet18(),
        50: multi_resnet50()
    }
        
    return resnets[resnet_version]

if __name__ == '__main__':
    SAVE_PATH = '/Users/han/Desktop/Fusic/2022/aws_batch_transform/batch_image/custom_model/model.pth'
    saved_ckpt = '/Users/han/Desktop/Fusic/2022/aws_batch_transform/batch_image/custom_model/epoch=120.ckpt'
    
    
    # checkpoint = torch.load(saved_ckpt, map_location=torch.device('cpu'))
    # model = get_multi_resnet(50)
    # # model = torch.nn.DataParallel(model)
    # state_dict = {k.replace('model.',''):v for k, v in checkpoint['state_dict'].items()}
    # model.load_state_dict(state_dict, strict=False)
    
    # torch.save(model.state_dict(), SAVE_PATH)
    model = get_multi_resnet(50)
    # state_dict = {k.replace('model.',''):v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(torch.load(SAVE_PATH))
    
    tmp_input = torch.rand(10,3,32,32)
    output = model(tmp_input)[0]
    output = torch.argmax(output, dim=1)
    import numpy as np
    output = np.array(output).tolist()
    print(output)