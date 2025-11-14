import torch
import torch.nn as nn
from collections import OrderedDict
import sys
sys.path.append('../../lightweight_openpose/models')
from mobilenetv3 import MobileNetV3_Small

def make_layers(block, no_relu_layers):
    """和原代码一样的层构建函数"""
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            # 检查是否是深度可分离卷积
            if len(v) == 6:  # 有groups参数
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                 kernel_size=v[2], stride=v[3],
                                 padding=v[4], groups=v[5])
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                 kernel_size=v[2], stride=v[3],
                                 padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class ImprovedOpenPose(nn.Module):
    def __init__(self, use_mobilenet=True, use_depthwise=True, optimize_7x7=True):
        super(ImprovedOpenPose, self).__init__()
        
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L2']
        
        blocks = {}
        
        # 改进1: 主干网络
        if use_mobilenet:
            self.model0 = MobileNetV3_Small(use_ema=True)
            # MobileNetV3输出96通道，需要适配
            self.adapt = nn.Conv2d(96, 128, 1)
        else:
            # 原始VGG主干
            block0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                ('conv4_4_CPM', [256, 128, 3, 1, 1])
            ])
            self.model0 = make_layers(block0, no_relu_layers)
        
        # Stage 1（改进2: 深度可分离卷积，合并前三层为共享层）
        if use_depthwise:
            # 共享的前3层（两个分支公用，减少重复计算）
            shared_stage1 = OrderedDict([
                ('conv5_1_CPM_shared_dw', [128, 128, 3, 1, 1, 128]),  # 深度卷积
                ('conv5_1_CPM_shared_pw', [128, 128, 1, 1, 0]),       # 逐点卷积
                ('conv5_2_CPM_shared_dw', [128, 128, 3, 1, 1, 128]),
                ('conv5_2_CPM_shared_pw', [128, 128, 1, 1, 0]),
                ('conv5_3_CPM_shared_dw', [128, 128, 3, 1, 1, 128]),
                ('conv5_3_CPM_shared_pw', [128, 128, 1, 1, 0]),
            ])

            # PAF分支（只有后两层）
            block1_1 = OrderedDict([
                # ('conv5_1_CPM_L1_dw', [128, 128, 3, 1, 1, 128]),  # 深度卷积
                # ('conv5_1_CPM_L1_pw', [128, 128, 1, 1, 0]),       # 逐点卷积
                # ('conv5_2_CPM_L1_dw', [128, 128, 3, 1, 1, 128]),
                # ('conv5_2_CPM_L1_pw', [128, 128, 1, 1, 0]),
                # ('conv5_3_CPM_L1_dw', [128, 128, 3, 1, 1, 128]),
                # ('conv5_3_CPM_L1_pw', [128, 128, 1, 1, 0]),
                ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
            ])
            # Heatmap分支（只有后两层）
            block1_2 = OrderedDict([
                # ('conv5_1_CPM_L2_dw', [128, 128, 3, 1, 1, 128]),
                # ('conv5_1_CPM_L2_pw', [128, 128, 1, 1, 0]),
                # ('conv5_2_CPM_L2_dw', [128, 128, 3, 1, 1, 128]),
                # ('conv5_2_CPM_L2_pw', [128, 128, 1, 1, 0]),
                # ('conv5_3_CPM_L2_dw', [128, 128, 3, 1, 1, 128]),
                # ('conv5_3_CPM_L2_pw', [128, 128, 1, 1, 0]),
                ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
            ])
        else:
            # 原始Stage 1（不共享前三层）
            shared_stage1 = None
            block1_1 = OrderedDict([
                ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
            ])
            block1_2 = OrderedDict([
                ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
            ])

        # 构建共享层
        if shared_stage1 is not None:
            blocks['shared_stage1'] = shared_stage1

        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2
        
        # Stage 2-6（改进3: 3个3×3替换7×7）
        for i in range(2, 7):
            if optimize_7x7:
                # 用3个3×3替换5个7×7
                # 定义每个Stage的共享前15层
                shared_refinement = OrderedDict([
                    ('Mconv1_stage%d_shared_a' % i, [185, 128, 3, 1, 1]),
                    ('Mconv1_stage%d_shared_b' % i, [128, 128, 3, 1, 1]),
                    ('Mconv1_stage%d_shared_c' % i, [128, 128, 3, 1, 1]),
                    ('Mconv2_stage%d_shared_a' % i, [128, 128, 3, 1, 1]),
                    ('Mconv2_stage%d_shared_b' % i, [128, 128, 3, 1, 1]),
                    ('Mconv2_stage%d_shared_c' % i, [128, 128, 3, 1, 1]),
                    ('Mconv3_stage%d_shared_a' % i, [128, 128, 3, 1, 1]),
                    ('Mconv3_stage%d_shared_b' % i, [128, 128, 3, 1, 1]),
                    ('Mconv3_stage%d_shared_c' % i, [128, 128, 3, 1, 1]),
                    ('Mconv4_stage%d_shared_a' % i, [128, 128, 3, 1, 1]),
                    ('Mconv4_stage%d_shared_b' % i, [128, 128, 3, 1, 1]),
                    ('Mconv4_stage%d_shared_c' % i, [128, 128, 3, 1, 1]),
                    ('Mconv5_stage%d_shared_a' % i, [128, 128, 3, 1, 1]),
                    ('Mconv5_stage%d_shared_b' % i, [128, 128, 3, 1, 1]),
                    ('Mconv5_stage%d_shared_c' % i, [128, 128, 3, 1, 1]),
                ])
                blocks['shared_stage%d' % i] = shared_refinement

                # PAF分支（只保留后2层）
                blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

                blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])
            else:
                # 原始7×7卷积
                blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])
                blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])
        
        # 构建所有blocks
        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)
        
        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']
        
        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']
        
        self.use_mobilenet = use_mobilenet
       
        if use_depthwise:
            self.shared_stage1 = blocks['shared_stage1']
        else:
            self.shared_stage1 = None

        # 保存共享层
        if optimize_7x7:
            self.shared_stage2 = blocks['shared_stage2']
            self.shared_stage3 = blocks['shared_stage3']
            self.shared_stage4 = blocks['shared_stage4']
            self.shared_stage5 = blocks['shared_stage5']
            self.shared_stage6 = blocks['shared_stage6']
        else:
            self.shared_stage2 = None
            self.shared_stage3 = None
            self.shared_stage4 = None
            self.shared_stage5 = None
            self.shared_stage6 = None

        self.optimize_7x7 = optimize_7x7

    def forward(self, x):
        if self.use_mobilenet:
            features = self.model0(x)
            out1 = self.adapt(features[-1])  # 适配通道数
        else:
            out1 = self.model0(x)
        
        # Stage 1: 共享前3层
        if self.shared_stage1 is not None:
            shared_features = self.shared_stage1(out1)
            out1_1 = self.model1_1(shared_features)
            out1_2 = self.model1_2(shared_features)
        else:
            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            
        out2 = torch.cat([out1_1, out1_2, out1], 1)
            
            # Stage 2: 共享前15层
        if self.optimize_7x7:
            shared2 = self.shared_stage2(out2)
            out2_1 = self.model2_1(shared2)
            out2_2 = self.model2_2(shared2)
        else:
            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
        
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        
        # Stage 3: 共享前15层
        if self.optimize_7x7:
            shared3 = self.shared_stage3(out3)
            out3_1 = self.model3_1(shared3)
            out3_2 = self.model3_2(shared3)
        else:
            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
        
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        
        # Stage 4: 共享前15层
        if self.optimize_7x7:
            shared4 = self.shared_stage4(out4)
            out4_1 = self.model4_1(shared4)
            out4_2 = self.model4_2(shared4)
        else:
            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
        
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        
        # Stage 5: 共享前15层
        if self.optimize_7x7:
            shared5 = self.shared_stage5(out5)
            out5_1 = self.model5_1(shared5)
            out5_2 = self.model5_2(shared5)
        else:
            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
        
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        
        # Stage 6: 共享前15层
        if self.optimize_7x7:
            shared6 = self.shared_stage6(out6)
            out6_1 = self.model6_1(shared6)
            out6_2 = self.model6_2(shared6)
        else:
            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
        
        return out6_1, out6_2


# 测试对比
if __name__ == "__main__":
    print("="*70)
    print("OpenPose模型对比测试")
    print("="*70)
    
    x = torch.randn(1, 3, 368, 368)
    
    # 原始模型
    print("\n【原始OpenPose】")
    from model import bodypose_model
    original = bodypose_model()
    paf_o, heatmap_o = original(x)
    params_o = sum(p.numel() for p in original.parameters())
    print(f"  参数量: {params_o:,} ({params_o/1e6:.2f}M)")
    print(f"  输出: PAF {paf_o.shape}, Heatmap {heatmap_o.shape}")
    
    # 改进版本
    configs = [
        ("只替换主干", True, False, False),
        ("主干+深度可分离", True, True, False),
        ("主干+深度可分离+7×7优化", True, True, True)
    ]
    
    for name, use_mb, use_dw, opt_7x7 in configs:
        print(f"\n【{name}】")
        model = ImprovedOpenPose(use_mobilenet=use_mb, 
                                use_depthwise=use_dw, 
                                optimize_7x7=opt_7x7)
        paf, heatmap = model(x)
        params = sum(p.numel() for p in model.parameters())
        reduction = (params_o - params) / params_o * 100
        print(f"  参数量: {params:,} ({params/1e6:.2f}M)")
        print(f"  压缩: {reduction:.1f}%")
        print(f"  输出: PAF {paf.shape}, Heatmap {heatmap.shape}")