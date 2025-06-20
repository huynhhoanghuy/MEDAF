import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):

    expansion = 1
    
    def __init__(self, mastermodel, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),   
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.mastermodel = mastermodel

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], avg_output=False, output_dim=-1, resprestride=1, res1ststride=1, res2ndstride=1, inchan=3):
        super().__init__()
        img_chan = inchan
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_chan, 64, kernel_size=3, padding=1, bias=False, stride=resprestride),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.in_channels = 64
        self.conv2_x = self._make_layer(block, 64, num_block[0], res1ststride)
        self.conv3_x = self._make_layer(block, 128, num_block[1], res2ndstride)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.conv6_x = nn.Identity() if output_dim <= 0 else self.conv_layer(512, output_dim, 1, 0)
        self.conv6_is_identity = output_dim <= 0
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if output_dim > -1:
            self.output_dim = output_dim
        else:
            self.output_dim = 512 * block.expansion
        self.avg_output = avg_output

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        print("conv layer input", input_channel, "output", output_channel)
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):
        print("Making resnet layer with channel", out_channels, "block", num_blocks, "stride", stride)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(None, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.conv6_x(output)
        if self.avg_output:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
        return output

def build_backbone(img_size, backbone_name, projection_dim, inchan = 3):
    if backbone_name == 'resnet18':
        backbone = ResNet(output_dim = projection_dim, inchan = inchan, resprestride=1, res1ststride = 1, res2ndstride = 2)
        cam_size = int(img_size / 8)
    elif backbone_name == 'resnet34':
        backbone = ResNet(output_dim = projection_dim, inchan = inchan, num_block = [3,4,6,3], resprestride=1, res1ststride = 2, res2ndstride = 2)
        cam_size = int(img_size / 32)
    else:
        valid_backbone = backbone_name
        raise Exception(f'Backbone \"{valid_backbone}\" is not defined.')
    
    return backbone, backbone.output_dim, cam_size    


class BaselineNet(nn.Module):
    def __init__(self, args):
        super(BaselineNet, self).__init__()
        backbone, feature_dim, _ = build_backbone(img_size=args['img_size'],
                                                  backbone_name=args['backbone'], 
                                                  pretrained=args['pretrained'], 
                                                  projection_dim=-1, 
                                                  inchan=3)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = conv1x1(feature_dim, args['num_known'])

    def forward(self, x, y=None):
        x = self.backbone(x)
        ft = self.classifier(x)
        logits = self.pool(ft)
        logits = logits.view(logits.size(0), -1)
        outputs = {'logits':[logits]}        
        return outputs

    def get_params(self, prefix = 'extractor'):
        extractor_params = list(self.backbone.parameters())
        extractor_params_ids = list(map(id, self.backbone.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
        
        
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit =logit.unsqueeze(0)
        return logit
    
    
class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        super(MultiBranchNet, self).__init__()
        backbone, feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              backbone_name=args['backbone'], 
                                                              projection_dim=-1, 
                                                              inchan=3)
        self.img_size  = args['img_size']
        self.gate_temp = args['gate_temp']
        self.num_known = args['num_known']
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)        
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-6])
        
        self.branch1_l4  = nn.Sequential(*list(backbone.children())[-6:-3])
        self.branch1_l5  = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_known)

        self.branch2_l4  = copy.deepcopy(self.branch1_l4)
        self.branch2_l5  = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_known)
        
        self.branch3_l4  = copy.deepcopy(self.branch1_l4)
        self.branch3_l5  = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_known)
        
        self.gate_l3  = copy.deepcopy(self.shared_l3)
        self.gate_l4  = copy.deepcopy(self.branch1_l4)
        self.gate_l5  = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(Classifier(feature_dim, int(feature_dim/4), bias=True), Classifier(int(feature_dim/4), 3, bias=True))

            
    def forward(self, x, y=None, return_ft=False):
         
        b = x.size(0)
        ft_till_l3 = self.shared_l3(x)
            
        branch1_l4 = self.branch1_l4(ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_ft_cams = self.branch1_cls(branch1_l5)
        b1_logits  = self.avg_pool(b1_ft_cams).view(b, -1)
        
        branch2_l4 = self.branch2_l4(ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_ft_cams = self.branch2_cls(branch2_l5)
        b2_logits  = self.avg_pool(b2_ft_cams).view(b, -1)
        
        branch3_l4 = self.branch3_l4(ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_ft_cams = self.branch3_cls(branch3_l5)
        b3_logits  = self.avg_pool(b3_ft_cams).view(b, -1)
        
        if y is not None:
            cams = torch.cat([
                b1_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b1_ft_cams.shape[-2], b1_ft_cams.shape[-1])),
                b2_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b2_ft_cams.shape[-2], b2_ft_cams.shape[-1])),
                b3_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b3_ft_cams.shape[-2], b3_ft_cams.shape[-1])),
            ], dim = 1)
        
        if return_ft:
            fts = b1_ft_cams.detach().clone() + b2_ft_cams.detach().clone() + b3_ft_cams.detach().clone()
    
        gate_l5   = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_pool = self.avg_pool(gate_l5).view(b, -1)
        gate_pred = F.softmax(self.gate_cls(gate_pool)/self.gate_temp, dim=1)

        gate_logits = torch.stack([b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1)
        gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
        gate_logits = gate_logits.sum(-1)

        logits_list = [b1_logits, b2_logits, b3_logits, gate_logits]
        if return_ft and y is None:
            outputs = {'logits':logits_list, 'gate_pred': gate_pred, 'fts': fts}
        else:
            outputs = {'logits':logits_list, 'gate_pred': gate_pred, 'cams': cams}
        
        return outputs


    def get_params(self, prefix='extractor'):
        extractor_params = list(self.shared_l3.parameters()) +\
                           list(self.branch1_l4.parameters()) + list(self.branch1_l5.parameters()) +\
                           list(self.branch2_l4.parameters()) + list(self.branch2_l5.parameters()) +\
                           list(self.branch3_l4.parameters()) + list(self.branch3_l5.parameters()) +\
                           list(self.gate_l3.parameters()) + list(self.gate_l4.parameters()) + list(self.gate_l5.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
        



def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else conv1x1(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class Expert(nn.Module):
    def __init__(self, in_channels, feature_dim, num_classes):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, feature_dim)
        self.cls = conv1x1(feature_dim, num_classes)

    def forward(self, x):
        feat = self.res_block(x)
        cam = self.cls(feat)
        logits = F.adaptive_avg_pool2d(cam, 1).view(x.size(0), -1)
        return logits, cam


class MultiExpertDynamicNet(nn.Module):
    def __init__(self, args, in_channels, num_classes, num_experts=30, gate_temp=1.0):
        super().__init__()
        backbone, feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              backbone_name=args['backbone'], 
                                                              projection_dim=-1, 
                                                              inchan=3)
        # in_channels = args['in_channels']
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.gate_temp = gate_temp

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ResidualBlock(64, feature_dim),
        )

        # Experts
        self.experts = nn.ModuleList([
            Expert(feature_dim, feature_dim, num_classes) for _ in range(num_experts)
        ])

        # Gate network
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Classifier(feature_dim, feature_dim // 2),
            nn.ReLU(),
            Classifier(feature_dim // 2, num_experts)
        )

    def forward(self, x, y=None, return_ft=False):
        shared_feat = self.shared(x)                      # [B, C, H, W]
        gate_logits = self.gate_net(shared_feat)          # [B, E]
        gate_weights = F.softmax(gate_logits / self.gate_temp, dim=1)  # [B, E]

        logits_list = []
        cams_list = []

        for i in range(self.num_experts):
            logits_i, cam_i = self.experts[i](shared_feat)   # [B, C], [B, C, H, W]
            logits_list.append(logits_i.unsqueeze(-1))       # [B, C, 1]
            if y is not None:
                cams_list.append(
                    cam_i.gather(1, y[:, None, None, None].expand(-1, 1, cam_i.size(2), cam_i.size(3)))
                )

        logits_stack = torch.cat(logits_list, dim=-1)     # [B, C, E]
        gate_weights = gate_weights.unsqueeze(1)          # [B, 1, E]
        logits_fused = torch.sum(logits_stack * gate_weights, dim=-1)  # [B, C]

        output = {
            'logits': [log.squeeze(-1) for log in logits_list] + [logits_fused],
            'gate_pred': gate_weights.squeeze(1)
        }

        if y is not None:
            output['cams'] = torch.cat(cams_list, dim=1)  # [B, E, H, W] nếu cần

        if return_ft:
            # tổng CAM các expert nếu cần cho scoring
            output['fts'] = sum([cam.detach() for _, cam in [self.experts[i](shared_feat) for i in range(self.num_experts)]])

        return output

    def get_params(self, prefix='extractor'):
        shared_params = list(self.shared.parameters())
        expert_params = [p for expert in self.experts for p in expert.parameters()]
        gate_params = list(self.gate_net.parameters())

        if prefix == 'extractor':
            return shared_params + expert_params
        elif prefix == 'classifier':
            all_params = list(self.parameters())
            extractor_ids = set(map(id, shared_params + expert_params))
            return filter(lambda p: id(p) not in extractor_ids, all_params)
    
    def prune_experts(self, keep_indices):
        """
        keep_indices: list of indices (int) of experts to keep.
        """
        # import pdb; pdb.set_trace()
        self.experts = nn.ModuleList([self.experts[i] for i in keep_indices])
        self.num_experts = len(keep_indices)

        # Update final linear layer in gate network
        input_dim = self.gate_net[-3].fc.out_features  # lấy từ FC trước ReLU
        self.gate_net[-1] = Classifier(input_dim, len(keep_indices), bias=True)



