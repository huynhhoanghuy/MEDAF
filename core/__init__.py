from .train import *
from .test import *
from .net import BaselineNet, MultiBranchNet, MultiExpertDynamicNet

def get_model(args):
    # net = MultiBranchNet(args)
    # return net
    print("args:",args)
    return MultiExpertDynamicNet(args,
        in_channels=3,
        num_classes=args['num_known'],         # số class known hiện tại
        num_experts=5,
        gate_temp=args.get('gate_temp', 1.0)   # hoặc hardcode tạm: 1.0
    )

    
