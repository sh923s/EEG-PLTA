import os
import torch
from ..models import *

def load_model(args, checkpoint_dir=None, device='gpu'):
    if args.backbone == 'eegnet':
        model = EEGNet(
            n_classes=args.class_num,
            Chans=args.ch_num,
            Samples=args.time_sample_num,
            kernLenght=int(args.sample_rate // 2),
            F1=4,
            D=2,
            F2=8,
            dropout_rate=0.25,
            norm_rate=0.5
        )
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        if torch.cuda.is_available() and device == 'gpu':
            model.cuda()
    else:
        raise ValueError("Invalid backbone")

        
    return model
