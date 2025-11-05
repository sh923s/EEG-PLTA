import math

from torch import optim, nn

from src.data import load_dataset
from .T3A import T3A
from .bn import AlphaBatchNorm
from .cotta import CoTTA
from .eata import EATA
from .lame import LAME
from .norm import Norm
from .sar import SAR
from .tent import Tent
from .tipi import TIPI
from .ttime import ttime
from .source import Source
from .plta import PLTA
from ..models import *
from ..utils.utils import split_up_model


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def setup_optimizer(params, cfg):
    return optim.Adam(params, lr=cfg['lr'])

def setup_NRC_optimizer(model, cfg):
    encoder, classifier = split_up_model(model)
    param_group = []
    param_group_c = []
    for k, v in encoder.named_parameters():
        param_group += [{'params': v, 'lr': cfg['lr'] * 0.1}]
    for k, v in classifier.named_parameters():
        param_group_c += [{'params': v, 'lr': cfg['lr']* 1}]
    optimizer = optim.Adam(param_group)
    optimizer_c = optim.Adam(param_group_c)
    return op_copy(optimizer), op_copy(optimizer_c)


def setup_shot_optimizer(model, cfg):
    encoder, classifier = split_up_model(model)
    param_group = []
    for k, v in encoder.named_parameters():
        param_group += [{'params': v, 'lr': cfg['lr']*0.1}]
    for k, v in classifier.named_parameters():
        v.requires_grad = False
    optimizer = optim.Adam(param_group)
    return op_copy(optimizer)


def setup_source(model, cfg=None):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    model = Source(model)
    return model, None


def setup_t3a(model, cfg=None):
    params, param_names = Tent.collect_params(model)
    model = T3A.configure_model(model, cfg)
    T3A_model = T3A(model, filter_k=cfg['filter_k'], cached_loader=False)
    return T3A_model, None


def setup_test_norm(model, cfg):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    model.eval()
    for m in model.modules():
        # Re-activate batchnorm layer
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # Wrap test normalization into Norm class to enable sliding window approach
    norm_model = Norm(model)
    return norm_model, None


def setup_tent(model, cfg):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent(model, optimizer,
                      steps=cfg['steps'],
                      episodic=cfg['episodic'],)
    return tent_model, param_names



def setup_ttime(model, cfg):
    model = ttime.configure_model(model)
    params, param_names = ttime.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    model = ttime(model, optimizer,
                      steps=cfg['steps'],
                      episodic=cfg['episodic'],)
    return model, param_names


def setup_cotta(model, cfg):
    model = CoTTA.configure_model(model)
    params, param_names = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    cotta_model = CoTTA(model, optimizer,
                        steps=cfg['steps'],
                        episodic=cfg['episodic'],
                        mt_alpha=cfg['mt'],
                        rst_m=cfg['rst'])
    return cotta_model, param_names


def setup_eata(model, cfg, dset, num_classes, batch_size):
    # compute fisher informatrix
    # sub dataset
    fisher_dataset = torch.utils.data.Subset(dset, range(int(len(dset)*0.3)))
    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=batch_size, shuffle=True)
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, batch in enumerate(fisher_loader, start=1):
        x = batch[0].cuda(non_blocking=True)
        outputs = model(x)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    # logger.info("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = setup_optimizer(params, cfg)
    eta_model = EATA(model, optimizer,
                     steps=cfg['steps'],
                     episodic=cfg['episodic'],
                     fishers=fishers,
                     fisher_alpha=cfg['fisher_alpha'],
                     e_margin=num_classes * 0.40,
                     d_margin=cfg['d_margin'],
                     )

    return eta_model, param_names


def setup_lame(model, cfg):
    model = LAME.configure_model(model)
    lame_model = LAME(model,
                      affinity=cfg['affinity'],
                      knn=cfg['knn'],
                      sigma=cfg['sigma'],
                      force_symmetry=cfg['force_symmetry'])
    return lame_model, None


def setup_alpha_norm(model, cfg):
    """Set up BN--0.1 (test-time normalization adaptation with source prior).
    Normalize features by combining the source moving statistics and the test batch statistics.
    """
    model.eval()
    norm_model = AlphaBatchNorm.adapt_model(model,
                                            alpha=cfg['bn_alpha']).cuda()  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model, None

def setup_sar(model, cfg, num_classes):
    # sar_model = SAR(model, lr=cfg['lr'], steps=cfg['steps'],
    #                 episodic=cfg['episodic'], reset_constant=cfg['reset_constant'],
    #                 e_margin=math.log(num_classes) * 0.40)
    sar_model = SAR(model, lr=cfg['lr'], steps=cfg['steps'],
                    episodic=cfg['episodic'], reset_constant=cfg['reset_constant'],
                    e_margin=cfg['e_margin'])

    return sar_model

def setup_plta(model, cfg, num_classes):
    model = PLTA(model, num_classes, cfg['lr'], optim='Adam', epsilon=cfg['epsilon'],  reverse_kl=cfg['reverse_kl'], conf_threshold=cfg['conf_threshold'], memory_size=cfg['memory_size'], temperature_scaling=cfg['temperature_scaling'], lambda_pi=cfg['lambda_pi'])
    model = PLTA.configure_model(model)
    return model

def setup_plta_ablation(model, cfg, num_classes, IM_loss, PI_loss, label_correction):
    model = pltaModel_ablation(model, num_classes=num_classes, lr=cfg['lr'], optim='Adam', epsilon=cfg['epsilon'], reverse_kl=cfg['reverse_kl'], conf_threshold=cfg['conf_threshold'], memory_size=cfg['memory_size'], temperature_scaling=cfg['temperature_scaling'], lambda_pi=cfg['lambda_pi'] , IM_loss=IM_loss, PI_loss=PI_loss, label_correction=label_correction)
    model = pltaModel_ablation.configure_model(model)
    return model

def setup_plta_runtime(model, cfg, num_classes):
    model = pltaModel_runtime(model, num_classes, cfg['lr'], optim='Adam', epsilon=cfg['epsilon'], reverse_kl=cfg['reverse_kl'], conf_threshold=cfg['conf_threshold'], memory_size=cfg['memory_size'], temperature_scaling=cfg['temperature_scaling'])
    model = pltaModel_runtime.configure_model(model)
    return model

def setup_tipi(model, cfg):
    model = TIPI(model, cfg['lr'], optim='Adam', epsilon=cfg['epsilon'], tent_coeff=cfg['tent_coeff'], random_init_adv=cfg['random_init_adv'], reverse_kl=cfg['reverse_kl'])
    return model
