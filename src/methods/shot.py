import logging
import os

import numpy as np
from scipy.spatial.distance import cdist

from .setup import setup_shot_optimizer
from ..data.data import load_dataset_idx
from ..models import *
from ..utils.utils import split_up_model, Entropy, cal_acc

logger = logging.getLogger(__name__)


def obtain_label(loader, encoder, classifier, cfg):
    start_test = True
    with torch.no_grad():
        for inputs, labels, idx in loader:
            inputs = inputs.cuda()
            feas = encoder(inputs).reshape(inputs.size(0), -1)
            outputs = classifier(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg['distance'] == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > cfg['threshold'])
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], cfg['distance'])
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logger.info(log_str)

    return predict.astype('int')


def train_target(args, type='eval'):
    model_config = args.model_config
    _, dataset = load_dataset_idx(args, pre_align=args.ea_align)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker)
    ## set base network
    model = load_model(args, checkpoint_dir=args.ckpt_path)
    if torch.cuda.is_available():
        model.cuda()
    # set optimizer
    optimizer = setup_shot_optimizer(model, model_config)
    encoder, classifier = split_up_model(model)
    # building feature bank and score bank
    classifier.eval()
    max_iter = args.test_epoch * len(loader)
    iter_num = 0
    for epoch in range(args.test_epoch):
        if model_config['cls_par'] > 0:
            encoder.eval()
            mem_label = obtain_label(loader, encoder, classifier, model_config)
            mem_label = torch.from_numpy(mem_label).cuda()
            encoder.train()
        for inputs_test, _, tar_idx in loader:
            iter_num += 1
            inputs_test = inputs_test.cuda()
            features_test = encoder(inputs_test).reshape(inputs_test.size(0), -1)
            outputs_test = classifier(features_test)
            if model_config['cls_par'] > 0:
                pred = mem_label[tar_idx]
                classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
                classifier_loss *= model_config['cls_par']
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + model_config['epsilon']))
            entropy_loss -= gentropy_loss
            im_loss = entropy_loss * model_config['ent_par']
            classifier_loss += im_loss

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
        encoder.eval()
        acc = cal_acc(loader, model)
        logger.info(f"Epoch: {epoch}, acc: {acc:.2f}%")
        encoder.train()
    return acc/100.
