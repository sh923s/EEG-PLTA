import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from src.data.data import load_dataset
from src.models import load_model
from src.utils import loss, fix_seed, print_args, load_dset_config
from src.utils.loss import CrossEntropyLabelSmooth


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_source(args):
    # load dataset
    dset, test_dset = load_dataset(args, args.session, args.align)
    dset_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.worker)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    
    # get base network
    model = load_model(args).cuda()
    
    # configure optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loader)
    interval_iter = max_iter // args.max_epoch
    iter_num = 0

    model.train()

    # train
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loader)
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = model(inputs_source)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, _ = cal_acc(test_loader, model, False)
            log_str = 'Task: T{}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.target_subject_id, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            model.train()

    torch.save(model.state_dict(), osp.join(args.output_dir_src, "model.pt"))

    return model


def test_target(args):
    ## set base network
    model = load_model(args).cuda()
    model.load_state_dict(torch.load(osp.join(args.output_dir_src, "model.pt")))
    _, dset = load_dataset(args, session=args.session, align=args.align)
    dset_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    model.eval()
    acc, _ = cal_acc(dset_loader, model, False)
    log_str = '\nTesting: {}, Task: T{}, Accuracy = {:.2f}%'.format(args.trte, args.target_subject_id, acc)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0,1', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--domain', type=str, default='MotorImagery')
    parser.add_argument('--dset_root', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--backbone', type=str, default='eegnet', help="eegnet", choices=['eegnet', 'basenet'])
    parser.add_argument('--num_rep', type=int, default=10, help="number of repetitions")
    parser.add_argument('--smooth', type=float, default=0) # cross entropy label smooth
    parser.add_argument('--output', type=str, default='./ckpt')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--scenario', type=str, default='normal', choices=['normal', 'noise',
                                                                          'channel_shuffle',
                                                                          'amplitude_scaling'])
    parser.add_argument('--session', type=int, default=1, help="session id for datasets with multiple sessions")
    parser.add_argument('--align', type=str, default='ea', choices=['no_align', 'ea', 'coral'])
    args = parser.parse_args()
        
    # set gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.domain == 'MotorImagery':
        dset_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))
    
    for dset in dset_list:
        args.dset = dset
        # load dataset config
        load_dset_config(args)

        for i in range(args.num_rep):
            seed = i
            # fix random seed
            fix_seed(seed)
            # iterate over target subjects
            for target_subject_id in range(args.subject_num):
                args.target_subject_id = target_subject_id
                args.output_dir_src = osp.join(args.output, args.domain, args.dset, f"Session{args.session}", args.backbone, args.align, f"Seed{seed}", f"T{target_subject_id}")
                if not osp.exists(args.output_dir_src):
                    os.system('mkdir -p ' + args.output_dir_src)
                if not osp.exists(args.output_dir_src):
                    os.mkdir(args.output_dir_src)

                args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
                args.out_file.write(print_args(args) + '\n')
                args.out_file.flush()
                train_source(args)
                test_target(args)
                args.out_file.close()
