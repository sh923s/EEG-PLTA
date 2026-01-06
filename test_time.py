import logging
import os
import time
from datetime import datetime
import numpy as np
import argparse
import torch
import yaml
from src.methods import *
from src.models.load_model import load_model
import pyriemann
from src.utils import get_metrics, get_online_accuracy, get_args, print_args, load_dset_config, fix_seed, log_init
from src.data import riemann_align
from sklearn.metrics import cohen_kappa_score, f1_score, balanced_accuracy_score, confusion_matrix
# from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
logger = logging.getLogger(__name__)            


def evaluate(args):
    if args.method != "riemann":
        base_model = load_model(args, checkpoint_dir=args.ckpt_path)
        # start evaluation
        _, adapt_dset = load_dataset(args, args.adapt_session, align="no_align")
        if args.adapt_session == args.test_session:
            test_dset = adapt_dset
        else:
            _, test_dset = load_dataset(args, args.test_session, align="no_align")
        
        logger.info(f"Setting up test-time adaptation method: {args.method}")
        model_config = args.model_config
        if args.method == "source" or args.method == "coral":  # BN--0
            model, param_names = setup_source(base_model)
        elif args.method == "t3a":
            model, param_names = setup_t3a(base_model, model_config)
        elif args.method == "predBN":  # BN--1
            model, param_names = setup_test_norm(base_model, model_config)
        elif args.method == "predBN+":  # BN--0.1
            model, param_names = setup_alpha_norm(base_model, model_config)
        elif args.method == "tent":
            model, param_names = setup_tent(base_model, model_config)
        elif args.method == "cotta":
            model, param_names = setup_cotta(base_model, model_config)
        elif args.method == "lame":
            model, param_names = setup_lame(base_model, model_config)
        elif args.method == "eata":
            model, param_names = setup_eata(base_model, model_config, test_dset, args.class_num, args.update_batch_size)
        elif args.method == "sar":
            model = setup_sar(base_model, model_config, args.class_num)
        elif args.method == "plta":
            model = setup_plta(base_model, model_config, args.class_num)
        elif args.method == "tipi":
            model = setup_tipi(base_model, model_config)
        elif args.method == "ttime":
            model, param_names = setup_ttime(base_model, model_config)
        else:
            raise ValueError(f"Adaptation method '{args.method}' is not supported!")

        if args.adapt_session == args.test_session:
            dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.update_batch_size, shuffle=args.shuffle_test_dset, num_workers=args.worker)
            acc, kappa, f1, bacc, cm, y_test, y_pred = get_metrics(model, dataloader, adapt=True, align=args.align)
        else:
            adapt_loader = torch.utils.data.DataLoader(adapt_dset, batch_size=args.update_batch_size, shuffle=args.shuffle_test_dset, num_workers=args.worker)
            test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.update_batch_size, shuffle=args.shuffle_test_dset, num_workers=args.worker)
            _, _, _, _, _, _, _ = get_metrics(model, adapt_loader, adapt=True, align=args.align)
            acc, kappa, f1, bacc, cm, y_test, y_pred = get_metrics(model, test_loader, adapt=args.use_test_to_adapt, align=args.align)
    else:
        train_dset, test_dset = load_dataset(args, session=args.test_session, align="no_align")
        X_train = train_dset.tensors[0].numpy()
        y_train = train_dset.tensors[1].numpy()
        X_test = test_dset.tensors[0].numpy()
        y_test = test_dset.tensors[1].numpy()
        cov_train_aligned, cov_test_aligned = riemann_align(X_train, X_test)
        clf = pyriemann.classification.MDM()
        clf.fit(cov_train_aligned, y_train)
        y_pred = clf.predict(cov_test_aligned)
        acc = np.mean(y_pred == y_test)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
    logger.info(f"accuracy % [T{args.target_subject_id}][#samples={len(test_dset)}]: {acc:.2%}")
    return acc, kappa, f1, bacc, cm, y_test, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0,1', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--domain', type=str, default='MotorImagery', choices=['MotorImagery', 'EmotionRecognition'])
    parser.add_argument('--dset_root', type=str, default='./data') 
    parser.add_argument('--model_cfg_root', type=str, default='./cfg')
    parser.add_argument('--backbone', type=str, default='eegnet', help="eegnet", choices=['eegnet', 'basenet'])
    parser.add_argument('--num_rep', type=int, default=10, help="number of repetitions")
    parser.add_argument('--smooth', type=float, default=0) # cross entropy label smooth
    parser.add_argument('--output', type=str, default='./ckpt')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--method', type=str, default='plta', choices=['source', 'riemann', 'coral', 't3a', 'predBN', 'predBN+', 'tent', 'cotta', 'lame', 'sar', 'plta', 'tipi', 'eata', 'ttime'])
    parser.add_argument('--scenario', type=str, default='normal', choices=['normal', 'noise',
                                                                          'channel_shuffle',
                                                                          'amplitude_scaling'])
    parser.add_argument('--snr', type=float, default=5)
    parser.add_argument('--noise_channel_ratio', type=float, default=1)
    parser.add_argument('--shuffle_channel_ratio', type=float, default=0.3)
    parser.add_argument('--maximum_amplitude_scaling', type=float, default=2)
    parser.add_argument('--align', type=str, default='ea', choices=['no_align', 'ea', 'coral'])
    parser.add_argument('--train_session', type=int, default=1, help="train session")
    parser.add_argument('--adapt_session', type=int, default=1, help="adapt session")
    parser.add_argument('--test_session', type=int, default=1, help="test session")
    parser.add_argument('--use_test_to_adapt', type=int, default=1, help="whether to use test session data to adapt the model")
    parser.add_argument('--multi_class', type=int, default=0, help="whether it is a multi-class classification task")

    args = parser.parse_args()
    if args.method == 'riemann':
        args.align = 'no_align'
    if args.method == 'coral':
        args.align = 'coral' 
    if args.adapt_session == args.test_session:
        args.use_test_to_adapt = 1
    # set gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.domain == 'MotorImagery':
        if args.multi_class == 0:
            dset_list = ['BNCI2014001', 'BNCI2014002','BNCI2015001']
        else:
            dset_list = ['BNCI2014001-4']
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))
    
    for dset in dset_list:
        args.dset = dset 
    
        # load dataset config
        load_dset_config(args)
        
        # load model config
        with open(os.path.join(args.model_cfg_root, args.domain, f"{args.method}.yaml"), 'r') as f:
            model_config = yaml.safe_load(f)
        args.model_config = model_config
        
        now = datetime.now()
        adapt_dset_name = ''
        if args.adapt_session == args.test_session or not args.use_test_to_adapt:
            adapt_dset_name = args.adapt_session
        else:
            adapt_dset_name = f"{args.adapt_session}+{args.test_session}"
        
        if args.scenario != 'noise':
            args.output_dir = f'test-time-evaluation/{args.domain}/{args.scenario}/{args.dset}/TrainSession{args.train_session}/AdaptSession{adapt_dset_name}/TestSession{args.test_session}/{args.backbone}/{args.align}/{args.method}/{now.strftime("%Y%m%d-%H%M%S")}'
        else:
            args.output_dir = f'test-time-evaluation/{args.domain}/{args.scenario}_{args.snr}dB/{args.dset}/TrainSession{args.train_session}/AdaptSession{args.adapt_session}/TestSession{args.test_session}/{args.backbone}/{args.align}/{args.method}/{now.strftime("%Y%m%d-%H%M%S")}'
        
        try:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            log_init(os.path.join(args.output_dir, 'log.txt'))
            logger.info(print_args(args))
            accs = np.zeros((args.num_rep, args.subject_num))
            kappas = np.zeros((args.num_rep, args.subject_num))
            f1s = np.zeros((args.num_rep, args.subject_num))
            baccs = np.zeros((args.num_rep, args.subject_num))
            cms = np.zeros((args.num_rep, args.subject_num, args.class_num, args.class_num))
            y_trues = [[None for _ in range(args.subject_num)] for _ in range(args.num_rep)]
            y_preds = [[None for _ in range(args.subject_num)] for _ in range(args.num_rep)]
            start_time = time.time()
            for i in range(args.num_rep):
                seed = i
                # fix random seed
                fix_seed(seed)
                # iterate over target subjects
                for target_subject_id in range(args.subject_num):
                    args.target_subject_id = target_subject_id
                    args.ckpt_path = os.path.join(args.output, args.domain, args.dset, f"Session{args.train_session}", args.backbone, args.align, f"Seed{seed}", f"T{target_subject_id}")
                    logger.info(f"Seed: {seed}, Target Subject: {target_subject_id}")
                    try:
                        acc, kappa, f1, bacc, cm, y_true, y_pred = evaluate(args)
                    except Exception as e:
                        logger.error(f"Evaluation failed for target subject {target_subject_id}, seed {seed}, with error: {e}")
                        acc = np.nan
                        kappa = np.nan
                        f1 = np.nan
                        bacc = np.nan
                        cm = np.full((args.class_num, args.class_num), np.nan)
                        y_true = np.array([])
                        y_pred = np.array([])
                    accs[i, target_subject_id] = acc
                    kappas[i, target_subject_id] = kappa
                    f1s[i, target_subject_id] = f1
                    baccs[i, target_subject_id] = bacc
                    cms[i, target_subject_id] = cm
                    y_trues[i][target_subject_id] = y_true
                    y_preds[i][target_subject_id] = y_pred
            logger.info('final result')
            logger.info(f"total mean accuracy: {np.mean(accs):.2%}")
            end_time = time.time()
            run_time = end_time - start_time
            logger.info(f"total run time: {run_time}s")
            
            # save metrics as a mat file
            import scipy.io as sio
            sio.savemat(os.path.join(args.output_dir, 'metrics.mat'), {'acc': accs, 'kappa': kappas, 'f1': f1s, 'bacc': baccs, 'cm': cms, 'y_true': np.array(y_trues), 'y_pred': np.array(y_preds)})
            
            
        except Exception as e:
            logger.error(e)
            raise e
