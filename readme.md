# [Enhancing Real-Time Cross-Subject EEG Decoding via Pseudo-Label Tuning and Adaptation]

## About
This repository provides the implementation of the PLTA framework corresponding to the manuscript *"Enhancing Real-Time Cross-Subject EEG Decoding via Pseudo-Label Tuning and Adaptation"*, currently under peer review.  

The code is shared for the purpose of facilitating reproducibility during the review process. A fully documented release will be made available upon official publication of the paper. 

## Prerequisites
To use the repository, install the conda dependencies using the following command:
```bash
conda env create -f environment.yml
conda activate plta
```

## To reproduce:
The entire adaptation pipeline (download data → train source model → test-time adaptation) can be executed using the following shell scripts.

For more detailed hyperparameter configuration (e.g., learning rate, optimizer, batch size), please refer to the argparse arguments defined in the python script.

### 1. Download datasets:
To download the datasets used in our experiment, run:
```bash
python download_data.py
```

Supported datasets are: BNCI2014001, BNCI2014002, BNCI2015001

### 2. Train the source model
To train only the source-domain model, run:
```bash
python train_source.py
```

For more detailed hyperparameter configuration (e.g., learning rate, optimizer, batch size), please refer to the argparse arguments defined in the python script.

The trained source model along with the training log will be saved under 
```
./ckpt/{dataset_name}
```

### 3. Test time adaptation
To perform online adaptation during testing, run:
```bash
python test_time.py --method "your_method_name"
```

Supported methods are (check the test_time.py script for the corresponding name used in the command):
- Source (without adaptation)
- RA-MDRM
- CORAL
- T3A
- PredBN
- PredBN+
- Tent
- COTTA
- LAME
- SAR
- TIPI
- EATA
- T-TIME
- PLTA (our work)

To modify the hyperparameters of a specific method, edit the corresponding YAML file: 
```
./cfg/MotorImagery/{method_name}.yaml
```

Adaptation results and logs will be saved under under 
```
./test-time-evaluation/{dataset_name} 
```

## Contact
For questions or requests, please contact: linzh23@m.fudan.edu.cn

## Acknowledgements
- The base framework of this repository is built upon: [Benchmark-TTA](https://github.com/sh923s/Benchmark-TTA/tree/master)
- Part of the data processing code and the T-TIME implementation: [T-TIME](https://github.com/sh923s/DeepTransferEEG/tree/main)