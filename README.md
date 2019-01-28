# VideoVAE

A pytorch implementation of VideoVAE: https://arxiv.org/pdf/1803.08085.pdf

Prepare Data
---
1. Download data from http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html, under the section of ``Classification Database''.
2. Make a dir, for classification data: 
```
mkdir -p data/classification
```
3. Unzip all .zip files (Walk, Run, ..., Skip) and put all of them under `data/classification`
4. run the following lines to get the preprocessed data:
```
chmod +x ./get_data.sh
./get_data.sh
```

Training
---
1. Pretrain the encoder and classifier
```
python train_attr_cls.py --use_cuda # GPU mode, or
python train_attr_cls.py            # CPU mode
```
2. Train VideoVAE
```
python main.py --use_cuda --cls_weight path/to/cls_weight # GPU mode, or
python main.py --cls_weight path/to/cls_weight            # CPU mode
```
For example,
```
python train_attr_cls.py --exp exp_test --use_cuda
```
and after running ~10 epochs, run
```
python main.py --use_cuda --cls_weight ExperimentAttr/exp_test/checkpoint/classifier_10.pth
```

Tensorboard
---
1. Run 
```
tensorboard --logdir .
```
to check the log for loss and generated sequences.

Acknowledgement:
---
The code is heavily borrowed from https://github.com/pytorch/examples/blob/master/vae/main.py

Disclaimer:
---
Since we do not know the exact parameters and model structures of the original paper (especially the settings in the structured latent space), we are not able to fully reproduce the synthesizing results from the original paper. We will update our implementation as soon as we figured out where the bug is in our code or we have any new idea that achieved better results.