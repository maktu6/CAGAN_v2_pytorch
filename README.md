# CAGAN_v2_pytorch
A PyTorch implementation of [CAGAN](https://arxiv.org/abs/1709.04695) combined with [StackGAN-v2](https://arxiv.org/abs/1710.10916)  
Code based on the [original implementation in keras](https://github.com/shaoanlu/Conditional-Analogy-GAN-keras)  
The following repositories were also referenced:   
- [https://github.com/hanzhanggit/StackGAN-v2](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  

## Prerequisites
- Python 3.6
- PyTorch 0.4.0
- Visdom

## Description
### Data
The dataset is just same as [here](https://github.com/shaoanlu/Conditional-Analogy-GAN-keras/blob/master/README.md#description) and the folder of dataset is structured as follows: 
```
├── data
    ├── imgs_test
    │   ├── 1
    │   └── 5
    └── imgs_train
        ├── 1
        └── 5

```

### Train
- Open a visdom server  
```bash
python -m visdom.server
```
- Train a model  
```bash
python main.py
```

### Test
Use trained model to generate images and calculate IS score  
```bash
python main.py --mode test --model_dir logs/origin/model_weight
```

