# Teacher-Agnostic Data-Free Knowledge Distillation(TA-DFKD)

Teacher as a Lenient Expert: Teacher-Agnostic Data-Free Knowledge Distillation

## Abstract 
Data-free knowledge distillation (DFKD) aims to distill pretrained knowledge to a student model with the help of a generator without using original data. In such data-free scenarios, achieving stable performance of DFKD is essential due to the unavailability of validation data. Unfortunately, this paper has discovered that existing DFKD methods are quite sensitive to different teacher models, occasionally showing catastrophic failures of distillation, even when using well-trained teacher models. Our observation is that the generator in DFKD is not always guaranteed to produce precise yet diverse samples using the existing representative strategy of minimizing both class-prior and adversarial losses. Through our empirical study, we focus on the fact that class-prior not only decreases the diversity of generated samples, but also cannot completely address the problem of generating unexpectedly low-quality samples depending on teacher models. In this paper, we propose the teacher-agnostic data-free knowledge distillation (TA-DFKD) method, with the goal of more robust and stable performance regardless of teacher models. Our basic idea is to assign the teacher model a lenient expert role for evaluating samples, rather than a strict supervisor that enforces its class-prior on the generator. Specifically, we design a sample selection approach that takes only clean samples verified by the teacher model without imposing restrictions on the power of generating diverse samples. Through extensive experiments, we show that our method successfully achieves both robustness and training stability across various teacher models, while outperforming the existing DFKD methods.

## Ready 
First, create a directory named "Pretrained_Teachers" in this project, where various teacher models are stored.</br>
After, create subdirectories named CIFAR10, CIFAR100, and TinyImageNet within it, and download and save the pretrained teacher models from the provided link.</br>
( Download link : https://drive.google.com/drive/folders/1nk_6dNXJdsGTlTR76YuCT2WLpCK9II8W?usp=sharing )

## run CIFAR-10 , CIFAR-100 and TinyImageNet 

### CIFAR-10
```
python TA-DFKD-main-CIFAR.py --dataset CIFAR10 --teacher_model /CIFAR10_9546.pt --n_epochs 200 --batch_size 1024
```

### CIFAR-10
```
python TA-DFKD-main-CIFAR.py --dataset CIFAR100 --teacher_model /CIFAR10_7501.pt --n_epochs 500 --batch_size 1024
```

### TinyImageNet
Before, running TinyImageNet, you need to download the publicly available TinyImageNet dataset in ./data.
```
python TA-DFKD-main-TinyImageNet.py --dataset TinyImageNet --teacher_model /TINY_7550.pt --n_epochs 500 --batch_size 1024
```


