from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from networks import resnet_tiny, gan, deepinversion
import torchvision.transforms as transforms
import random
import numpy as np
import torchvision
import Sample_Selection
import warnings
from torchvision import datasets

warnings.filterwarnings('ignore')


class Trasform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    def __call__(self, x, reverse=False):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return normalize(x, mean, std, reverse=reverse)


def JS_divergence(outputs, outputs_student):
    T = 3.0
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps = 0.0
    loss_verifier_cig = 0.5 * nn.KLDivLoss()(torch.log(P + eps), M) + 0.5 * nn.KLDivLoss()(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
    return loss_verifier_cig


Augmentation = Trasform([
    # Training view
    transforms.Compose([
        Normalizer()
    ]),
    # Testing view
    transforms.Compose([
        Normalizer()
    ])
])

def main(seed):

    parser = argparse.ArgumentParser(description='TINYIMAGENET_MAIN')
    parser.add_argument('--method', type=str, default='sample_selection', choices=['basic', 'sample_selection'])
    parser.add_argument('--dataset', type=str, default='TinyImageNet', choices=['TinyImageNet'])
    parser.add_argument('--seed', type=int, default=seed, metavar='S')
    parser.add_argument('--data', type=str, default='./data/')
    parser.add_argument('--input_dir', type=str, default='./Pretrained_Teachers/')
    parser.add_argument('--teacher_model', type=str, default='/TINY_7550.pt')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_S', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
    parser.add_argument('--epoch_iters', type=int, default=36)
    parser.add_argument('--BNS', type=float, default=10)
    parser.add_argument('--R_l2', type=float, default=6e-3)
    parser.add_argument('--R_tv', type=float, default=1.5e-5)
    parser.add_argument('--selection_score', default='0.5', type=float, help='selection score')
    parser.add_argument('--gpu1', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--gpu2', default='-1', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    print(args)
    available_gpu = ""
    if args.gpu1 > -1:
        available_gpu += str(args.gpu1) + ","
    if args.gpu2 > -1:
        available_gpu += str(args.gpu2) + ","

    available_gpu = available_gpu[:-1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    val_dir = './data/TinyImageNet/tiny-imagenet-200/val/images'
    val_data = datasets.ImageFolder(val_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]))
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    teacher_path = args.input_dir + args.dataset + args.teacher_model
    print('teacher path : {}'.format(teacher_path))
    teacher = resnet_tiny.ResNet34().cuda()
    teacher.load_state_dict(torch.load(teacher_path))
    student = resnet_tiny.ResNet18().cuda()
    generator = gan.Generator(nz=args.latent_dim, img_size=64).cuda()

    teacher = nn.DataParallel(teacher)
    student = nn.DataParallel(student)
    generator = nn.DataParallel(generator)

    teacher.eval()

    # BNS
    loss_r_feature_layers = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(deepinversion.DeepInversionFeatureHook(module))

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
    scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_S, T_max=args.n_epochs)

    acc_list = []
    for epoch in range(args.n_epochs):
        scheduler_S.step()

        # training step
        teacher.eval()
        student.train()
        generator.train()

        for i in range(args.epoch_iters):
            ##################
            # Student Training
            ##################
            for k in range(10):
                optimizer_S.zero_grad()

                if args.method == 'sample_selection':

                    correct_imgs = None

                    while True:
                        noises = torch.randn(args.batch_size, args.latent_dim).cuda()
                        fake = generator(noises).detach()
                        fake, fake_for_test = Augmentation(fake)
                        outputs_T, features_T = teacher(fake_for_test, out_feature=True)
                        pred = outputs_T.data.max(1)[1]
                        if correct_imgs == None:
                            correct_imgs, correct_lbls = Sample_Selection.split(teacher, fake_for_test, pred,
                                                                                target_imgs=fake,
                                                                                selection_score=args.selection_score)

                        else:
                            pre_correct_imgs, pre_correct_lbls = correct_imgs, correct_lbls
                            correct_imgs, correct_lbls = Sample_Selection.split(teacher, fake_for_test, pred,
                                                                                target_imgs=fake,
                                                                                selection_score=args.selection_score)

                            if (pre_correct_imgs.shape[0] + correct_imgs.shape[0]) <= args.batch_size:
                                correct_imgs = torch.cat([pre_correct_imgs, correct_imgs], dim=0)
                                correct_lbls = torch.cat([pre_correct_lbls, correct_lbls], dim=0)
                            else:
                                left_size = args.batch_size - pre_correct_imgs.shape[0]
                                rand = random.sample(range(0, correct_imgs.shape[0]), left_size)
                                correct_imgs = correct_imgs[rand, :]
                                correct_lbls = correct_lbls[rand]
                                correct_imgs = torch.cat([pre_correct_imgs, correct_imgs], dim=0)
                                correct_lbls = torch.cat([pre_correct_lbls, correct_lbls], dim=0)
                                break

                    t_logit = teacher(correct_imgs)
                    s_logit = student(correct_imgs)
                    loss_DE = F.l1_loss(s_logit, t_logit)


                else:
                    noises = torch.randn(args.batch_size, args.latent_dim).cuda()
                    fake = generator(noises).detach()
                    fake, fake_for_test = Augmentation(fake)
                    t_logit = teacher(fake)
                    s_logit = student(fake)
                    loss_DE = F.l1_loss(s_logit, t_logit)

                loss_S = loss_DE
                loss_S.backward()
                optimizer_S.step()

            ####################
            # Generator Training
            ####################
            noises = torch.randn(args.batch_size, args.latent_dim).cuda()
            optimizer_G.zero_grad()
            fake = generator(noises)
            fake, fake_for_test = Augmentation(fake)

            if args.method == 'sample_selection':
                outputs_T, features_T = teacher(fake_for_test, out_feature=True)
                pred = outputs_T.data.max(1)[1]
                correct_imgs, correct_lbls = Sample_Selection.split(teacher, fake_for_test, pred, target_imgs=fake,
                                                                    selection_score=args.selection_score)
                t_logit_corr = teacher(correct_imgs)
                s_logit_corr = student(correct_imgs)
                loss_Adv = JS_divergence(s_logit_corr, t_logit_corr)

            else:
                t_logit = teacher(fake)
                s_logit = student(fake)
                loss_Adv = JS_divergence(s_logit, t_logit.detach())

            diff1 = fake[:, :, :, :-1] - fake[:, :, :, 1:]
            diff2 = fake[:, :, :-1, :] - fake[:, :, 1:, :]
            diff3 = fake[:, :, 1:, :-1] - fake[:, :, :-1, 1:]
            diff4 = fake[:, :, :-1, :-1] - fake[:, :, 1:, 1:]
            loss_R_tv = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss_BNS = sum([mod.r_feature.cuda() for mod in loss_r_feature_layers])
            loss_R_l2 = torch.norm(fake, 2)
            loss_G = (args.R_tv * loss_R_tv.cuda()) + (args.BNS * loss_BNS.cuda()) + (
                    args.R_l2 * loss_R_l2.cuda()) + loss_Adv.cuda()

            loss_G.backward()
            optimizer_G.step()

            if i == 1:
                print("[Epoch %d/%d] [loss_kd: %f] [loss_G : %f] [loss_adv : %f]" % (
                    epoch, args.n_epochs, loss_DE.item(), loss_G.item(), loss_Adv.item()))

        # validataion step
        student.eval()
        generator.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_data_loader):
                data, target = data.cuda(), target.cuda()
                output = student(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_data_loader.dataset)

        print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(epoch, test_loss, correct,
                                                                                            len(val_data_loader.dataset),
                                                                                            100. * correct / len(
                                                                                                val_data_loader.dataset)))
        acc = correct / len(val_data_loader.dataset)

        acc_list.append(acc)

    print("datasets : " , args.dataset, "methods:" ,args.method, " end")
    print("Best Acc : {:.4f}".format(max(acc_list)))
    print("acc_list : {} ".format(acc_list))

    with open('./Results/{}_results.txt'.format(args.dataset), 'a') as file:
        file.write(str(args))
        file.write("\nBest Acc : {:.4f}\n".format(max(acc_list)))
        file.write("Accuracy_list : " + str(acc_list) + "\n \n")

seed = random.randint(0, 10000)
main(seed)




