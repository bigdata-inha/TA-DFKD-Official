import torch
from torch.utils.data import TensorDataset , DataLoader
import torch.nn as nn
from sklearn.mixture import GaussianMixture

'''
This is sample selection part for select clean samples for applying distillation and adversarial techniques.

We used Gaussian Mixture Model(GMM) for selecting clean samples as proposed in DivideMix(Li, Socher, and Hoi 2020).
'''

def split(teacher , mem_imgs , mem_lbls , target_imgs = None , selection_score =0.5):
    ce = nn.CrossEntropyLoss(reduction='none')
    teacher.eval()
    datasets = TensorDataset(mem_imgs, mem_lbls)
    loader = DataLoader(datasets , batch_size=32 , shuffle=False)

    losses = torch.tensor([])
    with torch.no_grad():
        for step, batch in enumerate(loader):
            imgs, label = batch[0], batch[1]
            label = label.type(torch.LongTensor)
            imgs , label = imgs.cuda() , label.cuda()
            outputs = teacher(imgs)
            loss = ce(outputs , label)
            losses = torch.cat([losses , loss.detach().cpu()])

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    input_loss = losses.reshape(-1,1)

    # GMM
    gmm = GaussianMixture(n_components=2 , max_iter=10 , tol=1e-2 , reg_covar=5e-4)

    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob_corr = prob[: , gmm.means_.argmin()]

    pred_corr = prob_corr > selection_score

    if target_imgs == None:
        correct_imgs = mem_imgs[pred_corr, :]
        correct_lbls = mem_lbls[pred_corr]

    else:
        correct_imgs = target_imgs[pred_corr, :]
        correct_lbls = mem_lbls[pred_corr]


    return correct_imgs , correct_lbls