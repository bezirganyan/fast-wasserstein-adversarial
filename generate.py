import argparse
import os
import time

import numpy as np
from torch import nn
from torchvision import models
import torch
import torch.multiprocessing as mp

from frank_wolfe import FrankWolfe


def craft(rank, dataloader_list, dataloader_path, result_path, eps_list):
    fparam = dataloader_list[0].split('_')
    dataloader = torch.load(os.path.join(dataloader_path, f'{fparam[0]}_{rank}_dtl.pt'))
    torch.save(dataloader, f'{result_path}/fw_{rank}_dtl.pt')
    device = torch.device(f'cuda:{rank}')
    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    # normalize = lambda x: (x - mu) / std
    unnormalize = lambda x: x * std + mu

    net = models.vgg16(pretrained=True).cuda(rank)
    net.eval()

    for param in net.parameters():
        param.requires_grad = False
    for eps in eps_list:
        print(f'{rank} ==> Craft attack with eps = {eps}')
        print('________________________________________')
        correct = 0
        total = 0

        frank_wolfe = FrankWolfe(predict=lambda x: net(x),  # X needs to be normalized
                                 loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                 eps=eps,
                                 kernel_size=5,
                                 nb_iter=20,
                                 entrp_gamma=1e-6,
                                 dual_max_iter=50,
                                 grad_tol=1e-5,
                                 int_tol=1e-5,
                                 device=device,
                                 postprocess=True,
                                 verbose=True)
        adv_samples = []
        perturbations = []
        durations = []


        for batch_idx, (cln_data, y) in enumerate(dataloader):
            cln_data, y = unnormalize(cln_data.to(device)), y.to(device)
            start_time = time.time()
            adv_data = frank_wolfe.perturb(cln_data, y)
            duration = time.time() - start_time
            durations.append(duration)
            assert adv_data is not None

            adv_samples.append(adv_data.detach().cpu())
            perturbation = adv_data - cln_data
            perturbations.append(perturbation.detach().cpu())

            with torch.no_grad():
                output = net(adv_data)

            prediction = output.max(dim=1)[1]

            correct += (prediction == y).sum().item()
            total += y.size(0)

            print("****************************************************************")
            print("batch idx: {:4d} num_batch: {:4d} acc: {:.3f}% ({:5d} / {:5d})".format(batch_idx + 1,
                                                                                          len(dataloader),
                                                                                          100. * correct / total,
                                                                                          correct,
                                                                                          total))
            print("****************************************************************")

            if frank_wolfe.__class__.__name__ == "Sinkhorn" and frank_wolfe.overflow is True:
                break

        adv_samples = torch.cat(adv_samples, dim=0)
        perturbations = torch.cat(perturbations, dim=0)
        adv_path = f'{result_path}/fw_wasserstein_{eps:.5f}_{rank}.pt'
        prt_path = f'{result_path}/fw_wasserstein_{eps:.5f}_{rank}_prt.pt'
        durations_path = f'{result_path}/fw_wasserstein_{eps:.5f}_{rank}_dur.pt'
        torch.save(adv_samples, adv_path)
        torch.save(perturbations, prt_path)
        torch.save(durations, durations_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dtl_folder', type=str, default='../adversarial_arena/results')
    parser.add_argument('--result_path', type=str, default='results_wass')
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--eps_start', type=float, default=0.05)
    parser.add_argument('--eps_end', type=float, default=1.0)
    parser.add_argument('--eps_count', type=int, default=15)

    args = parser.parse_args()

    file_list = os.listdir(args.dtl_folder)
    dataloader_list = [f for f in file_list if f.endswith('dtl.pt')]
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    eps_list = np.linspace(args.eps_start, args.eps_end, args.eps_count)
    mp.spawn(fn=craft, args=(dataloader_list, args.dtl_folder, args.result_path, eps_list), nprocs=args.n_procs)