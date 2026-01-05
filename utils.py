import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import numpy as np


def set_seed(seed):
    """
    set seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logger(args):
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('Prompt')
    logger.setLevel(logging.DEBUG)
    log_floder_path = f'{args.output_dir}logs/{args.runid}/'
    os.makedirs(log_floder_path, exist_ok=True)
    log_path = log_floder_path + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss,
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )
    txt_log = logging.FileHandler(log_path)

    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    print(f'log will be stored to {txt_log}')

    return logger


def creat_result_dict(args):
    """
    Create a result dictionary to store the results of each task.
    """
    I2T = np.zeros((args.num_tasks, args.num_tasks+3))
    T2I = np.zeros((args.num_tasks, args.num_tasks+3))
    result_dict = {
        'I2T' : I2T,
        'T2I' : T2I,
    }
    return result_dict


def save_result_dict(args, result_dict, csv_folder):
    """
    Save the result dictionary to a CSV file.
    The CSV file will contain the results of each task and the average results.
    """
    # Calculate the average results for each task and store them in the last column
    for i in range(args.num_tasks):
        map_sum_I2T = 0
        map_sum_T2I = 0
        for j in range(i+1):
            map_sum_I2T += result_dict['I2T'][i, j]
            map_sum_T2I += result_dict['T2I'][i, j]
        result_dict['I2T'][i, args.num_tasks] = map_sum_I2T / (i+1)
        result_dict['T2I'][i, args.num_tasks] = map_sum_T2I / (i+1)

    # Specify the CSV filepath
    os.makedirs(csv_folder, exist_ok=True)
    csv_file = csv_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss, 
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )

    # Write the numpy matrix from the dictionary to a CSV file
    with open(csv_file, 'w') as f:
        for key, matrix in result_dict.items():
            # Write the data for each matrix to a file
            f.write(key + '\n')
            np.savetxt(f, matrix, delimiter=',', fmt='%f')
    print(f'The result is already stored under {csv_file}')


def calc_map_k(qu_B, re_B, qu_L, re_L, topk=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = qu_L.shape[0]
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)   # default ascending=True
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue
        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_hammingDist(B1, B2):
    # B1: {-1,+1}^{m x q}
    # B2: {-1,+1}^{n x q}
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_neighbor(label1, label2, device):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    Sim = Sim.to(device)
    return Sim


def f_quantifyloss(x):
    """
    Quantization loss: minimize || sign(x) - x ||^2
    Inputs:
        x: [batch_size, bit]  Continuous output (before binarization)
    Outputs:
        quantization_loss: scalar
    """
    batch_size, bit = x.shape
    quantization_loss = torch.sum(torch.pow(torch.sign(x) - x, 2)) / (batch_size * bit)
    return quantization_loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        # Temperature 
        self.register_buffer('scale', torch.tensor(1.0 / temperature))

    def forward(self, image_embeddings, text_embeddings):
        # Similarity matrix
        logits_per_image = image_embeddings @ text_embeddings.T * self.scale.to(image_embeddings.dtype)
        logits_per_text  = logits_per_image.T

        B = image_embeddings.shape[0]
        labels = torch.arange(B, device=image_embeddings.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text,  labels)
        return (loss_i2t + loss_t2i) / 2


def l2_normalize(x):
    """
    L2 Normalization
    Inputs:
        x: [N, d]
    Outputs:
        x: [N, d] L2 Normalized
    """
    return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)


def f_Coarse_grained_similarity_loss(label_batch, label_all, hash_batch, hash_all):
    """
    Inputs:
        label_batch: [B, C] The labels of the current batch
        label_all:   [N, C] The labels of the entire training set
        hash_batch:  [B, L] The hash codes of the current batch (before binarization)
        hash_all:    [N, L] The hash codes of the entire training set (before binarization)
    Outputs:
        loss: scalar
    """

    B = label_batch.size(0)
    N = label_all.size(0)

    # L2 Normalization
    label_batch = l2_normalize(label_batch)
    label_all = l2_normalize(label_all)

    # calculate similarity
    sim_label = torch.matmul(label_batch, label_all.t())           # [B, N]
    sim_hash = torch.relu(torch.matmul(hash_batch, hash_all.t()))  # [B, N]

    # MSE Loss
    loss = torch.mean((sim_hash - sim_label) ** 2)

    return loss


def bit_balance_loss_pm1(Z_or_B, use_tanh=True):
    """
    Bit Balance Loss
    Inputs:
        Z_or_B: [N, L] Continuous output (before binarization) or Binarized output
        use_tanh: bool, whether Z_or_B is continuous output (before binarization)
    Outputs:
        loss: scalar
    Target: each bit has a 50% chance to be 1 or -1
    """
    H = torch.tanh(Z_or_B) if use_tanh else Z_or_B   # [-1,1] or {-1,1}
    col_mean = H.mean(dim=0)                         # [L]
    loss = (col_mean ** 2).mean()                    # -> scalar
    return loss


class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()      # 0/1

    @staticmethod
    def backward(ctx, g):
        return g                    # copy gradient