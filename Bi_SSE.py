
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import DCMamba

import models, trainers, datasets, utils
from config_trAISformer import Config

from LSTM import LSTM_for_AIS
from LSTM_atten import LSTMWithAttention

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data

    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model

    model = models.Mamba_For_Ais(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls,
        INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    #### 前面的是对模型进行训练  训练结束之后 在这里进行评估
    ## Evaluation
    # ===============================
    # Load the best model 加载模型
    model.load_state_dict(torch.load(cf.ckpt_path))

    #
    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 10  # 最大长度

    model.eval()  # 将摩西改为评估模式
    l_min_errors, l_mean_errors, l_masks = [], [], []
    # 对测试集的dataloader进行加载
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        # 这里拿到的是一整个batch
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            # 拿到要预测的数据
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            # print("seqs_init-shape")

            # 拿到需要的masks 这里的mask包括了待预测值和预测值
            masks = masks[:, :max_seqlen].to(cf.device)

            # 拿到batch_size
            batchsize = seqs.shape[0]

            # 先创造一个不知道干什么的o张量
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)

            # 进行n个样本预测  每一个待预测的样本进行n_sample次预测
            for i_sample in range(cf.n_samples):
                # 拿到对应预测到的轨迹
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)

                # 这是真实值
                inputs = seqs[:, :max_seqlen, :].to(cf.device)

                # 转换为实际的地理坐标
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180

                # 将预测到的值也转化为实际的地理坐标
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180

                # print(input_coords.shape)
                # print(pred_coords.shape)
                # 计算两个坐标之间的距离
                d = utils.haversine(input_coords, pred_coords) * masks

                # 将预测到的数据放入error_ens之中
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            # Accumulation through batches 经过批次进行累积
            l_min_errors.append(error_ens.min(dim=-1))  # 拿到最小误差
            l_mean_errors.append(error_ens.mean(dim=-1))  # 拿到平均误差
            l_masks.append(masks[:, cf.init_seqlen:])  # 拿到mask

    # 拿到最小的误差
    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot 进行绘图
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6-1  # 6表示一个小时
    plt.plot(1, pred_errors[timestep], "o")
    # 这是画一条垂直线段
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12-1  # 12表示两个小时
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18-1  # 18 表示三个小时
    # 绘制误差点
    plt.plot(3, pred_errors[timestep], "o")

    # 绘制垂直线
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")

    # 绘制水平线
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")

    # 标注误差
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 24-1  # 24 表示四个个小时
    plt.plot(4, pred_errors[timestep], "o")
    plt.plot([4, 4], [0, pred_errors[timestep]], "r")
    plt.plot([0, 4], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(4.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 30-1  # 30 表示五个个小时
    plt.plot(5, pred_errors[timestep], "o")
    plt.plot([5, 5], [0, pred_errors[timestep]], "r")
    plt.plot([0, 5], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(5.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 36-1  # 36 表示六个个小时
    plt.plot(6, pred_errors[timestep], "o")
    plt.plot([6, 6], [0, pred_errors[timestep]], "r")
    plt.plot([0, 6], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(6.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 42-1  # 42 表示七个个小时
    plt.plot(7, pred_errors[timestep], "o")
    plt.plot([7, 7], [0, pred_errors[timestep]], "r")
    plt.plot([0, 7], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(7.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 48-1  # 48 表示八个个小时
    plt.plot(8, pred_errors[timestep], "o")
    plt.plot([8, 8], [0, pred_errors[timestep]], "r")
    plt.plot([0, 8], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(8.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 54-1  # 54 表示九个个小时
    plt.plot(9, pred_errors[timestep], "o")
    plt.plot([9, 9], [0, pred_errors[timestep]], "r")
    plt.plot([0, 9], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(9.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 60-1  # 60 表示十个个小时
    plt.plot(10, pred_errors[timestep], "o")
    plt.plot([10, 10], [0, pred_errors[timestep]], "r")
    plt.plot([0, 10], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(10.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 50])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")

    # Yeah, done!!!
