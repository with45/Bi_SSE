# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Configuration flags to run the main script.
"""

import os
import pickle
import torch
from torch import nn
from pytorch.dcformer.configuration_dcformer import DCFormerConfig


class Config():
    train_model = 'Mamba2'
    retrain = True
    tb_log = False
    device = torch.device("cuda")
    #     device = torch.device("cpu")

    max_epochs = 20
    batch_size = 16
    n_samples = 10

    init_seqlen = 30
    max_seqlen = 120
    min_seqlen = 36

    dataset_name = "ct_dma"

    if dataset_name == "ct_dma":  # ==============================

        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and
        # dlon
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    # ===========================================================================
    # Model and sampling flags
    mode = "pos"  # "pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1",
    # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode = "pos_vicinity"  # "pos", "pos_vicinity" or "velo"
    top_k = 10  # int or None
    r_vicinity = 40  # int

    # Blur flags
    # ===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    # Data flags
    # ===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"

    # model parameters
    # ===================================================
    n_head = 8
    n_layer = 10
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.3
    resid_pdrop = 0.5
    attn_pdrop = 0.1

    # optimization parameters
    # ===================================================
    learning_rate = 1e-4  # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 512 * 20  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    num_workers = 0  # for DataLoader

    # mamba
    # n_embed不用管
    mamba_d_intermediate = 1024
    mamba_ssm_cfg = {'layer': 'bi-Mamba2'}
    mamba_attn_layer_idx = [7,15,23]
    # mamba_attn_layer_idx = [7,15,23]
    # mamba_attn_layer_idx = [11,23]
    # for i in range(24):
    #     mamba_attn_layer_idx.append(i)
    mamba_attn_cfg = {'num_heads': 8, 'num_heads_kv': None, 'head_dim': None, 'mlp_dim': 1024,
                      'qkv_proj_bias': True, 'out_proj_bias': True, 'softmax_scale': None, 'causal': True,
                      'd_conv': 3, 'rotary_emb_base': 10000.0, 'rotary_emb_interleaved': False,
                      }
    mamba_norm_epsilon = 1e-6
    mamba_rms_norm = True
    mamba_residual_in_fp32 = True
    fused_add_norm = True
    mamba_dtype = None
    deepth = 10

    se_reduction = 16

    # for i in range(deepth):
    #     if i % 2 == 1:
    #         mamba_attn_layer_idx.append(i)

    # LSTM和LSTM_attention
    lstm_input_size = 4
    lstm_num_layers = 7
    lstm_hidden_size = n_embd
    lstm_output_size = full_size
    filename = f"{dataset_name}" \
               + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}" \
               + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}" \
               + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}" \
               + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}" \
               + f"-head-{n_head}-{n_layer}" \
               + f"-bs-{batch_size}" \
               + f"-lr-{learning_rate}" \
               + f"-seqlen-{init_seqlen}-{max_seqlen}" \
               + f'train_model-{train_model}' \
               + f'deepth-{deepth}'
    savedir = "./results/" + filename + "/"

    ckpt_path = os.path.join(savedir, "model.pt")
