

import math
import logging
import pdb
import copy

import torch
import torch.nn as nn
from ray.tune.examples.pbt_dcgan_mnist.common import batch_size
from torch.nn import functional as F

from functools import partial

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from biMamba2 import bi_Mamba2

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None




logger = logging.getLogger(__name__)

class FSEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FSEAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        y = F.relu(self.fc1(input))
        y = self.sigmoid(self.fc2(y))

        return input * y



class SEMHA(nn.Module):
    def __init__(self, d_model, se_reduction, attn_cfg, **factory_kwargs):
        super(SEMHA, self).__init__()
        self.d_model = d_model
        self.SE_Block = FSEAttention(d_model, reduction=se_reduction)
        self.M_HA = MHA(d_model, **attn_cfg, **factory_kwargs)

    def forward(self, x, inference_params=None):
        y = self.SE_Block(x)
        y = self.M_HA(y)
        return y


class create_block_mha(nn.Module):
    def __init__(self, d_model, num_heads=8, casual=False, dropout=0.1, mlp_hidden_dim=1024):
        super(create_block_mha, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.casual = casual

        self.attn = torch.nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, d_model),
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def _generate_causal_mask(self, seq_len):
        """
        Generates a causal mask to prevent attending to future tokens.
        The mask is a square matrix of shape (seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, x, inference_params=None):
        """
        Arguments:
            x: (batch_size, seq_len, hidden_dim) - Input tensor
        """
        bs, seq_len, _ = x.shape

        # Create causal mask if needed
        attn_mask = None
        if self.casual:
            attn_mask = self._generate_causal_mask(seq_len).to(x.device)

        # Apply multi-head attention (Q, K, V all equal for self-attention)
        attn_output, attn_weights = self.attn(x, x, x, attn_mask=attn_mask)  # Shape: (batch_size, seq_len, embed_dim)

        # Apply MLP (optional)
        mlp_output = self.mlp(attn_output)

        # Apply output projection
        output = self.out_proj(mlp_output)

        return output


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        se_reduction=16,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2","bi-Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        # mixer_cls = partial(
        #     Mamba2 if ssm_layer == "Mamba2" else Mamba,
        #     layer_idx=layer_idx,
        #     **ssm_cfg,
        #     **factory_kwargs
        # )
        if ssm_layer == "Mamba1":
            mixer_cls = partial(
                Mamba,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        elif ssm_layer == "Mamba2":
            mixer_cls = partial(
                Mamba2,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        else:
            mixer_cls = partial(
                bi_Mamba2,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
    else:  #### MHA在这里 写一个模型来代替MHA
        # mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
        # mixer_cls = partial(create_block_mha, num_heads=attn_cfg['num_heads'], casual=attn_cfg['causal'],
        #                     mlp_hidden_dim=attn_cfg['mlp_dim'])
        mixer_cls = partial(SEMHA, se_reduction=se_reduction, attn_cfg=attn_cfg, **factory_kwargs)

    ### 上面是进行处理注意力层或者是mamba层的，下面是处理归一化层以及
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class Mamba_For_Ais(nn.Module):
    """
    An improved transformer-based Mamba model for predicting AIS datasets
    """

    def __init__(self, config, partition_model=None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd

        ### mamba 部分
        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.mamba_residual_in_fp32

        self.register_buffer(
            "att_sizes",
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes",
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))

        if hasattr(config, "partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model

        if hasattr(config, "blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1 / 3)
            else:
                self.blur_module = None

        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer--->mamba
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        factory_kwargs = {'device': config.device, 'dtype': config.mamba_dtype}
        self.blocks = nn.ModuleList(
            [
                create_block(
                    d_model=config.n_embd,
                    d_intermediate=config.mamba_d_intermediate,
                    ssm_cfg=config.mamba_ssm_cfg,
                    # ssm模型配置
                    attn_layer_idx=config.mamba_attn_layer_idx,
                    attn_cfg=config.mamba_attn_cfg,
                    norm_epsilon=config.mamba_norm_epsilon,
                    rms_norm=config.mamba_rms_norm,
                    residual_in_fp32=config.mamba_residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,

                    **factory_kwargs
                )
                for i in range(config.deepth)
            ]
        )
        self.norm_f = (nn.LayerNorm if not config.mamba_rms_norm else RMSNorm)(
            config.n_embd,
            eps=config.mamba_norm_epsilon,
            **factory_kwargs
        )

        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)  # Classification head

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.

        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            model: currenly only supports "uniform".

        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x * self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):

            idxs = (x * self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:, :, :2])
            #             pdb.set_trace()
            idxs[:, :, 0] = torch.round(lat_ids.reshape((bs, seqlen))).long()
            idxs[:, :, 1] = torch.round(lon_ids.reshape((bs, seqlen))).long()
            return idxs, idxs_uniform

    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False,
                inference_params=None):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:],
                otherwise inputs = x.
        Returns:
            logits, loss
        """

        if self.mode in ("mlp_pos", "mlp",):
            idxs, idxs_uniform = x, x  # use the real-values of x.
        else:
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)

        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()
            targets_uniform = idxs_uniform[:, 1:, :].contiguous()
            inputs_real = x[:, :-1, :].contiguous()
            targets_real = x[:, 1:, :].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:, :, 0])  # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings), dim=-1)

        position_embeddings = self.pos_emb[:, :seqlen,
                              :]  # each position maps to a (learnable) vector (1, seqlen, n_embd)
        hidden_states = self.drop(token_embeddings + position_embeddings)

        ### 改进的地方
        # fea = self.blocks(fea)
        # fea = self.ln_f(fea)  # (bs, seqlen, n_embd)
        residual = None
        for layer in self.blocks:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:  # --
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        logits = self.head(hidden_states)  # (bs, seqlen, full_size) or (bs, seqlen, n_embd)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size),
                                       targets[:, :, 2].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                       targets[:, :, 3].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                       targets[:, :, 0].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                       targets[:, :, 1].view(-1),
                                       reduction="none").view(batchsize, seqlen)

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                # 进行平滑的次数
                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1, 1, self.lat_size)).reshape(
                        lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1, 1, self.lon_size)).reshape(
                        lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1, 1, self.sog_size)).reshape(
                        sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1, 1, self.cog_size)).reshape(
                        cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:, :, 0].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:, :, 1].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:, :, 2].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:, :, 3].view(-1),
                                                  reduction="none").view(batchsize, seqlen)

                    lat_loss += self.blur_loss_w * blurred_lat_loss
                    lon_loss += self.blur_loss_w * blurred_lon_loss
                    sog_loss += self.blur_loss_w * blurred_sog_loss
                    cog_loss += self.blur_loss_w * blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)

            if masks is not None:
                loss = (loss * masks).sum(dim=1) / masks.sum(dim=1)

            loss = loss.mean()

        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss

