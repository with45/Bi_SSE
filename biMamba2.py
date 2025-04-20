import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class bi_Mamba2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=128,
            d_conv=4,
            conv_init=None,
            expand=2,
            headdim=64,
            d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            ngroups=1,
            A_init_range=(1, 16),
            D_has_hdim=False,
            rmsnorm=True,
            norm_before_gate=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            process_group=None,
            sequence_parallel=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]  输入投影纬度
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        # 投影层
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # 这是conv1d第一个，因为要进行双向SSD，那么就要进行两次SSD
        self.conv1d_1 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # 这是第二个conv1d
        self.conv1d_2 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # 激活函数
        self.act = nn.SiLU()

        # 初始化delta   这是第一个delta
        dt1 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt1 = torch.clamp(dt1, min=dt_init_floor)

        inv_dt = dt1 + torch.log(-torch.expm1(-dt1))
        self.dt1_bias = nn.Parameter(inv_dt)
        self.dt1_bias._no_weight_decay = True

        # 初始化第二个delta
        dt2 = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt2 = torch.clamp(dt2, min=dt_init_floor)

        inv_dt2 = dt2 + torch.log(-torch.expm1(-dt2))
        self.dt2_bias = nn.Parameter(inv_dt2)
        self.dt2_bias._no_weight_decay = True

        ### ******
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]

        # 初始化A 第一个A
        A1 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log1 = torch.log(A1).to(dtype=dtype)
        self.A_log1 = nn.Parameter(A_log1)
        self.A_log1._no_weight_decay = True

        # 初始化第二个A
        A2 = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log2 = torch.log(A2).to(dtype=dtype)
        self.A_log2 = nn.Parameter(A_log2)
        self.A_log2._no_weight_decay = True

        # 初始化D  第一个D
        self.D1 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D1._no_weight_decay = True

        # 初始化D,第二个D
        self.D2 = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D2._no_weight_decay = True

        # norm参数
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            # 这里用到的是这个
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group,
                                              sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
        self.end_out = nn.Linear(self.d_model, self.d_model)
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None):
        """
        输入的u为（B,T,C）
        输出为（B,T,C）
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        # 这里其实也用不上
        conv_state, ssm_state = None, None

        # 这里实际上用不到 吊用莫得
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        # 拿到两个原始数据和翻转之后的数据
        zxbcdt = self.in_proj(u)
        # 正向往前
        zxbcdt_f = zxbcdt
        # 后向往后
        zxbcdt_b = torch.flip(zxbcdt, dims=[-1])

        A1 = -torch.exp(self.A_log1.float())
        A2 = -torch.exp(self.A_log2.float())

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and inference_params is None:
            # print("tttt")
            out_f = mamba_split_conv1d_scan_combined(
                zxbcdt_f,
                rearrange(self.conv1d_1.weight, "d 1 w -> d w"),
                self.conv1d_1.bias,
                self.dt1_bias,
                A1,
                D=rearrange(self.D1, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D1,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            # 这个可以忽略
            if seqlen_og is not None:
                out_f = rearrange(out_f, "b l d -> (b l) d")

            # 这个也可以忽略
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out_f = reduce_fn(out_f, self.process_group)

            out_b = mamba_split_conv1d_scan_combined(
                zxbcdt_b,
                rearrange(self.conv1d_2.weight, "d 1 w -> d w"),
                self.conv1d_2.bias,
                self.dt2_bias,
                A2,
                D=rearrange(self.D2, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D2,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            # 下面两个if都可以忽略不计
            if seqlen_og is not None:
                out_b = rearrange(out_b, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out_b = reduce_fn(out_b, self.process_group)
            end_ans = self.end_out(self.w1 * out_f + self.w2 * out_b)
            return end_ans
        else:
            # print("fffff")
            d_mlp = (zxbcdt_f.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0_f, x0_f, z_f, xBC_f, dt_f = torch.split(
                zxbcdt_f,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            z0_b, x0_b, z_b, xBC_b, dt_b = torch.split(
                zxbcdt_b,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            # 如果存在卷积状态 conv_state，对其更新为当前块的状态。//这里也没用上
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t_f = rearrange(xBC_f, "b l d -> b d l")
                xBC_t_b = rearrange(xBC_b, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t_f, (self.d_conv - xBC_t_f.shape[-1], 0)))  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC_f = self.act(
                    self.conv1d_1(xBC_f.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
                xBC_b = self.act(
                    self.conv1d_2(xBC_b.transpose(1, 2)).transpose(1, 2)
                )
            else:
                xBC_f = causal_conv1d_fn(
                    xBC_f.transpose(1, 2),
                    rearrange(self.conv1d_1.weight, "d 1 w -> d w"),
                    bias=self.conv1d_1.bias,
                    activation=self.activation,
                ).transpose(1, 2)
                xBC_b = causal_conv1d_fn(
                    xBC_b.transpose(1, 2),
                    rearrange(self.conv1d_2.weight, "d 1 w -> d w"),
                    bias=self.conv1d_2.bias,
                    activation=self.activation,
                )

            # 状态更新与扫描
            x_f, B_f, C_f = torch.split(xBC_f, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                        dim=-1)
            x_b, B_b, C_b = torch.split(xBC_b, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                        dim=-1)

            y_f = mamba_chunk_scan_combined(
                rearrange(x_f, "b l (h p) -> b l h p", p=self.headdim),
                dt_f,
                A1,
                rearrange(B_f, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C_f, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D1, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D1,
                z=rearrange(z_f, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt1_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            y_b = mamba_chunk_scan_combined(
                rearrange(x_b, "b l (h p) -> b l h p", p=self.headdim),
                dt_b,
                A2,
                rearrange(B_b, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C_b, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D2, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D2,
                z=rearrange(z_b, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt2_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            # 这里也是用不上
            if ssm_state is not None:
                y, last_state = y_f
                ssm_state.copy_(last_state)

            # 将y进行转变
            y_f = rearrange(y_f, "b l h p -> b l (h p)")
            y_b = rearrange(y_b, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y_f = self.norm(y_f, z_f)
                y_b = self.norm(y_b, z_b)
            if d_mlp > 0:
                y_f = torch.cat([F.silu(z0_f) * x0_f, y_f], dim=-1)
                y_b = torch.cat([F.silu(z0_b) * x0_b, y_b], dim=-1)
            if seqlen_og is not None:
                y_f = rearrange(y_f, "b l d -> (b l) d")
                y_b = rearrange(y_b, "b l d -> (b l) d")
            out = self.out_proj(y_f + y_b)
            return out
