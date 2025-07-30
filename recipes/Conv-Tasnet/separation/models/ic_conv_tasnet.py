# conv_tasnet_ic.py
import torch
from torch.autograd import Variable
from .config import config
from . import modules

# 1. 编码器模块：封装原self.encoder的Conv1d
class Encoder(torch.nn.Module):
    def __init__(self,
                 mic_num=config.model_mic_num,
                 enc_dim=config.model_enc_dim,
                 win=int(config.sample_rate * config.model_win / 1000),  # 转换为采样点
                 stride=config.model_win // 2
                 ):
        super(Encoder, self).__init__()
        self.mic_num = mic_num
        self.enc_dim = enc_dim
        self.win = win
        self.stride = stride
        self.encoder = torch.nn.Conv1d(
            mic_num,
            enc_dim,
            win,
            bias=False,
            stride=stride
        )
        # 复用原初始化逻辑
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入：(B, mic_num, T)，输出：(B, enc_dim, L)
        return self.encoder(x)


# 2. 掩码网络模块：封装原self.TCN（TCN）
class MaskNet(torch.nn.Module):
    def __init__(self,
                 mic_num=config.model_mic_num,
                 ch_dim=config.model_ch_dim,
                 input_dim=config.model_enc_dim,
                 output_dim=config.model_enc_dim * config.model_num_spk,  # enc_dim*num_spk
                 BN_dim=config.model_feature_dim,
                 hidden_dim=config.model_feature_dim * 4,  # 保持原逻辑（feature_dim*4）
                 layer=config.model_layer,
                 stack=config.model_stack,
                 kernel=config.model_kernel,
                 causal=config.model_causal):
        super(MaskNet, self).__init__()
        self.TCN = modules.TCN(
            mic_num=mic_num,
            ch_dim=ch_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            BN_dim=BN_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
            causal=causal
        )
        # 复用原TCN的 receptive_field 属性（用于后续处理）
        self.receptive_field = self.TCN.receptive_field

    def forward(self, x):
        # 输入：(B, C, N, L)，输出：(B, C, enc_dim*num_spk, L)（与原TCN输出一致）
        return self.TCN(x)


# 3. 解码器模块：封装原self.decoder的ConvTranspose1d
class Decoder(torch.nn.Module):
    def __init__(self,
                 in_channels=config.model_enc_dim * config.model_num_spk,  # enc_dim*num_spk
                 out_channels=config.model_num_spk,  # 输出num_spk路音频
                 win=int(config.sample_rate * config.model_win / 1000),  # 与Encoder保持一致的win计算方式
                 stride=None,  # stride为win的一半（原逻辑）
                 groups=config.model_num_spk):  # 按num_spk分组
        super(Decoder, self).__init__()
        if stride is None:
            stride = win // 2  # 内部计算stride
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            win,
            bias=False,
            stride=stride,
            groups=groups
        )
        # 复用原初始化逻辑
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入：(B, enc_dim*num_spk, L)，输出：(B, num_spk, T_out)
        return self.decoder(x)
class TasNet(torch.nn.Module):
    def __init__(self):
        super(TasNet, self).__init__()
        self.encoder = Encoder()  # 自动从config获取参数
        self.masknet = MaskNet()  # 自动从config获取参数
        self.decoder = Decoder()  # 自动从config获取参数
        self.receptive_field = self.masknet.receptive_field  # 保持感受野引用
    # ---------- 工具方法（与原代码一致，无需修改） ----------
    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    # ---------- 前向传播（逻辑不变，仅调用拆分后的模块） ----------
    def forward(self, input):
        output, rest = self.pad_signal(input)          # (B, 2, T)

        B, C, T = output.shape                         # C=2
        # 1. 编码：(B, 2, T) → (B, 512, L)
        enc = self.encoder(output)
        # 2. 处理为TCN输入格式 (B, C, N, L)，其中C=2（麦克风数），N=512（特征维度）
        enc_4d = enc.unsqueeze(1).repeat(1, 2, 1, 1)   # (B, 1, 512, L) → 重复为(B, 2, 512, L)
        # 3. 掩码网络输出掩码：(B, 2, 4×512, L) → 经sigmoid激活
        masks_4d = torch.sigmoid(self.masknet(enc_4d))

        # 4. 取参考麦克风（第0路）的掩码，与编码特征相乘
        masks = masks_4d.squeeze(1)                    # (B, 4×512, L)
        masks = masks.view(B, self.num_spk, self.enc_dim, -1)  # (B, 4, 512, L)
        ref_enc = enc.unsqueeze(1)                     # (B, 1, 512, L)
        masked = ref_enc * masks                       # (B, 4, 512, L)

        # 5. 解码：(B, 4×512, L) → (B, 4, T_out)
        wav = self.decoder(masked.view(B, 4 * self.enc_dim, -1))
        # 移除填充，恢复原始长度
        wav = wav[:, :, self.stride:-(rest + self.stride)].contiguous()
        return wav.view(B, 4, -1)