#!/usr/bin/env/python3

import csv
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from models import TasNet

# Define training procedure
class Separation(sb.Brain,TasNet):
    def compute_forward(self, mix, targets, stage, noise=None):
        # 解析双通道输入
        mix1, mix2 = mix
        mix1, mix1_lens = mix1  # mix1 形状: (batch, time)（单通道）
        mix2, mix2_lens = mix2  # mix2 形状: (batch, time)（单通道）
        mix1, mix2 = mix1.to(self.device), mix2.to(self.device)
        mix_lens = mix1_lens
        
        # 关键修正：合并为 (batch, 2, time)，通道维度在 dim=1
        # 先在最后一维增加通道维度，再转置为 (batch, 2, time)
        mix1 = mix1.unsqueeze(1)  # 形状: (batch, 1, time)
        mix2 = mix2.unsqueeze(1)  # 形状: (batch, 1, time)
        mix = torch.cat([mix1, mix2], dim=1)  # 合并通道，形状: (batch, 2, time)
        
        # # 验证形状是否正确（batch, 2, time）
        # assert mix.dim() == 3, f"mix 维度错误，预期3维，实际{mix.dim()}维"
        # assert mix.size(1) == 2, f"合并后通道数应为2，实际{mix.size(1)}"
        # print(f"修正后 mix 形状: {mix.shape}")  # 应输出 (batch, 2, time)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # 数据增强（若有）
        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb:
                    # 保存原始随机种子状态
                    original_state = torch.random.get_rng_state()
                    
                    # 分别处理两个通道的速度扰动，但使用相同的随机种子
                    mix_channels = []
                    targets_channels = []
                    
                    for ch in range(mix.size(1)):  # 通道维度
                        # 重置随机种子，确保两个通道使用相同的速度扰动参数
                        torch.random.set_rng_state(original_state)
                        
                        # 提取当前通道的混合信号
                        mix_ch = mix[:, ch]  # [batch, time]
                        
                        # 重建当前通道的targets
                        targets_ch = targets.clone()
                        
                        # 应用速度扰动（与原单通道逻辑一致）
                        mix_ch, targets_ch = self.add_speed_perturb(targets_ch, mix_lens)
                        
                        # 保存处理后的结果
                        mix_channels.append(mix_ch.unsqueeze(1))  # 添加通道维度
                        targets_channels.append(targets_ch)
                    
                    # 恢复原始随机状态
                    torch.random.set_rng_state(original_state)
                    
                    # 合并两个通道的混合信号
                    mix = torch.cat(mix_channels, dim=1)  # [batch, 2, time]
                    
                    # 选择其中一个通道的targets即可
                    targets = targets_channels[0]

                if self.hparams.use_wavedrop:
                    # 对双通道分别应用WaveDrop
                    mix_wavedrop = []
                    for ch in range(mix.size(1)):
                        dropped_ch = self.hparams.drop_chunk(mix[:, ch], mix_lens)
                        dropped_ch = self.hparams.drop_freq(dropped_ch)
                        mix_wavedrop.append(dropped_ch.unsqueeze(1))
                    
                    mix = torch.cat(mix_wavedrop, dim=1)
                    
                if self.hparams.limit_training_signal_len:
                    # 同步截断输入和目标
                    mix, targets = self.cut_signals(mix, targets)

        # # 验证增强后的形状
        # assert mix.size(1) == 2, f"数据增强后通道数应为2，实际{mix.size(1)}"
        # print(f"最终输入到 Encoder 的 mix 形状: {mix.shape}")

        # 模型前向计算（分离过程）
        mix_w = self.hparams.Encoder(mix)  # 编码器处理双通道输入
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)  # 复制到多源
        sep_h = mix_w * est_mask  # 应用掩码

        # 解码得到预测音频
        est_source = torch.cat(
            [self.hparams.Decoder(sep_h[i]).unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,  # 形状: (B, T_est, num_spks)
        )

        # 关键：强制对齐预测与目标的长度和形状
        # 1. 对齐时间长度（以targets为准）
        T_target = targets.size(1)  # 目标音频的时间长度
        T_est = est_source.size(1)  # 预测音频的时间长度
        if T_est > T_target:
            est_source = est_source[:, :T_target, :]  # 截断过长的预测
        else:
            est_source = F.pad(est_source, (0, 0, 0, T_target - T_est))  # 补零过短的预测

        # 2. 确保维度数量一致（均为3维：B, T, num_spks）
        assert est_source.dim() == targets.dim(), f"预测维度{est_source.dim()}与目标维度{targets.dim()}不匹配"
        assert est_source.size(-1) == targets.size(-1), f"预测源数量{est_source.size(-1)}与目标源数量{targets.size(-1)}不匹配"

        return est_source, targets

    def apply_fixed_speed_perturb(self, targets, lengths, speed_idx):
        """使用固定的速度索引应用速度扰动"""
        # 保存原始的随机种子状态
        original_state = torch.random.get_rng_state()
        
        # 设置固定的随机种子，确保两个通道使用相同的速度变化
        torch.random.manual_seed(speed_idx)  # 使用速度索引作为种子
        
        # 应用速度扰动（这将根据预配置的speeds列表随机选择一个速度）
        perturbed_targets = []
        
        for i in range(targets.size(-1)):  # 遍历每个声源
            source = targets[..., i]  # [batch, time]
            perturbed = self.hparams.speed_perturb(source)
            perturbed_targets.append(perturbed.unsqueeze(-1))
        
        # 恢复原始的随机种子状态
        torch.random.set_rng_state(original_state)
        
        # 重新组合所有声源
        perturbed_targets = torch.cat(perturbed_targets, dim=-1)
        
        # 计算新的长度（这需要根据实际选择的速度来计算）
        # 由于我们无法直接获取实际使用的速度，这里使用平均速度变化
        avg_speed = sum(self.hparams.speed_changes) / len(self.hparams.speed_changes) / 100.0
        new_lengths = (lengths.float() * avg_speed).long()
        
        return None, perturbed_targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        # 解析双通道混合音频
        mix1 = batch.mix1_sig  # (音频张量, 长度)
        mix2 = batch.mix2_sig
        mixture = (mix1, mix2)  # 打包为元组传入模型

        # 构建目标音频（4个源）
        targets = [batch.s1_sig, batch.s2_sig, batch.s3_sig, batch.s4_sig]
        # # 转换为形状：(batch, time, num_spks)
        # targets = torch.cat(
        #     [t[0].unsqueeze(-1) for t in targets],  # t[0]是音频张量，t[1]是长度
        #     dim=-1
        # ).to(self.device)

        # 前向计算与损失计算
        with self.training_ctx:
            predictions, targets = self.compute_forward(mixture, targets, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, targets)  # 此时形状已匹配

            # hard threshold the easy dataitems
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss = loss[loss > th]
                if loss.nelement() > 0:
                    loss = loss.mean()
            else:
                loss = loss.mean()

        if loss.nelement() > 0 and loss < self.hparams.loss_upper_lim:
            self.scaler.scale(loss).backward()
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0.0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mix1 = batch.mix1_sig  # 第一个通道
        mix2 = batch.mix2_sig  # 第二个通道
        mixture = (mix1, mix2)  # 打包成元组传入模型
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.num_spks == 4:
            targets.append(batch.s3_sig)
            targets.append(batch.s4_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_checkpoint(
                meta={"si-snr": stage_loss},  # 仅保存当前 ckpt，不删除旧的
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    if self.hparams.num_spks == 4:
                        targets.append(batch.s3_sig)
                        targets.append(batch.s4_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create output folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. 定义数据集
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. 定义音频加载管道（重点修改部分）
    # 加载第一个单通道混合音频
    @sb.utils.data_pipeline.takes("mix1_wav")
    @sb.utils.data_pipeline.provides("mix1_sig")
    def audio_pipeline_mix1(mix1_wav):
        mix1_sig = sb.dataio.dataio.read_audio(mix1_wav)
        return mix1_sig

    # 加载第二个单通道混合音频
    @sb.utils.data_pipeline.takes("mix2_wav")
    @sb.utils.data_pipeline.provides("mix2_sig")
    def audio_pipeline_mix2(mix2_wav):
        mix2_sig = sb.dataio.dataio.read_audio(mix2_wav)
        return mix2_sig

    # 加载四个源音频（保持不变）
    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    @sb.utils.data_pipeline.takes("s3_wav")
    @sb.utils.data_pipeline.provides("s3_sig")
    def audio_pipeline_s3(s3_wav):
        s3_sig = sb.dataio.dataio.read_audio(s3_wav)
        return s3_sig

    @sb.utils.data_pipeline.takes("s4_wav")
    @sb.utils.data_pipeline.provides("s4_sig")
    def audio_pipeline_s4(s4_wav):
        s4_sig = sb.dataio.dataio.read_audio(s4_wav)
        return s4_sig

    # 3. 将管道添加到数据集
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix1)  # 新增：添加第一个混合音频管道
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix2)  # 新增：添加第二个混合音频管道
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s4)

    # 4. 设置输出键（重点修改：删除 mix_sig，添加 mix1_sig 和 mix2_sig）
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "mix1_sig", "mix2_sig", "s1_sig", "s2_sig", "s3_sig", "s4_sig"],
    )

    return train_data, valid_data, test_data

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = get_logger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from preprocess_dynamic_mixing import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Collecting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
