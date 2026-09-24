"""M4 cleaned-data comparison: Kaldi five-channel prosody, original fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder

from pitchfusion.models.run_paper_reproduction import (
    DEFAULT_KALDI_EXECUTABLE, ROOT, OfficialPitchFusion, atomic_csv, atomic_json,
    atomic_torch_save, build_clean_stratified_split, decode_wav, evaluate,
    load_audio_bytes, prediction_frame, resolve, run_kaldi_feature_batch,
    seed_everything, sha256_file, train,
)

CONFIG = "configs/models/m3_paper_reproduction_clean_v1.json"
SPLIT = "data/manifests/paper_reproduction_clean_seed391.csv"
FEATURE_ROOT = "data/interim/features/m4_kaldi_multi_prosody_clean_v1"
ARTIFACT_ROOT = "artifacts/paper_reproduction/m4_multi_prosody"
REPORT_ROOT = "reports/paper_reproduction"
CHANNELS = ("log_f0", "nccf", "log_energy", "delta_log_f0", "delta_log_energy")


def frame_energy(audio: np.ndarray, frames: int) -> np.ndarray:
    # Kaldi pitch can retain one final frame whose 25 ms window extends past
    # the waveform. Pad only that unavailable tail, while using Kaldi's exact
    # frame count as the alignment authority for all five M4 channels.
    required_samples = (frames - 1) * 160 + 400
    if len(audio) < required_samples:
        audio = np.pad(audio, (0, required_samples - len(audio)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160][:frames]
    if len(windows) != frames:
        raise ValueError(f"Could not align {len(windows)} energy frames to Kaldi {frames}")
    rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    return np.log(np.maximum(rms, 1e-8)).astype(np.float32)


def matrix_to_prosody(matrix: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"Expected Kaldi [frames,2], received {matrix.shape}")
    nccf = matrix[:, 0].astype(np.float32)
    log_f0 = np.log(np.maximum(matrix[:, 1], 1e-6)).astype(np.float32)
    energy = frame_energy(audio, len(matrix))
    values = np.stack(
        (log_f0, nccf, energy, np.gradient(log_f0), np.gradient(energy)), axis=0
    ).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite M4 prosody feature")
    return values


def prepare_features(manifest: pd.DataFrame, audio_bytes: list[bytes], root: Path,
                     kaldi: Path, config: dict[str, Any], force: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows = list(manifest.itertuples(index=False))
    missing = [r for r in rows if force or not (root / f"{r.sample_id}.npy").is_file()]
    started = time.perf_counter()
    for offset in range(0, len(missing), 128):
        batch = missing[offset:offset + 128]
        matrices = run_kaldi_feature_batch(batch, audio_bytes, kaldi, root / ".work", config["pitch"])
        for row in batch:
            audio = decode_wav(audio_bytes[int(row.source_row_index)])
            values = matrix_to_prosody(matrices[str(row.sample_id)], audio)
            target = root / f"{row.sample_id}.npy"
            temporary = target.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            os.replace(temporary, target)
        done = min(offset + len(batch), len(missing))
        print(f"[m4-prosody] {done}/{len(missing)} | elapsed={time.perf_counter()-started:.1f}s", flush=True)
    sums = np.zeros(5, np.float64); squares = np.zeros(5, np.float64); count = 0
    index = []
    for row in rows:
        path = root / f"{row.sample_id}.npy"; values = np.load(path, allow_pickle=False)
        if row.split == "train":
            sums += values.sum(1); squares += np.square(values.astype(np.float64)).sum(1); count += values.shape[1]
        index.append({"sample_id": row.sample_id, "split": row.split, "artifact_path": path.relative_to(ROOT).as_posix(), "frames": values.shape[1], "sha256": sha256_file(path)})
    mean = sums / count; std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-12))
    stats = {"channels": list(CHANNELS), "mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "validation_and_test_used_for_fit": False}
    atomic_csv(pd.DataFrame(index), root / "index.csv"); atomic_json(stats, root / "normalization_train.json")
    return stats


class M4Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, audio: list[bytes], root: Path,
                 label_to_id: dict[str, int], stats: dict[str, Any]):
        self.frame=frame.reset_index(drop=True); self.audio=audio; self.root=root; self.labels=label_to_id
        self.mean=np.asarray(stats["mean"],np.float32)[:,None]; self.std=np.asarray(stats["std"],np.float32)[:,None]
    def __len__(self): return len(self.frame)
    def __getitem__(self, index: int):
        row=self.frame.iloc[index]; waveform=decode_wav(self.audio[int(row.source_row_index)])
        raw=np.load(self.root/f"{row.sample_id}.npy",allow_pickle=False); raw=(raw-self.mean)/self.std
        target=np.empty((5,len(waveform)),np.float32)
        for channel in range(5):
            target[channel]=np.interp(np.linspace(0,len(waveform),len(waveform)),np.linspace(0,raw.shape[1],raw.shape[1]),raw[channel])
        return waveform,target,self.labels[str(row.emotion)]


def collate(batch):
    width=max(len(x[0]) for x in batch); audio=torch.zeros(len(batch),width); prosody=torch.zeros(len(batch),5,width); labels=torch.empty(len(batch),dtype=torch.long)
    for i,(a,p,y) in enumerate(batch): audio[i,:len(a)]=torch.from_numpy(a.copy()); prosody[i,:,:len(a)]=torch.from_numpy(p.copy()); labels[i]=y
    return audio,prosody,labels


class MultiChannelFeatureEncoder(nn.Module):
    """Safely mix five standardized cues, then reuse M3's exact pitch CNN."""

    def __init__(self, config: Any, channels: int = 5):
        super().__init__()
        self.input_adapter = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        self.feature_encoder = Wav2Vec2FeatureEncoder(config)
        self.reset_input_adapter()

    def reset_input_adapter(self) -> None:
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.weight[0, 0, 0] = 1.0
            self.input_adapter.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixed = self.input_adapter(values).squeeze(1)
        # Wav2Vec2FeatureEncoder.forward() assigns requires_grad on its input
        # while training. The adapter output is not a leaf tensor, and modern
        # PyTorch correctly rejects that assignment. Running the same official
        # convolution stack directly preserves the full autograd graph.
        hidden_states = mixed[:, None]
        for conv_layer in self.feature_encoder.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class M4Model(OfficialPitchFusion):
    def __init__(self, config: Any):
        super().__init__(config)
        self.pitch_encoder = MultiChannelFeatureEncoder(config)
        self.post_init()
        self.pitch_encoder.reset_input_adapter()


class DirectFiveChannelFeatureEncoder(nn.Module):
    """Preserve the trained log-F0 convolution and add four separate inputs."""

    def __init__(self, config: Any):
        super().__init__()
        self.feature_encoder = Wav2Vec2FeatureEncoder(config)
        first_conv = self.feature_encoder.conv_layers[0].conv
        self.auxiliary_conv = nn.Conv1d(
            4,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            bias=False,
        )
        self.register_buffer("base_scale", torch.ones(1))
        self.register_buffer("base_bias", torch.zeros(1))
        self.reset_auxiliary()

    def reset_auxiliary(self) -> None:
        with torch.no_grad():
            self.auxiliary_conv.weight.zero_()

    def initialize_from_adapter(
        self, adapter_weight: torch.Tensor, adapter_bias: torch.Tensor
    ) -> None:
        weights = adapter_weight.detach().flatten()
        if len(weights) != 5:
            raise ValueError(f"Expected five adapter weights, received {len(weights)}")
        with torch.no_grad():
            self.base_scale.copy_(weights[0].reshape_as(self.base_scale))
            self.base_bias.copy_(adapter_bias.detach().reshape_as(self.base_bias))
            first_weight = self.feature_encoder.conv_layers[0].conv.weight
            self.auxiliary_conv.weight.copy_(
                first_weight * weights[1:].reshape(1, 4, 1)
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first_layer = self.feature_encoder.conv_layers[0]
        base_values = values[:, :1] * self.base_scale + self.base_bias
        hidden_states = first_layer.conv(base_values)
        hidden_states = hidden_states + self.auxiliary_conv(values[:, 1:])
        hidden_states = first_layer.layer_norm(hidden_states)
        hidden_states = first_layer.activation(hidden_states)
        for conv_layer in self.feature_encoder.conv_layers[1:]:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class DirectFiveChannelM4Model(OfficialPitchFusion):
    def __init__(self, config: Any):
        super().__init__(config)
        self.pitch_encoder = DirectFiveChannelFeatureEncoder(config)
        self.post_init()
        self.pitch_encoder.reset_auxiliary()


class UnifiedFiveChannelFeatureEncoder(nn.Module):
    """Run all five standardized prosody streams through one 5-channel CNN."""

    def __init__(self, config: Any):
        super().__init__()
        self.feature_encoder = Wav2Vec2FeatureEncoder(config)
        original = self.feature_encoder.conv_layers[0].conv
        self.feature_encoder.conv_layers[0].conv = nn.Conv1d(
            5,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            bias=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden_states = values
        for conv_layer in self.feature_encoder.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class UnifiedFiveChannelM4Model(OfficialPitchFusion):
    def __init__(self, config: Any):
        super().__init__(config)
        self.pitch_encoder = UnifiedFiveChannelFeatureEncoder(config)
        self.post_init()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--device",choices=("cuda","cpu"),default="cuda"); parser.add_argument("--kaldi-executable",default=DEFAULT_KALDI_EXECUTABLE); parser.add_argument("--variant",choices=("v1","regularized","direct5","full5"),default="v1"); parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--prepare-only",action="store_true"); parser.add_argument("--force-features",action="store_true"); args=parser.parse_args()
    config_path=resolve(CONFIG); config=json.loads(config_path.read_text(encoding="utf-8"))
    if args.variant == "regularized":
        config["training"] = {**config["training"], "weight_decay": 0.01}
    if args.variant == "direct5":
        config["training"] = {
            **config["training"],
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        }
    if args.variant == "full5":
        config["training"] = {
            **config["training"],
            "epochs": 10,
            "learning_rate": 5e-5,
            "weight_decay": 0.0,
            "warmup_steps": 500,
        }
    manifest=build_clean_stratified_split(resolve(config["data"]["clean_manifest"]),resolve(SPLIT),resolve("reports/paper_reproduction/split_report.json"),config,force=False)
    audio=load_audio_bytes(resolve(config["data"]["raw_parquet"])); feature_root=resolve(FEATURE_ROOT); kaldi=resolve(args.kaldi_executable)
    stats=prepare_features(manifest,audio,feature_root,kaldi,config,args.force_features)
    if args.prepare_only: return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device=torch.device(args.device); seed_everything(int(config["seed"])); labels={x:i for i,x in enumerate(config["emotion_order"])}
    parts={x:manifest[manifest.split==x].reset_index(drop=True) for x in ("train","validation","test")}; loaders={x:DataLoader(M4Dataset(parts[x],audio,feature_root,labels,stats),batch_size=2,shuffle=x=="train",num_workers=0,pin_memory=device.type=="cuda",collate_fn=collate) for x in parts}
    source=resolve(config["backbone"]["local_snapshot"])
    direct5 = args.variant == "direct5"
    full5 = args.variant == "full5"
    model_class = UnifiedFiveChannelM4Model if full5 else (DirectFiveChannelM4Model if direct5 else M4Model)
    model=model_class.from_pretrained(str(source),local_files_only=True,attention_dropout=.1,hidden_dropout=.1,feat_proj_dropout=.1,final_dropout=.1,mask_time_prob=.05,layerdrop=.1,num_labels=4,classifier_proj_size=256)
    if direct5 or full5:
        baseline_path = resolve("artifacts/paper_reproduction/m4_multi_prosody/best_validation_ua.pt")
        if not baseline_path.is_file():
            raise FileNotFoundError(f"M4-v1 best checkpoint is required: {baseline_path}")
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        adapter_weight = baseline["model_state"]["pitch_encoder.input_adapter.weight"]
        adapter_bias = baseline["model_state"]["pitch_encoder.input_adapter.bias"]
        baseline_state = {
            name: value for name, value in baseline["model_state"].items()
            if not name.startswith("pitch_encoder.input_adapter.")
        }
        if full5:
            first_weight_name = "pitch_encoder.feature_encoder.conv_layers.0.conv.weight"
            original_first_weight = baseline_state[first_weight_name]
            baseline_state[first_weight_name] = (
                original_first_weight * adapter_weight.flatten().reshape(1, 5, 1)
            )
            baseline_state["pitch_encoder.feature_encoder.conv_layers.0.conv.bias"] = (
                original_first_weight.sum(dim=(1, 2)) * adapter_bias.flatten()[0]
            )
        incompatible = model.load_state_dict(baseline_state, strict=False)
        expected_missing = set() if full5 else {
            "pitch_encoder.auxiliary_conv.weight",
            "pitch_encoder.base_scale",
            "pitch_encoder.base_bias",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected direct5 transplant mismatch: {incompatible}"
            )
        if direct5:
            model.pitch_encoder.initialize_from_adapter(adapter_weight, adapter_bias)
        for parameter in model.parameters():
            parameter.requires_grad = False
        if direct5:
            for parameter in model.pitch_encoder.auxiliary_conv.parameters():
                parameter.requires_grad = True
        else:
            trainable_modules = (
                model.pitch_encoder.feature_encoder.conv_layers[0].conv,
                model.pitch_projection,
                model.attn_a_p,
                model.attn_p_a,
                model.self_attn,
                model.projector,
                model.classifier,
            )
            for module in trainable_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        del baseline
    else:
        # `from_pretrained` initializes missing adapter weights after __init__.
        model.pitch_encoder.reset_input_adapter()
        model.freeze_feature_extractor()
    regularized = args.variant == "regularized"
    early_stopping_patience = 3 if (direct5 or full5) else (5 if regularized else None)
    tracked_parameters = ("pitch_encoder.input_adapter.weight",) if regularized else ()
    optimizer_groups = None
    if full5:
        first_conv_parameters = list(model.pitch_encoder.feature_encoder.conv_layers[0].conv.parameters())
        first_conv_ids = {id(parameter) for parameter in first_conv_parameters}
        adaptation_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in first_conv_ids
        ]
        optimizer_groups = [
            {"name": "first_conv_5ch", "params": first_conv_parameters, "lr": 5e-5, "weight_decay": 0.0},
            {"name": "fusion_classifier", "params": adaptation_parameters, "lr": 3e-6, "weight_decay": 0.0},
        ]
    if args.preflight_only:
        model.to(device).train()
        if direct5:
            initial_parameter = model.pitch_encoder.auxiliary_conv.weight.detach().cpu().clone()
            if not torch.isfinite(initial_parameter).all():
                raise RuntimeError("Direct5 auxiliary convolution initialization is non-finite")
        elif full5:
            initial_parameter = model.pitch_encoder.feature_encoder.conv_layers[0].conv.weight.detach().cpu().clone()
            if not torch.isfinite(initial_parameter).all():
                raise RuntimeError("Full5 first convolution initialization is non-finite")
        else:
            initial_parameter = model.pitch_encoder.input_adapter.weight.detach().cpu().clone()
            expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
            if not torch.equal(initial_parameter.flatten(), expected):
                raise RuntimeError(f"Unexpected M4 adapter initialization: {initial_parameter.flatten().tolist()}")
        a,p,y=next(iter(loaders["train"])); a_device=a[:1].to(device); p_device=p[:1].to(device)
        baseline_max_abs_difference = None
        if direct5 or full5:
            model.eval()
            with torch.no_grad():
                direct_baseline_logits = model(a_device, p_device).logits
            reference = M4Model.from_pretrained(str(source), local_files_only=True, num_labels=4, classifier_proj_size=256)
            reference_checkpoint = torch.load(resolve("artifacts/paper_reproduction/m4_multi_prosody/best_validation_ua.pt"), map_location="cpu", weights_only=True)
            reference.load_state_dict(reference_checkpoint["model_state"])
            reference.to(device).eval()
            with torch.no_grad():
                reference_logits = reference(a_device, p_device).logits
            baseline_max_abs_difference = float((direct_baseline_logits-reference_logits).abs().max().cpu())
            if baseline_max_abs_difference > 1e-5:
                raise RuntimeError(f"Five-channel transplant changed baseline logits by {baseline_max_abs_difference}")
            del reference, reference_checkpoint, reference_logits, direct_baseline_logits
            torch.cuda.empty_cache()
            model.train()
        out=model(a_device,p_device).logits
        if not torch.isfinite(out).all():
            raise RuntimeError("M4 preflight produced non-finite logits")
        loss = nn.CrossEntropyLoss()(out, y[:1].to(device)); loss.backward()
        gradients = [v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("M4 preflight produced missing or non-finite gradients")
        parameter_step_delta_norm = None
        if regularized or direct5 or full5:
            smoke_optimizer = torch.optim.AdamW(
                optimizer_groups or (
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ),
                lr=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
            )
            smoke_optimizer.step()
            updated_parameter = (
                model.pitch_encoder.auxiliary_conv.weight if direct5
                else (
                    model.pitch_encoder.feature_encoder.conv_layers[0].conv.weight
                    if full5 else model.pitch_encoder.input_adapter.weight
                )
            ).detach().cpu()
            parameter_step_delta_norm = float((updated_parameter - initial_parameter).norm())
            if not torch.isfinite(updated_parameter).all() or parameter_step_delta_norm == 0.0:
                raise RuntimeError("M4 added parameter did not update safely in optimizer smoke step")
        print(json.dumps({"status":"PASS","model":f"m4_multi_prosody_{args.variant}","baseline_max_abs_logit_difference":baseline_max_abs_difference,"added_parameter_initial_norm":float(initial_parameter.norm()),"added_parameter_smoke_step_delta_norm":parameter_step_delta_norm,"finite_logits":True,"finite_gradients":True,"smoke_loss":float(loss.detach().cpu()),"logits_shape":list(out.shape),"parameters":sum(v.numel() for v in model.parameters()),"trainable_parameters":sum(v.numel() for v in model.parameters() if v.requires_grad),"training":config["training"]},indent=2)); return
    artifact=resolve("artifacts/paper_reproduction/m4_multi_prosody_full5" if full5 else ("artifacts/paper_reproduction/m4_multi_prosody_direct5" if direct5 else ("artifacts/paper_reproduction/m4_multi_prosody_regularized" if regularized else ARTIFACT_ROOT)))
    model_name="m4_multi_prosody_full5" if full5 else ("m4_multi_prosody_direct5" if direct5 else ("m4_multi_prosody_clean_regularized" if regularized else "m4_multi_prosody_clean"))
    architecture_version="unified_conv1d_5_to_512_v1" if full5 else ("direct_5_channel_first_conv_v1" if direct5 else "pointwise_5_to_1_logf0_init_v1")
    training_strategy="m4_v1_transplant_full5_with_fusion" if full5 else ("m4_v1_transplant_aux_conv_only" if direct5 else ("shared_lr_weight_decay_early_stop" if regularized else "shared_lr_v1"))
    metadata={"model":model_name,"architecture_version":architecture_version,"training_strategy":training_strategy,"config_sha256":sha256_file(config_path),"split_sha256":sha256_file(resolve(SPLIT)),"feature_index_sha256":sha256_file(feature_root/"index.csv")}
    history,best_epoch,best_ua,runtime=train(model,loaders["train"],loaders["validation"],device,config,artifact,metadata,force_training=False,run_name=model_name,optimizer_parameter_groups=optimizer_groups,early_stopping_patience=early_stopping_patience,tracked_parameter_names=tracked_parameters,evaluate_before_training=(direct5 or full5))
    best=torch.load(artifact/"best_validation_ua.pt",map_location="cpu",weights_only=False); model.load_state_dict(best["model_state"]); model.to(device)
    validation,*_=evaluate(model,loaders["validation"],device,config["emotion_order"]); test,truth,pred,prob,test_loss=evaluate(model,loaders["test"],device,config["emotion_order"])
    report_root=resolve(REPORT_ROOT)
    prediction_name="m4_multi_prosody_full5_test_predictions.csv" if full5 else ("m4_multi_prosody_direct5_test_predictions.csv" if direct5 else ("m4_multi_prosody_regularized_test_predictions.csv" if regularized else "m4_multi_prosody_test_predictions.csv"))
    report_name="m4_multi_prosody_full5_report.json" if full5 else ("m4_multi_prosody_direct5_report.json" if direct5 else ("m4_multi_prosody_regularized_report.json" if regularized else "m4_multi_prosody_report.json"))
    model_file="m4_multi_prosody_full5.pt" if full5 else ("m4_multi_prosody_direct5.pt" if direct5 else ("m4_multi_prosody_regularized.pt" if regularized else "m4_multi_prosody.pt"))
    atomic_csv(prediction_frame(parts["test"],truth,pred,prob,config["emotion_order"]),report_root/prediction_name)
    atomic_torch_save({"model_state":model.state_dict(),"config":config,"best_epoch":best_epoch,"metadata":metadata},artifact/model_file)
    atomic_json({"model":model_name,"protocol":config["split_protocol"],"split_counts":{k:len(v) for k,v in parts.items()},"best_epoch":best_epoch,"best_validation_ua":best_ua,"validation":validation,"test":{**test,"loss":test_loss},"history":history,"runtime_seconds":runtime,"lineage":metadata},report_root/report_name)
    print(json.dumps({"status":"PASS","best_epoch":best_epoch,"test_UA":test["ua"],"test_WA":test["wa"],"test_macro_f1":test["macro_f1"]},indent=2))


if __name__ == "__main__": main()
