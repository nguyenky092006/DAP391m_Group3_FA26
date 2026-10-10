from pathlib import Path
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from .config import LABEL_TO_ID, NUM_SAMPLES, SAMPLE_RATE, SEED, WAV_DIR

def load_metadata(meta_csv):
    df = pd.read_csv(meta_csv)
    required = {"path", "emotion", "speaker", "accent"}
    if required - set(df): raise ValueError(f"Missing metadata columns: {required-set(df)}")
    df = df.copy(); df["path"] = df.path.map(lambda p: str(WAV_DIR / Path(str(p)).name))
    df["speaker"] = df.speaker.astype(str); df["emotion"] = df.emotion.str.lower(); df["accent"] = df.accent.str.lower()
    if not set(df.emotion).issubset(LABEL_TO_ID): raise ValueError("Unexpected emotion label")
    missing = [p for p in df.path if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} audio files")
    return df

def make_speaker_split(df, split_csv):
    if "split" in df and set(df.split.dropna().unique()) >= {"train","val","test"}:
        out=df.copy(); source="existing metadata split"
    elif Path(split_csv).exists():
        old=pd.read_csv(split_csv,dtype={"speaker":str})
        if len(old)==len(df) and set(old.split.unique())=={"train","val","test"} and "weighted-v2" in str(old.split_source.iloc[0]): return old
        # A prior generated split from an older algorithm is not treated as an
        # externally supplied project split and is regenerated deterministically.
        return _weighted_split(df, split_csv)
    else:
        return _weighted_split(df, split_csv)
    for name in ("train","val","test"):
        if set(out[out.split==name].emotion)!=set(LABEL_TO_ID): raise ValueError(f"Split {name} lacks a class")
    if set(out[out.split=="train"].speaker)&set(out[out.split=="test"].speaker): raise ValueError("Speaker leakage")
    out["split_source"]=source; Path(split_csv).parent.mkdir(parents=True,exist_ok=True); out.to_csv(split_csv,index=False); return out

def _weighted_split(df, split_csv):
    """Weighted speaker split; ID 0 is a missing-speaker bucket and is stratified by row."""
    rng=np.random.default_rng(SEED); names=["train","val","test"];ratios=np.array([.70,.15,.15])
    known=df[df.speaker!="0"].copy(); unknown=df[df.speaker=="0"].copy(); mapping={}
    # Balance sample counts, emotion counts, and accents while keeping every
    # identified speaker in exactly one split.
    dims=["total"]+[f"e:{x}" for x in sorted(df.emotion.unique())]+[f"a:{x}" for x in sorted(df.accent.unique())]
    vectors={}
    for speaker,g in known.groupby("speaker"):
        v=[len(g)]+[(g.emotion==x).sum() for x in sorted(df.emotion.unique())]+[(g.accent==x).sum() for x in sorted(df.accent.unique())]
        vectors[speaker]=np.asarray(v,float)
    total=sum(vectors.values(),start=np.zeros(len(dims)));target=ratios[:,None]*total[None,:];current=np.zeros_like(target)
    speakers=list(vectors);rng.shuffle(speakers);speakers.sort(key=lambda s:vectors[s][0],reverse=True)
    for speaker in speakers:
        v=vectors[speaker];scores=[]
        for j in range(3):
            cand=current.copy();cand[j]+=v;scores.append(np.sum(((cand-target)/np.maximum(target,1))**2))
        j=int(np.argmin(scores));mapping[speaker]=names[j];current[j]+=v
    out=known.copy();out["split"]=out.speaker.map(mapping)
    # ID 0 covers 2,217 heterogeneous files and cannot represent one real
    # speaker. Stratifying this placeholder prevents it dominating one split.
    pieces=[]
    for _,g in unknown.groupby(["emotion","accent"]):
        idx=g.index.to_numpy().copy();rng.shuffle(idx);a=round(.70*len(idx));b=round(.15*len(idx));
        gg=g.copy();gg["split"]="test";gg.loc[idx[:a],"split"]="train";gg.loc[idx[a:a+b],"split"]="val";pieces.append(gg)
    out=pd.concat([out,*pieces]).sort_index();source="generated weighted-v2 split, seed 2025; identified speakers disjoint; speaker=0 missing-ID bucket stratified by row"
    for name in names:
        if set(out[out.split==name].emotion)!=set(LABEL_TO_ID):raise ValueError(f"Split {name} lacks a class")
    out["split_source"]=source;Path(split_csv).parent.mkdir(parents=True,exist_ok=True);out.to_csv(split_csv,index=False);return out

def load_audio(path, sr=SAMPLE_RATE):
    y,native=sf.read(path,dtype="float32",always_2d=False)
    if y.ndim==2:y=y.mean(1)
    if native!=sr:y=librosa.resample(y,orig_sr=native,target_sr=sr)
    if len(y)==0 or not np.isfinite(y).all():raise ValueError(f"Invalid waveform {path}")
    return np.asarray(y,np.float32)

def fix_length(y, training=False, rng=None):
    rng=rng or np.random.default_rng(SEED)
    if len(y)>NUM_SAMPLES:
        start=int(rng.integers(0,len(y)-NUM_SAMPLES+1)) if training else (len(y)-NUM_SAMPLES)//2; y=y[start:start+NUM_SAMPLES]
    elif len(y)<NUM_SAMPLES:
        n=NUM_SAMPLES-len(y); y=np.pad(y,(n//2,n-n//2))
    return y.astype(np.float32)

class AudioDataset(Dataset):
    def __init__(self,frame,training=False):self.frame=frame.reset_index(drop=True);self.training=training;self.epoch=0
    def __len__(self):return len(self.frame)
    def __getitem__(self,idx):
        r=self.frame.iloc[idx]; rng=np.random.default_rng(SEED+self.epoch*100003+idx)
        return torch.from_numpy(fix_length(load_audio(r.path),self.training,rng)),LABEL_TO_ID[r.emotion],r.path
