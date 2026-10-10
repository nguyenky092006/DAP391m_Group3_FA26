import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from .config import SAMPLE_RATE
from .data import load_audio

def mean_log_rms(y):
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0]
    return float(np.mean(np.log(np.maximum(rms,1e-7))))

def match_log_rms(out, reference):
    """Match the exact energy statistic used by the intervention protocol."""
    gain=float(np.exp(mean_log_rms(reference)-mean_log_rms(out)));out=out*gain
    peak=float(np.max(np.abs(out)))
    if peak>.99:out*=.99/peak
    return out.astype(np.float32)

def acoustic_features(y,sr=SAMPLE_RATE):
    sound=parselmouth.Sound(y,sampling_frequency=sr); pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=60,pitch_ceiling=500)
    f0=pitch.selected_array["frequency"]; f0=f0[(f0>0)&np.isfinite(f0)]; lf=np.log(f0) if len(f0) else np.array([])
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0]; lr=np.log(np.maximum(rms,1e-7))
    return {"logf0_mean":float(np.mean(lf)) if len(lf) else np.nan,"logf0_std":float(np.std(lf)) if len(lf) else np.nan,
      "logf0_p05":float(np.percentile(lf,5)) if len(lf) else np.nan,"logf0_p95":float(np.percentile(lf,95)) if len(lf) else np.nan,
      "logf0_range":float(np.percentile(lf,95)-np.percentile(lf,5)) if len(lf) else np.nan,
      "log_rms_mean":float(np.mean(lr)),"log_rms_std":float(np.std(lr)),"duration":len(y)/sr,
      "voiced_ratio":len(f0)/max(1,len(pitch.xs())),"peak":float(np.max(np.abs(y)))}

def extract_row(row):return {"path":row.path,**acoustic_features(load_audio(row.path))}

def aggregate_training_stats(features,split_df):
    train=split_df[["path","emotion","split"]].merge(features,on="path");train=train[train.split=="train"];rows=[]
    for emotion in ("neutral","angry"):
      sub=train[train.emotion==emotion]
      for feature in ["logf0_mean","logf0_std","logf0_range","log_rms_mean","log_rms_std","duration","voiced_ratio"]:
        v=sub[feature].dropna().to_numpy();rows.append({"emotion":emotion,"feature":feature,"mean":v.mean(),"std":v.std(),"median":np.median(v),"p05":np.percentile(v,5),"p95":np.percentile(v,95),"n":len(v)})
    table=pd.DataFrame(rows)
    def g(e,f,s):return float(table[(table.emotion==e)&(table.feature==f)][s].iloc[0])
    params={"mu_logf0_neutral":g("neutral","logf0_mean","mean"),"mu_logf0_angry":g("angry","logf0_mean","mean"),
      "sigma_logf0_neutral":max(g("neutral","logf0_std","mean"),1e-3),"sigma_logf0_angry":max(g("angry","logf0_std","mean"),1e-3),
      "f0_clip_low":g("angry","logf0_p05" if False else "logf0_mean","p05")-g("angry","logf0_std","p95"),
      "f0_clip_high":g("angry","logf0_mean","p95")+g("angry","logf0_std","p95"),
      "delta_log_rms":g("angry","log_rms_mean","mean")-g("neutral","log_rms_mean","mean"),
      "duration_factor":float(np.clip(g("angry","duration","median")/g("neutral","duration","median"),.75,1.25))}
    return table,params

def pitch_transform(y,p,strength=1.,mode="full"):
    snd=parselmouth.Sound(y,sampling_frequency=SAMPLE_RATE);man=call(snd,"To Manipulation",.01,60,500);tier=call(man,"Extract pitch tier");n=int(call(tier,"Get number of points"));shifts=[]
    new_tier=call("Create PitchTier","counterfactual",float(snd.xmin),float(snd.xmax))
    ratio=p["sigma_logf0_angry"]/p["sigma_logf0_neutral"]
    for i in range(1,n+1):
      f=float(call(tier,"Get value at index",i))
      if f<=0:continue
      old=np.log(f)
      if mode=="full":target=p["mu_logf0_angry"]+ratio*(old-p["mu_logf0_neutral"])
      elif mode=="mean":target=old+(p["mu_logf0_angry"]-p["mu_logf0_neutral"])
      elif mode=="range":target=p["mu_logf0_neutral"]+ratio*(old-p["mu_logf0_neutral"])
      else:raise ValueError(f"Unknown pitch mode: {mode}")
      new=old+strength*(target-old)
      if strength>0:new=np.clip(new,p["f0_clip_low"],p["f0_clip_high"])
      else:new=np.clip(new,np.log(60.),np.log(500.))
      t=float(call(tier,"Get time from index",i));call(new_tier,"Add point",t,float(np.exp(new)));shifts.append(new-old)
    call([man,new_tier],"Replace pitch tier");out=call(man,"Get resynthesis (overlap-add)").values[0].astype(np.float32)
    # Restore the protocol's log-RMS statistic so pitch-only does not become
    # an unintended energy intervention.
    out=match_log_rms(out,y)
    return out,float(np.mean(shifts) if shifts else 0)

def energy_transform(y,p,strength=1.):
    req=float(np.exp(strength*p["delta_log_rms"]));out=y*req;lim=min(1.,.99/max(float(np.max(np.abs(out))),1e-8));return (out*lim).astype(np.float32),req*lim

def duration_transform(y,p,strength=1.):
    factor=float(np.clip(1+strength*(p["duration_factor"]-1),.75,1.25));out=librosa.effects.time_stretch(y,rate=1/factor).astype(np.float32)
    # Restore pre-stretch mean log-RMS (the energy statistic used here).
    out=match_log_rms(out,y)
    return out.astype(np.float32),factor

def apply_transform(y,components,p,strength=1.):
    out=y.copy();shift=0.;gain=1.;factor=1.
    if "pitch" in components:out,shift=pitch_transform(out,p,strength)
    if "duration" in components:out,factor=duration_transform(out,p,strength)
    # Apply the learned energy shift last, after RMS-preserving pitch/duration.
    if "energy" in components:out,gain=energy_transform(out,p,strength)
    if len(out)==0 or not np.isfinite(out).all():raise ValueError("Invalid transformed waveform")
    peak=float(np.max(np.abs(out)))
    if peak>1:out*=.99/peak
    return out.astype(np.float32),{"f0_shift_applied":shift,"energy_gain_applied":gain,"duration_factor":factor}
