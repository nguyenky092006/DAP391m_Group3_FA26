"""Pre-registered-style pitch dose response and non-adversarial controls."""
import json, logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import spearmanr, wilcoxon

from cf_ser.acoustics import acoustic_features, pitch_transform
from cf_ser.config import *
from cf_ser.data import load_audio, fix_length
from cf_ser.models import MODEL_CLASSES

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOSE_AUDIO=ARTIFACTS/"audio_examples_dose";DOSE_AUDIO.mkdir(parents=True,exist_ok=True)
CONDITIONS={
 "original":(None,0.),"sham_resynthesis":("full",0.),
 "pitch_25":("full",.25),"pitch_50":("full",.50),"pitch_75":("full",.75),"pitch_100":("full",1.),
 "pitch_125_stress":("full",1.25),"pitch_opposite_100":("full",-1.),
 "pitch_mean_100":("mean",1.),"pitch_range_100":("range",1.),
}

def load_families():
 out={}
 for key,cls in MODEL_CLASSES.items():
  members=[]
  for seed in (2025,2026,2027):
   ck=torch.load(CHECKPOINTS/"ensemble"/f"{key}_seed{seed}.pt",map_location=DEVICE,weights_only=False);m=cls().to(DEVICE);m.load_state_dict(ck["state_dict"]);m.eval();members.append((m,ck))
  out[cls.name]=members
 return out

@torch.no_grad()
def family_prob(members,y):
 x=torch.tensor(fix_length(y),device=DEVICE).unsqueeze(0);f=acoustic_features(y);probs=[]
 for model,ck in members:
  n=ck["prosody_norm"];a=np.array([f[c] for c in n["cols"]],np.float32);a=np.where(np.isfinite(a),a,np.array(n["median"],np.float32));a=(a-np.array(n["mean"]))/np.array(n["std"]);pro=torch.tensor(a,dtype=torch.float32,device=DEVICE).unsqueeze(0);probs.append(torch.softmax(model(x,pro),1)[0].cpu().numpy())
 return np.mean(probs,axis=0)

def holm(p):
 p=np.asarray(p,float);order=np.argsort(p);out=np.empty_like(p);run=0
 for rank,i in enumerate(order):run=max(run,(len(p)-rank)*p[i]);out[i]=min(1,run)
 return out

def main():
 logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s");df=pd.read_csv(SPLIT_CSV,dtype={"speaker":str});params=json.loads((ARTIFACTS/"transformation_params.json").read_text());families=load_families();neutral=df[(df.split=="test")&(df.emotion=="neutral")];rows=[];qc=[]
 for si,r in enumerate(neutral.itertuples(index=False)):
  original=load_audio(r.path);waves={}
  for condition,(mode,strength) in CONDITIONS.items():
   if mode is None:y=original.copy();shift=0.
   else:y,shift=pitch_transform(original,params,strength,mode)
   waves[condition]=(y,shift)
   if si<5:
    path=DOSE_AUDIO/f"{Path(r.path).stem}_{condition}.wav";sf.write(path,y,SAMPLE_RATE,subtype="PCM_16");z,sr=sf.read(path,dtype="float32");feat=acoustic_features(z,sr);qc.append({"file":path.name,"condition":condition,"sample_rate":sr,"finite":np.isfinite(z).all(),"peak":np.max(np.abs(z)),"clipped_fraction":np.mean(np.abs(z)>=.999),**feat})
  for family,members in families.items():
   p0=family_prob(members,original);op=LABELS[int(p0.argmax())]
   for condition,(y,shift) in waves.items():
    p=family_prob(members,y);rows.append({"sample_id":Path(r.path).stem,"speaker_id":r.speaker,"accent":r.accent,"model":family,"condition":condition,"original_prediction":op,"transformed_prediction":LABELS[int(p.argmax())],"p_angry_original":p0[3],"p_angry_transformed":p[3],"delta_p_angry":p[3]-p0[3],"mean_logf0_shift":shift})
  if (si+1)%20==0:logging.info("Pitch dose %d/%d",si+1,len(neutral))
 pred=pd.DataFrame(rows);pred.to_csv(ARTIFACTS/"pitch_dose_predictions.csv",index=False);qc=pd.DataFrame(qc);qc.to_csv(ARTIFACTS/"pitch_dose_qc.csv",index=False)
 assert len(qc)==5*len(CONDITIONS) and qc.finite.all() and (qc.sample_rate==SAMPLE_RATE).all() and (qc.clipped_fraction<.01).all()
 summaries=[];tests=[]
 for model,g in pred.groupby("model"):
  valid=set(g[(g.condition=="original")&(g.original_prediction=="neutral")].sample_id)
  h=g[g.sample_id.isin(valid)]
  for condition,q in h.groupby("condition"):
   d=q.delta_p_angry.to_numpy();summaries.append({"model":model,"condition":condition,"mean_delta_p_angry":d.mean(),"median_delta_p_angry":np.median(d),"std":d.std(),"angry_flip_rate":np.mean(q.transformed_prediction=="angry"),"neutral_retention_rate":np.mean(q.transformed_prediction=="neutral"),"n":len(q)})
   if condition!="original":
    try:w,p=wilcoxon(d,zero_method="zsplit")
    except ValueError:w,p=np.nan,1.
    tests.append({"model":model,"condition":condition,"wilcoxon":w,"p_value":p,"n":len(d)})
 summary=pd.DataFrame(summaries);tests=pd.DataFrame(tests)
 for _,idx in tests.groupby("model").groups.items():tests.loc[idx,"p_holm"]=holm(tests.loc[idx,"p_value"])
 summary.to_csv(ARTIFACTS/"pitch_dose_summary.csv",index=False);tests.to_csv(ARTIFACTS/"pitch_dose_tests.csv",index=False)
 mono=[];dose_names=["original","pitch_25","pitch_50","pitch_75","pitch_100"];dose=np.array([0,.25,.5,.75,1.])
 for model,g in summary[summary.condition.isin(dose_names)].groupby("model"):
  vals=np.array([g[g.condition==x].mean_delta_p_angry.iloc[0] for x in dose_names]);rho,p=spearmanr(dose,vals);mono.append({"model":model,"spearman_dose_response":rho,"p_value":p,"delta_at_100":vals[-1],"strictly_monotonic":bool(np.all(np.diff(vals)>0))})
 monotonic=pd.DataFrame(mono);monotonic.to_csv(ARTIFACTS/"pitch_dose_monotonicity.csv",index=False)
 order=list(CONDITIONS);pivot=summary.pivot(index="condition",columns="model",values="mean_delta_p_angry").reindex(order);fig,ax=plt.subplots(figsize=(12,5));pivot.plot.bar(ax=ax);ax.axhline(0,color="black",lw=.7);ax.set_ylabel("Mean delta P(Angry)");fig.tight_layout();fig.savefig(FIGURES/"pitch_dose_response.png",dpi=180);plt.close(fig)
 dose_table=summary[summary.condition.isin(dose_names)].pivot(index="condition",columns="model",values="mean_delta_p_angry").reindex(dose_names);fig,ax=plt.subplots(figsize=(8,5))
 for model in dose_table:ax.plot(dose,dose_table[model],marker="o",label=model)
 ax.set_xlabel("Pitch intervention strength");ax.set_ylabel("Mean delta P(Angry)");ax.legend();ax.axhline(0,color="black",lw=.7);fig.tight_layout();fig.savefig(FIGURES/"pitch_dose_curve.png",dpi=180);plt.close(fig)
 primary=summary.groupby("condition").mean(numeric_only=True);lines=["# Pitch dose-response findings","","Transformation strengths were fixed in advance from train-derived Neutral-to-Angry statistics; model predictions were not used to tune them.","","## Main 0–100% dose response","",dose_table.to_markdown(),"","## Monotonicity","",monotonic.to_markdown(index=False),"",f"At 100%, macro-average flip rate was **{primary.loc['pitch_100','angry_flip_rate']:.2%}**. At 125% stress-test intensity it was **{primary.loc['pitch_125_stress','angry_flip_rate']:.2%}**; the latter is not a primary result.",f"Sham mean delta was **{primary.loc['sham_resynthesis','mean_delta_p_angry']:.4f}** and opposite-direction mean delta was **{primary.loc['pitch_opposite_100','mean_delta_p_angry']:.4f}**.",f"Mean-only pitch delta was **{primary.loc['pitch_mean_100','mean_delta_p_angry']:.4f}**; range-only delta was **{primary.loc['pitch_range_100','mean_delta_p_angry']:.4f}**.","","Human-listening candidates selected by protocol coverage rather than flip optimization: original, sham, pitch 50%, pitch 100%, and opposite 100%."]
 (ARTIFACTS/"PITCH_DOSE_FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
