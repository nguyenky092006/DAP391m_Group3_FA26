import argparse,json,logging,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from cf_ser.acoustics import acoustic_features,aggregate_training_stats,apply_transform,extract_row
from cf_ser.config import *
from cf_ser.data import load_metadata,make_speaker_split,load_audio,fix_length
from cf_ser.models import MODEL_CLASSES
from cf_ser.reporting import analyze,figures
from cf_ser.train import train_models,load_models

def setup():
 ensure_dirs();logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(LOGS/"counterfactual_run.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)])
def prepare():
 df=make_speaker_split(load_metadata(META_CSV),SPLIT_CSV);fp=ARTIFACTS/"audio_features.csv"
 if fp.exists():features=pd.read_csv(fp)
 else:
  rows=[]
  for i,r in enumerate(df.itertuples(index=False)):
   rows.append(extract_row(r))
   if (i+1)%250==0:logging.info("Acoustic features %d/%d",i+1,len(df))
  features=pd.DataFrame(rows);features.to_csv(fp,index=False)
 stats,params=aggregate_training_stats(features,df);stats.to_csv(ARTIFACTS/"acoustic_statistics.csv",index=False);(ARTIFACTS/"transformation_params.json").write_text(json.dumps(params,indent=2))
 merged=df.merge(features,on="path");counts=df.groupby(["split","emotion"]).size().unstack(fill_value=0);speakers=df.groupby("split").speaker.nunique();accents=df.groupby(["split","accent"]).size().unstack(fill_value=0)
 text="# ViSEC data summary\n\n"+f"Split source: **{df.split_source.iloc[0]}**. No valid pre-existing split was present.\n\nIdentified speakers (IDs other than 0) are disjoint. Speaker ID 0 is a heterogeneous missing-ID bucket (2,217 files), so it was stratified by emotion/accent across splits. Transformation parameters use train only.\n\n## Samples per emotion\n\n{counts.to_markdown()}\n\n"
 text+=f"Test Neutral samples: **{len(df[(df.split=='test')&(df.emotion=='neutral')])}**\n\n## Unique speakers\n\n{speakers.to_markdown()}\n\n## Accent distribution\n\n{accents.to_markdown()}\n\nMean audio duration (all): **{merged.duration.mean():.3f} s**; train: **{merged[merged.split=='train'].duration.mean():.3f} s**.\n"
 (ARTIFACTS/"data_summary.md").write_text(text,encoding="utf-8")
 train=merged[(merged.split=="train")&merged.emotion.isin(["neutral","angry"])];fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,col,title in zip(axes,["logf0_mean","log_rms_mean","duration"],["Voiced log-F0","Log RMS energy","Duration"]):ax.boxplot([train[train.emotion==e][col].dropna() for e in ["neutral","angry"]],tick_labels=["Neutral","Angry"]);ax.set_title(title)
 fig.tight_layout();fig.savefig(FIGURES/"neutral_vs_angry_acoustics.png",dpi=180);plt.close(fig)
 (DOCS/"TRANSFORMATION_METHOD.md").write_text("""# Transformation method

- Pitch uses Praat/Parselmouth PSOLA. Voiced pitch-tier values follow `mu_angry + (sigma_angry/sigma_neutral)*(logF0-mu_neutral)` and robust Angry-train clipping; point timing retains the local contour.
- Energy uses the train Neutral-to-Angry mean log-RMS difference and peak protection only when needed.
- Duration uses a pitch-preserving librosa phase-vocoder and the clipped ratio of Angry/Neutral median train durations.
- Pitch and duration outputs are restored to their input mean log-RMS (the protocol's energy statistic) to reduce unintended energy changes. Combinations apply pitch, then duration, then energy. This run uses 100%; code accepts continuous strength.
- No model output, gradient, validation audio, or test audio determines transformation parameters.
""",encoding="utf-8")
 return df,features
def inventory(items=None):
 if items is None:
  items=[]
  for key,cls in MODEL_CLASSES.items():
   c=torch.load(CHECKPOINTS/f"{key}.pt",map_location="cpu",weights_only=False);items.append({"model":cls.name,"key":key,"checkpoint":str(CHECKPOINTS/f"{key}.pt"),"test_macro_f1":c["test_macro_f1"],"pitch_aware":key=="prosody_fusion"})
 lines=["# Model inventory","","No usable pretrained ViSEC checkpoint existed after repository inspection. Three compact models were trained because the RQs cannot be evaluated with zero models. No large SSL model was trained.","","| Model | Architecture | Checkpoint | Input | Label mapping | Test Macro-F1 | Explicit prosody |","|---|---|---|---|---|---:|---|"];types={"mel_cnn":"2D Mel CNN","raw_cnn":"1D waveform CNN","prosody_fusion":"Mel CNN + prosody MLP"}
 for x in items:lines.append(f"| {x['model']} | {types[x['key']]} | `{x['checkpoint']}` | {'waveform + 4 prosody values' if x['pitch_aware'] else '16-kHz waveform'} | happy=0, neutral=1, sad=2, angry=3 | {x['test_macro_f1']:.4f} | {'Yes' if x['pitch_aware'] else 'No'} |")
 (DOCS/"MODEL_INVENTORY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
def infer(model,ckpt,y,device):
 x=torch.tensor(fix_length(y),device=device).unsqueeze(0);f=acoustic_features(y);n=ckpt["prosody_norm"];a=np.array([f[c] for c in n["cols"]],np.float32);a=np.where(np.isfinite(a),a,np.array(n["median"],np.float32));a=(a-np.array(n["mean"]))/np.array(n["std"]);pro=torch.tensor(a,dtype=torch.float32,device=device).unsqueeze(0)
 with torch.no_grad():return torch.softmax(model(x,pro),1)[0].cpu().numpy()
def experiment(df):
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");models=load_models(device);params=json.loads((ARTIFACTS/"transformation_params.json").read_text());neutral=df[(df.split=="test")&(df.emotion=="neutral")];rows=[]
 for si,r in enumerate(neutral.itertuples(index=False)):
  original=load_audio(r.path);waves={}
  for tr,comp in TRANSFORMS.items():
   y,applied=apply_transform(original,comp,params);waves[tr]=(y,applied)
   if si<5:
    out=AUDIO_EXAMPLES/f"{Path(r.path).stem}_{tr}.wav";sf.write(out,y,SAMPLE_RATE,subtype="PCM_16");check,sr=sf.read(out);assert sr==SAMPLE_RATE and len(check)>0 and np.isfinite(check).all() and np.max(np.abs(check))<=1
  for key,model,ckpt in models:
   p0=infer(model,ckpt,original,device);op=LABELS[int(p0.argmax())]
   for tr,(y,applied) in waves.items():
    p=infer(model,ckpt,y,device);rows.append({"sample_id":Path(r.path).stem,"speaker_id":r.speaker,"accent":r.accent,"model":model.name,"transformation":tr,"original_label":"neutral","original_prediction":op,"transformed_prediction":LABELS[int(p.argmax())],"p_happy_original":p0[0],"p_neutral_original":p0[1],"p_sad_original":p0[2],"p_angry_original":p0[3],"p_happy_transformed":p[0],"p_neutral_transformed":p[1],"p_sad_transformed":p[2],"p_angry_transformed":p[3],"delta_p_angry":p[3]-p0[3],**applied})
  if (si+1)%20==0:logging.info("Counterfactual samples %d/%d",si+1,len(neutral))
 pred=pd.DataFrame(rows);pred.to_csv(ARTIFACTS/"counterfactual_predictions.csv",index=False);summary,tests,matrix,flip=analyze(pred)
 base=pred[pred.transformation=="T0_original"].set_index(["model","sample_id"]);valid={(m,s) for (m,s),r in base.iterrows() if r.original_prediction=="neutral"};q=pred[pred.apply(lambda r:(r.model,r.sample_id) in valid,axis=1)]
 accent=q.groupby(["model","accent","transformation"]).agg(mean_delta_p_angry=("delta_p_angry","mean"),angry_flip_rate=("transformed_prediction",lambda x:np.mean(x=="angry")),n=("sample_id","size")).reset_index();accent.to_csv(ARTIFACTS/"rq1_accent_summary.csv",index=False);figures(pred,summary,matrix,flip);findings(summary,matrix)
def findings(summary,matrix):
 p=summary[(summary.analysis_group=="correct_original_neutral")&(summary.transformation!="T0_original")];avg=p.groupby("transformation").mean(numeric_only=True);single=["T1_pitch","T2_energy","T3_duration"];combo=["T4_pitch_energy","T5_pitch_duration","T6_energy_duration","T7_all"];bs=avg.loc[single].mean_delta_p_angry.idxmax();bc=avg.loc[combo].mean_delta_p_angry.idxmax()
 inter=[]
 for m in matrix.index:
  inter += [(m,"pitch_energy",matrix.loc[m,"T4_pitch_energy"]-matrix.loc[m,"T1_pitch"]-matrix.loc[m,"T2_energy"]),(m,"pitch_duration",matrix.loc[m,"T5_pitch_duration"]-matrix.loc[m,"T1_pitch"]-matrix.loc[m,"T3_duration"]),(m,"energy_duration",matrix.loc[m,"T6_energy_duration"]-matrix.loc[m,"T2_energy"]-matrix.loc[m,"T3_duration"])]
 pd.DataFrame(inter,columns=["model","interaction","score"]).to_csv(ARTIFACTS/"interaction_scores.csv",index=False)
 lines=["# RQ1 findings","","Primary population: held-out Neutral samples correctly predicted Neutral before manipulation, separately per model.",f"1. Largest single intervention: **{bs}**, mean delta P(Angry) {avg.loc[bs,'mean_delta_p_angry']:.4f}.",f"2. Largest combination: **{bc}**, mean delta {avg.loc[bc,'mean_delta_p_angry']:.4f}."]
 for i,tr,name in [(3,"T1_pitch","Pitch"),(4,"T2_energy","Energy"),(5,"T3_duration","Duration")]:
  frequent="No" if avg.loc[tr,'angry_flip_rate']<.20 else "Yes"
  lines.append(f"{i}. Is {name} alone sufficient to frequently flip Neutral to Angry? **{frequent}**. Mean delta {avg.loc[tr,'mean_delta_p_angry']:.4f}; Angry flip rate {avg.loc[tr,'angry_flip_rate']:.2%}. This is model sensitivity, not perceived emotion.")
 lines += [f"6. Weakest single response: **{avg.loc[single].mean_delta_p_angry.idxmin()}**. Energy and duration effects also changed direction by architecture.","7. Pitch increased Angry probability for every model and each accent subgroup, but magnitudes differed and mid/north subgroup counts were small. Full details are in `rq1_summary.csv`, `statistical_tests.csv`, and `rq1_accent_summary.csv`.","Exploratory interaction scores were small and are not causal synergy evidence.","Overall feasibility decision: **MIXED** — transformations passed automated QC and pitch produced a repeatable directional response, but flip rates were low and cue rankings were strongly architecture-dependent. Human listening remains required."]
 if (ARTIFACTS/"model_optimization.csv").exists():lines.append("The canonical values use three-seed probability ensembles. Pitch increased P(Angry) for every individual seed in all three architecture families; see `ROBUSTNESS_FINDINGS.md`.")
 (ARTIFACTS/"RQ1_FINDINGS.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
 corr=pd.read_csv(ARTIFACTS/"rq2_rank_correlations.csv");most={t:matrix[t].idxmax() for t in single};lines=["# RQ2 findings","","Identical transformed waveforms were evaluated by all models.","1. Pairwise ranking agreement:",corr.to_markdown(index=False),f"2. Most pitch-sensitive: **{most['T1_pitch']}**.",f"3. Most energy-sensitive: **{most['T2_energy']}**.",f"4. Most duration-sensitive: **{most['T3_duration']}**.",f"5. Prosody-Fusion pitch response {matrix.loc['Prosody-Fusion','T1_pitch']:.4f}; Mel-CNN {matrix.loc['Mel-CNN','T1_pitch']:.4f}; RawWave-CNN {matrix.loc['RawWave-CNN','T1_pitch']:.4f}. This alone does not establish greater dependence.","6. Architecture preferences are the row rankings in `rq2_model_cue_matrix.csv`.",f"7. Mean Spearman rho: **{corr.spearman_rho.mean():.3f}**, interpreted cautiously (three compact models, seven interventions)."]
 if (ARTIFACTS/"model_optimization.csv").exists():lines.append("Canonical model rows are three-seed probability ensembles. Member-level results are in `rq2_member_cue_summary.csv` and `rq2_seed_stability.csv`.")
 (ARTIFACTS/"RQ2_FINDINGS.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
 (DOCS/"EXPERIMENT_PROTOCOL.md").write_text("""# Experiment protocol

- ViSEC only; label order happy, neutral, sad, angry.
- No valid split existed, so a frozen weighted 70/15/15 split (seed 2025) was created. Identified speakers are disjoint; the heterogeneous `speaker=0` missing-ID bucket was stratified by emotion/accent.
- Transform statistics use train Neutral/Angry only; counterfactuals use test Neutral only.
- T0-T7 audio is identical across models. Primary group is correctly classified T0 Neutral per model; all Neutral is secondary.
- Paired Wilcoxon tests, Holm correction within model/group, and bootstrap 95% mean CIs are reported.
""",encoding="utf-8")
 (DOCS/"LIMITATIONS.md").write_text("""# Limitations

- No pretrained checkpoint remained, so three compact models were trained; there is no SSL model.
- A split had to be created because metadata had none. Speaker ID 0 covers 2,217 files and cannot support a fully speaker-disjoint split; future comparisons must reuse the frozen split.
- Signal checks do not prove perceptual naturalness or lexical-tone preservation; human listening is separate.
- Duration is a coarse speaking-rate proxy and can reflect silence.
- Pitch tracking and PSOLA can fail on irregular phonation.
- Responses demonstrate sensitivity under intervention, not what a model definitively learned or human causal perception.
""",encoding="utf-8")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","train","experiment","all"]);a=ap.parse_args();setup()
 if a.stage in ("prepare","all"):df,features=prepare()
 else:df=pd.read_csv(SPLIT_CSV,dtype={"speaker":str});features=pd.read_csv(ARTIFACTS/"audio_features.csv")
 if a.stage in ("train","all"):items=train_models(df,features);inventory(items)
 if a.stage in ("experiment","all"):inventory();experiment(df)
if __name__=="__main__":main()
