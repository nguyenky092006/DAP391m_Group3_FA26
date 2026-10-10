"""Run the frozen counterfactual protocol with three-seed family ensembles."""
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from cf_ser.acoustics import acoustic_features, apply_transform
from cf_ser.config import *
from cf_ser.data import load_audio, fix_length
from cf_ser.models import MODEL_CLASSES
from cf_ser.reporting import analyze, figures
from run_pipeline import findings

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_families():
    families={}
    for key,cls in MODEL_CLASSES.items():
        members=[]
        for seed in (2025,2026,2027):
            ck=torch.load(CHECKPOINTS/"ensemble"/f"{key}_seed{seed}.pt",map_location=DEVICE,weights_only=False)
            model=cls().to(DEVICE);model.load_state_dict(ck["state_dict"]);model.eval();members.append((model,ck))
        families[cls.name]=members
    return families

def feature_tensor(y,norm):
    f=acoustic_features(y);a=np.array([f[c] for c in norm["cols"]],np.float32);a=np.where(np.isfinite(a),a,np.array(norm["median"],np.float32));a=(a-np.array(norm["mean"]))/np.array(norm["std"])
    return torch.tensor(a,dtype=torch.float32,device=DEVICE).unsqueeze(0)

@torch.no_grad()
def member_probs(members,y):
    x=torch.tensor(fix_length(y),device=DEVICE).unsqueeze(0);probs=[]
    for model,ck in members:
        pro=feature_tensor(y,ck["prosody_norm"]);probs.append(torch.softmax(model(x,pro),1)[0].cpu().numpy())
    return np.asarray(probs)

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    df=pd.read_csv(SPLIT_CSV,dtype={"speaker":str});params=json.loads((ARTIFACTS/"transformation_params.json").read_text());families=load_families();neutral=df[(df.split=="test")&(df.emotion=="neutral")];rows=[];member_rows=[]
    for si,r in enumerate(neutral.itertuples(index=False)):
        original=load_audio(r.path);waves={tr:apply_transform(original,comp,params) for tr,comp in TRANSFORMS.items()}
        for family,members in families.items():
            p0_members=member_probs(members,original);p0=p0_members.mean(0);op=LABELS[int(p0.argmax())]
            for tr,(y,applied) in waves.items():
                ps=member_probs(members,y);p=ps.mean(0);rows.append({"sample_id":Path(r.path).stem,"speaker_id":r.speaker,"accent":r.accent,"model":family,"transformation":tr,"original_label":"neutral","original_prediction":op,"transformed_prediction":LABELS[int(p.argmax())],"p_happy_original":p0[0],"p_neutral_original":p0[1],"p_sad_original":p0[2],"p_angry_original":p0[3],"p_happy_transformed":p[0],"p_neutral_transformed":p[1],"p_sad_transformed":p[2],"p_angry_transformed":p[3],"delta_p_angry":p[3]-p0[3],**applied})
                for j,seed in enumerate((2025,2026,2027)):
                    a=p0_members[j];b=ps[j];member_rows.append({"sample_id":Path(r.path).stem,"speaker_id":r.speaker,"accent":r.accent,"family":family,"seed":seed,"model":f"{family}_seed{seed}","transformation":tr,"original_prediction":LABELS[int(a.argmax())],"transformed_prediction":LABELS[int(b.argmax())],"p_angry_original":a[3],"p_angry_transformed":b[3],"delta_p_angry":b[3]-a[3]})
        if (si+1)%20==0:logging.info("Ensemble counterfactual %d/%d",si+1,len(neutral))
    # Preserve the prior single-seed file, then make ensemble the canonical result.
    old=ARTIFACTS/"counterfactual_predictions.csv"
    if old.exists() and not (ARTIFACTS/"counterfactual_predictions_single_seed.csv").exists():old.replace(ARTIFACTS/"counterfactual_predictions_single_seed.csv")
    pred=pd.DataFrame(rows);pred.to_csv(old,index=False);members_df=pd.DataFrame(member_rows);members_df.to_csv(ARTIFACTS/"counterfactual_predictions_members.csv",index=False);summary,tests,matrix,flip=analyze(pred)
    base=pred[pred.transformation=="T0_original"].set_index(["model","sample_id"]);valid={(m,s) for (m,s),r in base.iterrows() if r.original_prediction=="neutral"};q=pred[pred.apply(lambda r:(r.model,r.sample_id) in valid,axis=1)]
    accent=q.groupby(["model","accent","transformation"]).agg(mean_delta_p_angry=("delta_p_angry","mean"),angry_flip_rate=("transformed_prediction",lambda x:np.mean(x=="angry")),n=("sample_id","size")).reset_index();accent.to_csv(ARTIFACTS/"rq1_accent_summary.csv",index=False)
    figures(pred,summary,matrix,flip);findings(summary,matrix)
    # Member-level primary analysis, followed by family mean/std across seeds.
    member_summary=[]
    for model,g in members_df.groupby("model"):
        valid_ids=set(g[(g.transformation=="T0_original")&(g.original_prediction=="neutral")].sample_id)
        for tr,h in g[(g.transformation!="T0_original")&g.sample_id.isin(valid_ids)].groupby("transformation"):
            member_summary.append({"family":h.family.iloc[0],"model":model,"seed":int(h.seed.iloc[0]),"transformation":tr,"mean_delta_p_angry":h.delta_p_angry.mean(),"angry_flip_rate":np.mean(h.transformed_prediction=="angry"),"n":len(h)})
    ms=pd.DataFrame(member_summary);ms.to_csv(ARTIFACTS/"rq2_member_cue_summary.csv",index=False)
    stability=ms.groupby(["family","transformation"]).agg(mean_delta_across_seeds=("mean_delta_p_angry","mean"),std_delta_across_seeds=("mean_delta_p_angry","std"),min_delta=("mean_delta_p_angry","min"),max_delta=("mean_delta_p_angry","max"),mean_flip_rate=("angry_flip_rate","mean")).reset_index();stability.to_csv(ARTIFACTS/"rq2_seed_stability.csv",index=False)
    pitch=stability[stability.transformation=="T1_pitch"]
    robust=all(pitch.min_delta>0);note="# Multi-seed robustness\n\n"+f"Pitch delta P(Angry) was positive for every individual seed and architecture: **{robust}**.\n\n"+pitch.to_markdown(index=False)+"\n\nFull member results are in `rq2_member_cue_summary.csv`. Architecture-family conclusions use the three-seed probability ensembles.\n";(ARTIFACTS/"ROBUSTNESS_FINDINGS.md").write_text(note,encoding="utf-8")
    (ARTIFACTS/"ENSEMBLE_NOTE.md").write_text("# Ensemble protocol\n\nCanonical RQ1/RQ2 results average probabilities from three independently initialized members per architecture. Learning rate was selected using validation Macro-F1 only. Counterfactual parameters remained frozen and were never tuned against model predictions.\n",encoding="utf-8")

if __name__=="__main__":main()
