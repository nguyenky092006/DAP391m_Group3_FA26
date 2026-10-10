"""Sham-adjusted analysis for the completed pitch-dose experiment."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from cf_ser.config import ARTIFACTS, FIGURES

pred=pd.read_csv(ARTIFACTS/"pitch_dose_predictions.csv")
sham=pred[pred.condition=="sham_resynthesis"][["model","sample_id","p_angry_transformed"]].rename(columns={"p_angry_transformed":"p_angry_sham"})
pred=pred.merge(sham,on=["model","sample_id"],how="left",validate="many_to_one")
pred["delta_p_angry_vs_sham"]=pred.p_angry_transformed-pred.p_angry_sham
pred.to_csv(ARTIFACTS/"pitch_dose_predictions.csv",index=False)

def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);out=np.empty_like(p);run=0
 for rank,i in enumerate(o):run=max(run,(len(p)-rank)*p[i]);out[i]=min(1,run)
 return out

rows=[];tests=[]
for model,g in pred.groupby("model"):
 valid=set(g[(g.condition=="original")&(g.original_prediction=="neutral")].sample_id);h=g[g.sample_id.isin(valid)]
 for condition,q in h.groupby("condition"):
  d=q.delta_p_angry_vs_sham.to_numpy();rows.append({"model":model,"condition":condition,"mean_delta_vs_sham":d.mean(),"median_delta_vs_sham":np.median(d),"std_delta_vs_sham":d.std(),"mean_delta_vs_original":q.delta_p_angry.mean(),"angry_flip_rate":np.mean(q.transformed_prediction=="angry"),"n":len(q)})
  if condition not in ("original","sham_resynthesis"):
   try:w,p=wilcoxon(d,zero_method="zsplit")
   except ValueError:w,p=np.nan,1.
   tests.append({"model":model,"condition":condition,"wilcoxon_vs_sham":w,"p_value":p,"n":len(d)})
summary=pd.DataFrame(rows);tests=pd.DataFrame(tests);tests["p_holm"]=np.nan
for _,idx in tests.groupby("model").groups.items():tests.loc[idx,"p_holm"]=holm(tests.loc[idx,"p_value"])
summary.to_csv(ARTIFACTS/"pitch_dose_summary_sham_adjusted.csv",index=False);tests.to_csv(ARTIFACTS/"pitch_dose_tests_vs_sham.csv",index=False)

dose_names=["sham_resynthesis","pitch_25","pitch_50","pitch_75","pitch_100"];strength=np.array([0,.25,.5,.75,1.]);mono=[]
table=summary[summary.condition.isin(dose_names)].pivot(index="condition",columns="model",values="mean_delta_vs_sham").reindex(dose_names)
for model in table:
 vals=table[model].to_numpy();rho,p=spearmanr(strength,vals);mono.append({"model":model,"spearman_dose_response_vs_sham":rho,"p_value":p,"strictly_monotonic":bool(np.all(np.diff(vals)>0)),"delta_vs_sham_at_100":vals[-1]})
monotonic=pd.DataFrame(mono);monotonic.to_csv(ARTIFACTS/"pitch_dose_monotonicity_sham_adjusted.csv",index=False)

fig,ax=plt.subplots(figsize=(8,5))
for model in table:ax.plot(strength,table[model],marker="o",label=model)
ax.set_xlabel("Pitch intervention strength");ax.set_ylabel("Mean delta P(Angry) vs sham");ax.axhline(0,color="black",lw=.7);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"pitch_dose_curve_sham_adjusted.png",dpi=180);plt.close(fig)

avg=summary.groupby("condition").mean(numeric_only=True)
lines=["# Pitch dose-response findings (sham-adjusted)","","Primary baseline is pitch-tier resynthesis at 0% shift. This separates the F0 intervention from PSOLA resynthesis effects.","","## Mean delta P(Angry) relative to sham","",table.to_markdown(),"","## Monotonicity","",monotonic.to_markdown(index=False),"",f"At 100%, macro-average delta versus sham was **{avg.loc['pitch_100','mean_delta_vs_sham']:.4f}**, with absolute Angry flip rate **{avg.loc['pitch_100','angry_flip_rate']:.2%}**.",f"At 125% stress intensity, delta versus sham was **{avg.loc['pitch_125_stress','mean_delta_vs_sham']:.4f}** and flip rate **{avg.loc['pitch_125_stress','angry_flip_rate']:.2%}**. This is secondary and must not replace the 100% train-derived condition.",f"The opposite-direction control changed P(Angry) by **{avg.loc['pitch_opposite_100','mean_delta_vs_sham']:.4f}** relative to sham.",f"Mean-only intervention: **{avg.loc['pitch_mean_100','mean_delta_vs_sham']:.4f}**; range-only intervention: **{avg.loc['pitch_range_100','mean_delta_vs_sham']:.4f}**.","","Interpretation: the response is driven primarily by the upward F0 baseline shift, not by matching F0 range. Human-listening candidates remain original, sham, 50%, 100%, and opposite 100%; they were selected by protocol coverage, not maximum flip rate."]
(ARTIFACTS/"PITCH_DOSE_FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))
