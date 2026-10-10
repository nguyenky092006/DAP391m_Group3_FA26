import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,wilcoxon
from .config import ARTIFACTS,FIGURES,TRANSFORMS

def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);out=np.empty_like(p);running=0
 for rank,i in enumerate(o):running=max(running,(len(p)-rank)*p[i]);out[i]=min(1,running)
 return out
def ci(x,n=2000):
 x=np.asarray(x,float)
 if len(x)<2:return np.nan,np.nan
 rng=np.random.default_rng(2025);v=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)]);return tuple(np.percentile(v,[2.5,97.5]))
def analyze(pred):
 sums=[];tests=[]
 for model in pred.model.unique():
  base=pred[(pred.model==model)&(pred.transformation=="T0_original")].set_index("sample_id");correct=set(base[base.original_prediction=="neutral"].index)
  for group,ids in (("correct_original_neutral",correct),("all_neutral",set(base.index))):
   for tr in TRANSFORMS:
    q=pred[(pred.model==model)&(pred.transformation==tr)&pred.sample_id.isin(ids)];d=q.delta_p_angry.to_numpy();lo,hi=ci(d)
    sums.append({"model":model,"analysis_group":group,"transformation":tr,"mean_delta_p_angry":np.mean(d) if len(d) else np.nan,"median_delta_p_angry":np.median(d) if len(d) else np.nan,"std_delta_p_angry":np.std(d) if len(d) else np.nan,"ci95_low":lo,"ci95_high":hi,"angry_flip_rate":np.mean(q.transformed_prediction=="angry") if len(q) else np.nan,"neutral_retention_rate":np.mean(q.transformed_prediction=="neutral") if len(q) else np.nan,"n":len(q)})
    if tr!="T0_original" and len(d):
     try:stat,p=wilcoxon(d,zero_method="zsplit")
     except ValueError:stat,p=np.nan,1.
     nz=d[d!=0];effect=(np.sum(nz>0)-np.sum(nz<0))/len(nz) if len(nz) else 0;tests.append({"model":model,"analysis_group":group,"transformation":tr,"wilcoxon_stat":stat,"p_value":p,"direction_effect":effect,"n":len(d)})
 summary=pd.DataFrame(sums);tests=pd.DataFrame(tests);tests["p_holm"]=np.nan
 for _,idx in tests.groupby(["model","analysis_group"]).groups.items():tests.loc[idx,"p_holm"]=holm(tests.loc[idx,"p_value"])
 summary.to_csv(ARTIFACTS/"rq1_summary.csv",index=False);tests.to_csv(ARTIFACTS/"statistical_tests.csv",index=False)
 primary=summary[(summary.analysis_group=="correct_original_neutral")&(summary.transformation!="T0_original")]
 matrix=primary.pivot(index="model",columns="transformation",values="mean_delta_p_angry");flip=primary.pivot(index="model",columns="transformation",values="angry_flip_rate")
 matrix.to_csv(ARTIFACTS/"rq2_model_cue_matrix.csv");flip.to_csv(ARTIFACTS/"rq2_model_flip_matrix.csv")
 rows=[]
 for a,b in itertools.combinations(matrix.index,2):rho,p=spearmanr(matrix.loc[a],matrix.loc[b]);rows.append({"model_a":a,"model_b":b,"spearman_rho":rho,"p_value":p,"n_interventions":matrix.shape[1]})
 pd.DataFrame(rows).to_csv(ARTIFACTS/"rq2_rank_correlations.csv",index=False);return summary,tests,matrix,flip
def figures(pred,summary,matrix,flip):
 p=summary[(summary.analysis_group=="correct_original_neutral")&(summary.transformation!="T0_original")]
 for metric,title,file in (("mean_delta_p_angry","Mean delta P(Angry)","delta_p_angry.png"),("angry_flip_rate","Angry flip rate","angry_flip_rate.png")):
  fig,ax=plt.subplots(figsize=(11,5));p.pivot(index="transformation",columns="model",values=metric).plot.bar(ax=ax);ax.set_title(title);ax.axhline(0,color="black",lw=.7);fig.tight_layout();fig.savefig(FIGURES/file,dpi=180);plt.close(fig)
 for tab,title,file in ((matrix,"Mean delta P(Angry)","model_intervention_heatmap.png"),(flip,"Angry flip rate","model_flip_heatmap.png")):
  fig,ax=plt.subplots(figsize=(10,3.8));im=ax.imshow(tab.values,aspect="auto",cmap="coolwarm");ax.set_xticks(range(tab.shape[1]),tab.columns,rotation=35,ha="right");ax.set_yticks(range(tab.shape[0]),tab.index)
  for i in range(tab.shape[0]):
   for j in range(tab.shape[1]):ax.text(j,i,f"{tab.iloc[i,j]:.3f}",ha="center",va="center",fontsize=8)
  ax.set_title(title);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIGURES/file,dpi=180);plt.close(fig)
 ranks=matrix.rank(axis=1,ascending=False);fig,ax=plt.subplots(figsize=(10,4));ranks.T.plot(ax=ax,marker="o");ax.invert_yaxis();ax.set_ylabel("Rank (1=strongest)");fig.tight_layout();fig.savefig(FIGURES/"per_model_cue_ranking.png",dpi=180);plt.close(fig)
 q=pred[pred.transformation.isin(["T0_original","T7_all"])];fig,ax=plt.subplots(figsize=(9,5))
 for (m,t),g in q.groupby(["model","transformation"]):ax.hist(g.p_angry_transformed,bins=25,alpha=.25,label=f"{m} {t}")
 ax.set_xlabel("P(Angry)");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIGURES/"original_vs_transformed_distribution.png",dpi=180);plt.close(fig)
