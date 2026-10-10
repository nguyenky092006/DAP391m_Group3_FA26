import random,time
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score
from torch.utils.data import DataLoader
from .config import CHECKPOINTS,LABELS,SEED
from .data import AudioDataset
from .models import MODEL_CLASSES

def set_seed(seed=SEED):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def fit_norm(train_df,features):
 cols=["logf0_mean","logf0_std","log_rms_mean","duration"];z=train_df[["path"]].merge(features,on="path")[cols];med=z.median();z=z.fillna(med);mean=z.mean();std=z.std().replace(0,1)
 return {"cols":cols,"median":med.tolist(),"mean":mean.tolist(),"std":std.tolist()}
def prosody_tensor(paths,fmap,norm,device):
 a=np.asarray([[fmap[str(p)].get(c,np.nan) for c in norm["cols"]] for p in paths],np.float32);med=np.asarray(norm["median"],np.float32);a=np.where(np.isfinite(a),a,med)
 return torch.tensor((a-np.asarray(norm["mean"]))/np.asarray(norm["std"]),dtype=torch.float32,device=device)
@torch.no_grad()
def evaluate(model,loader,device,fmap,norm):
 model.eval();ys=[];ps=[]
 for x,y,paths in loader:
  logits=model(x.to(device),prosody_tensor(paths,fmap,norm,device));ps.extend(logits.argmax(1).cpu().tolist());ys.extend(y.tolist())
 return f1_score(ys,ps,average="macro"),accuracy_score(ys,ps)
def train_models(df,features,epochs=15):
 set_seed();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");fmap={str(r.path):r._asdict() for r in features.itertuples(index=False)}
 train=df[df.split=="train"];val=df[df.split=="val"];test=df[df.split=="test"];norm=fit_norm(train,features);counts=train.emotion.value_counts();weights=torch.tensor([len(train)/(4*counts[x]) for x in LABELS],device=device,dtype=torch.float32)
 loaders={k:DataLoader(AudioDataset(d,k=="train"),batch_size=32,shuffle=k=="train",num_workers=0,pin_memory=True) for k,d in (("train",train),("val",val),("test",test))};inventory=[]
 for key,cls in MODEL_CLASSES.items():
  set_seed();model=cls().to(device);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);criterion=torch.nn.CrossEntropyLoss(weight=weights);best=None;bad=0;started=time.time()
  for epoch in range(epochs):
   model.train();loaders["train"].dataset.epoch=epoch
   for x,y,paths in loaders["train"]:
    x=x.to(device,non_blocking=True);y=y.to(device);pro=prosody_tensor(paths,fmap,norm,device);opt.zero_grad(set_to_none=True);loss=criterion(model(x,pro),y);loss.backward();opt.step()
   vf,va=evaluate(model,loaders["val"],device,fmap,norm);print(f"{key} epoch={epoch+1} val_f1={vf:.4f}",flush=True)
   if best is None or vf>best[0]+1e-4:best=(vf,va,epoch,{k:v.detach().cpu() for k,v in model.state_dict().items()});bad=0
   else:bad+=1
   if bad>=4:break
  model.load_state_dict(best[3]);tf,ta=evaluate(model,loaders["test"],device,fmap,norm);path=CHECKPOINTS/f"{key}.pt"
  torch.save({"model_key":key,"state_dict":model.state_dict(),"labels":LABELS,"prosody_norm":norm,"val_macro_f1":best[0],"test_macro_f1":tf,"test_accuracy":ta,"best_epoch":best[2]+1},path)
  inventory.append({"model":model.name,"key":key,"checkpoint":str(path),"test_macro_f1":tf,"test_accuracy":ta,"pitch_aware":key=="prosody_fusion","seconds":time.time()-started})
 return inventory
def load_models(device):
 out=[]
 for key,cls in MODEL_CLASSES.items():
  c=torch.load(CHECKPOINTS/f"{key}.pt",map_location=device,weights_only=False);m=cls().to(device);m.load_state_dict(c["state_dict"]);m.eval();out.append((key,m,c))
 return out
