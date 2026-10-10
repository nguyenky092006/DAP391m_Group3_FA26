"""Validation-only model tuning and three-seed family ensembles."""
from pathlib import Path
import json, shutil, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from cf_ser.config import ARTIFACTS, CHECKPOINTS, LABELS, SEED, SPLIT_CSV
from cf_ser.data import AudioDataset
from cf_ser.models import MODEL_CLASSES
from cf_ser.train import set_seed, fit_norm, prosody_tensor, evaluate

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT=CHECKPOINTS/"ensemble";OUT.mkdir(parents=True,exist_ok=True)

def train_one(key, seed, lr, df, features, output):
    set_seed(seed);cls=MODEL_CLASSES[key];model=cls().to(DEVICE)
    fmap={str(r.path):r._asdict() for r in features.itertuples(index=False)}
    train=df[df.split=="train"];val=df[df.split=="val"];test=df[df.split=="test"];norm=fit_norm(train,features)
    counts=train.emotion.value_counts();weights=torch.tensor([len(train)/(4*counts[x]) for x in LABELS],device=DEVICE,dtype=torch.float32)
    train_loader=DataLoader(AudioDataset(train,True),batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
    val_loader=DataLoader(AudioDataset(val),batch_size=32,num_workers=0,pin_memory=True);test_loader=DataLoader(AudioDataset(test),batch_size=32,num_workers=0,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    best=None;bad=0;started=time.time()
    for epoch in range(20):
        model.train();train_loader.dataset.epoch=epoch
        for x,y,paths in train_loader:
            x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE);pro=prosody_tensor(paths,fmap,norm,DEVICE)
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(x,pro),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        vf,va=evaluate(model,val_loader,DEVICE,fmap,norm)
        print(f"{key} seed={seed} lr={lr:g} epoch={epoch+1} val_f1={vf:.4f}",flush=True)
        if best is None or vf>best[0]+1e-4:
            best=(vf,va,epoch,{k:v.detach().cpu() for k,v in model.state_dict().items()});bad=0
        else:bad+=1
        if bad>=6:break
    model.load_state_dict(best[3]);tf,ta=evaluate(model,test_loader,DEVICE,fmap,norm)
    ckpt={"model_key":key,"state_dict":model.state_dict(),"labels":LABELS,"prosody_norm":norm,"seed":seed,"lr":lr,
          "val_macro_f1":best[0],"test_macro_f1":tf,"test_accuracy":ta,"best_epoch":best[2]+1,"seconds":time.time()-started}
    torch.save(ckpt,output);return ckpt

def main():
    df=pd.read_csv(SPLIT_CSV,dtype={"speaker":str});features=pd.read_csv(ARTIFACTS/"audio_features.csv");rows=[]
    for key in MODEL_CLASSES:
        canonical=torch.load(CHECKPOINTS/f"{key}.pt",map_location="cpu",weights_only=False)
        candidate_path=OUT/f"{key}_lr3e-4_seed2025.pt"
        candidate=train_one(key,2025,3e-4,df,features,candidate_path)
        if candidate["val_macro_f1"]>canonical["val_macro_f1"]:
            chosen_lr=3e-4;seed_path=candidate_path;chosen=candidate
        else:
            chosen_lr=8e-4;seed_path=CHECKPOINTS/f"{key}.pt";chosen=canonical
        final_seed_path=OUT/f"{key}_seed2025.pt";shutil.copy2(seed_path,final_seed_path)
        rows.append({"key":key,"seed":2025,"lr":chosen_lr,"val_macro_f1":chosen["val_macro_f1"],"test_macro_f1":chosen["test_macro_f1"],"checkpoint":str(final_seed_path)})
        for seed in (2026,2027):
            path=OUT/f"{key}_seed{seed}.pt";ck=train_one(key,seed,chosen_lr,df,features,path)
            rows.append({"key":key,"seed":seed,"lr":chosen_lr,"val_macro_f1":ck["val_macro_f1"],"test_macro_f1":ck["test_macro_f1"],"checkpoint":str(path)})
    result=pd.DataFrame(rows);result.to_csv(ARTIFACTS/"model_optimization.csv",index=False)
    print(result.to_string(index=False));print("\nMeans\n",result.groupby("key")[["val_macro_f1","test_macro_f1"]].agg(["mean","std"]))

if __name__=="__main__":main()
