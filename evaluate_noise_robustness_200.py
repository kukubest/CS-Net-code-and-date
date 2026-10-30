"""Inference-only multi-level noise robustness test on 200 independent volumes."""
import csv, gc, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ablation_metrics import volume_metrics, summarize_metric_rows
from unified_evaluate_200 import pairs, load_raw, make_model, CHECKPOINTS, PAD, COPPER

ROOT=Path(r"G:\CT_RE\CT_DRA")
OUT=ROOT/'outputs'/'noise_robustness_200'
SEED=20260923
CONDITIONS=[
    {'id':'baseline','family':'none','label':'Baseline','sigma':0.0},
    {'id':'g001','family':'gaussian','label':'Gaussian $\\sigma=0.01$','sigma':0.01},
    {'id':'g003','family':'gaussian','label':'Gaussian $\\sigma=0.03$','sigma':0.03},
    {'id':'g005','family':'gaussian','label':'Gaussian $\\sigma=0.05$','sigma':0.05},
    {'id':'g008','family':'gaussian','label':'Gaussian $\\sigma=0.08$','sigma':0.08},
    {'id':'pg_low','family':'poisson_gaussian','label':'P-G low','peak':2000,'sigma':0.005},
    {'id':'pg_mid','family':'poisson_gaussian','label':'P-G medium','peak':500,'sigma':0.01},
    {'id':'pg_high','family':'poisson_gaussian','label':'P-G high','peak':100,'sigma':0.02},
]
METHODS=['SIRT input','3D U-Net','UNETR','Swin-UNETR','CS-Net']
METRICS=['rmse','psnr','ssim','gradient_mae','copper_dice','copper_iou']

def summarize_block(block):
    return summarize_metric_rows([{'filename':r['filename'],**{k:r[k] for k in METRICS}} for r in block])

def perturb(x, condition, index):
    if condition['family']=='none': return x.copy()
    rng=np.random.default_rng(SEED + index*1009 + sum(map(ord,condition['id'])))
    if condition['family']=='gaussian':
        y=x+rng.normal(0,condition['sigma'],x.shape).astype(np.float32)
    else:
        peak=float(condition['peak'])
        y=rng.poisson(np.clip(x,0,1)*peak).astype(np.float32)/peak
        y+=rng.normal(0,condition['sigma'],x.shape).astype(np.float32)
    return np.clip(y,0,1).astype(np.float32)

def save_rows(rows):
    fields=['method','condition','family','filename','rmse','psnr','ssim','gradient_mae','copper_dice','copper_iou']
    with (OUT/'noise_robustness_per_sample.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ps=pairs(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; summaries={}; started=time.perf_counter()

    # No-denoising reference: exactly the same perturbed inputs used by networks.
    for ci,cond in enumerate(CONDITIONS):
        block=[]
        for i,(d,c) in enumerate(ps):
            pred=perturb(load_raw(d),cond,i); gt=load_raw(c)
            rec={'method':'SIRT input','condition':cond['id'],'family':cond['family'],'filename':c.name,**volume_metrics(pred,gt,COPPER)}
            rows.append(rec); block.append(rec)
        summaries.setdefault('SIRT input',{})[cond['id']]=summarize_block(block)
        print(f"[SIRT input] {ci+1}/{len(CONDITIONS)} {cond['id']}",flush=True)

    for method in METHODS[1:]:
        model=make_model(method).to(device)
        ckpt=torch.load(CHECKPOINTS[method],map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'] if method=='CS-Net' else ckpt,strict=True); model.eval()
        for ci,cond in enumerate(CONDITIONS):
            block=[]
            for i,(d,c) in enumerate(ps):
                x=perturb(load_raw(d),cond,i); gt=load_raw(c)
                xt=torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
                xt=F.pad(xt,PAD,mode='reflect')
                with torch.inference_mode(),torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    pred=model(xt)[:,:,12:52].squeeze().float().cpu().numpy()
                pred=np.clip(pred,0,1)
                rec={'method':method,'condition':cond['id'],'family':cond['family'],'filename':c.name,**volume_metrics(pred,gt,COPPER)}
                rows.append(rec); block.append(rec)
                if (i+1)%50==0: print(f"[{method}] {cond['id']} {i+1}/200",flush=True)
            summaries.setdefault(method,{})[cond['id']]=summarize_block(block)
        del model; gc.collect(); torch.cuda.empty_cache()

    save_rows(rows)
    payload={'created_at':datetime.now(timezone.utc).isoformat(),'inference_only':True,'samples':200,'seed':SEED,
             'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
             'platform':platform.platform(),'torch':torch.__version__,'conditions':CONDITIONS,
             'checkpoints':{k:str(v) for k,v in CHECKPOINTS.items()},'summaries':summaries,
             'wall_time_seconds':time.perf_counter()-started}
    (OUT/'noise_robustness_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'noise_robustness_summary.json')

if __name__=='__main__': main()
