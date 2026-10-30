"""Unified inference-only evaluation on the 200-sample independent PCB test set."""

import argparse, csv, gc, json, math, os, platform, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import astra
from monai.networks.nets import UNet, UNETR, SwinUNETR

from ablation_metrics import volume_metrics, summarize_metric_rows
from ablation_models import CSNet
from generate_fdk_baseline import create_cl_vector_geometry

ROOT = Path(r"G:\CT_RE\CT_DRA")
DIRTY = Path(r"G:\CT_RE\dataset\Final_Adaptive_Dirty\independent_test")
CLEAN = Path(r"G:\CT_RE\dataset\Final_Adaptive_GT\independent_test")
OUT = ROOT / "outputs" / "unified_200_test"
SHAPE = (40, 256, 256)
PAD = (0, 0, 0, 0, 12, 12)
COPPER = {"enabled": True, "prediction_threshold": 0.91}
SEED = 20260916

CHECKPOINTS = {
    "3D U-Net": ROOT / "checkpoints" / "3d_unet_best.pth",
    "UNETR": ROOT / "checkpoints" / "unetr_baseline_best.pth",
    "Swin-UNETR": ROOT / "checkpoints" / "swin_unetr_best.pth",
    "CS-Net": ROOT / "outputs" / "ablation" / "csnet_l1" / "seed_20260916" / "best_val.pt",
}

def pairs(limit=None):
    dirty = sorted(DIRTY.glob("*_dirty.raw"))
    result=[]
    for d in dirty:
        c = CLEAN / d.name.replace("_dirty.raw", ".raw")
        if not c.is_file(): raise FileNotFoundError(c)
        if d.stat().st_size != np.prod(SHAPE)*4 or c.stat().st_size != np.prod(SHAPE)*4:
            raise ValueError(f"Unexpected RAW size: {d} / {c}")
        result.append((d,c))
    if len(result) != 200: raise RuntimeError(f"Expected 200 paired samples, found {len(result)}")
    return result[:limit] if limit else result

def load_raw(path): return np.fromfile(path,dtype=np.float32).reshape(SHAPE)

def write_rows(name, rows, extra):
    safe=name.lower().replace(" ","_").replace("-","_")
    csv_path=OUT/f"{safe}_per_sample.csv"
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    payload={"method":name,"samples":len(rows),"summary":summarize_metric_rows(rows),**extra}
    (OUT/f"{safe}_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload

def evaluate_arrays(name, iterator, total, extra):
    rows=[]
    started=time.perf_counter()
    for i,(filename,pred,gt) in enumerate(iterator,1):
        pred=np.clip(np.asarray(pred,dtype=np.float32),0,1)
        row={"filename":filename,**volume_metrics(pred,gt,COPPER)}
        rows.append(row)
        if i==1 or i%20==0: print(f"[{name}] {i}/{total} PSNR={row['psnr']:.3f} SSIM={row['ssim']:.5f}",flush=True)
    extra=dict(extra); extra["wall_time_seconds"]=time.perf_counter()-started
    return write_rows(name,rows,extra)

def sirt_iterator(ps):
    for d,c in ps: yield c.name,load_raw(d),load_raw(c)

def add_poisson_same_stream(sinogram, I0=5e4):
    photons=I0*np.exp(-sinogram); noisy=np.random.poisson(photons); noisy[noisy==0]=1
    return -np.log(noisy/I0)

def fdk_iterator(ps):
    np.random.seed(SEED)
    vol_geom=astra.create_vol_geom(SHAPE[2],SHAPE[1],SHAPE[0])
    vectors=create_cl_vector_geometry(180,45.0,500.0,500.0,1.0)
    proj_geom=astra.create_proj_geom('cone_vec',300,300,vectors)
    for _,c in ps:
        gt=load_raw(c)
        sino_id,sino=astra.create_sino3d_gpu(gt,proj_geom,vol_geom)
        noisy=add_poisson_same_stream(sino,5e4)
        noisy_id=astra.data3d.create('-sino',proj_geom,noisy)
        recon_id=astra.data3d.create('-vol',vol_geom)
        cfg=astra.astra_dict('FDK_CUDA'); cfg['ReconstructionDataId']=recon_id; cfg['ProjectionDataId']=noisy_id
        alg_id=astra.algorithm.create(cfg); astra.algorithm.run(alg_id)
        pred=astra.data3d.get(recon_id)
        astra.algorithm.delete(alg_id); astra.data3d.delete(recon_id); astra.data3d.delete(sino_id); astra.data3d.delete(noisy_id)
        pred=(pred-pred.min())/(pred.max()-pred.min()+1e-8)
        yield c.name,pred.astype(np.float32),gt

def make_model(name):
    if name=="3D U-Net":
        return UNet(spatial_dims=3,in_channels=1,out_channels=1,channels=(16,32,64,128,256),strides=(2,2,2,2),num_res_units=2,norm="instance")
    if name=="UNETR":
        return UNETR(spatial_dims=3,in_channels=1,out_channels=1,img_size=(64,256,256),feature_size=16,hidden_size=768,mlp_dim=3072,num_heads=12,norm_name="instance",res_block=True)
    if name=="Swin-UNETR":
        return SwinUNETR(spatial_dims=3,in_channels=1,out_channels=1,feature_size=24,use_checkpoint=False)
    if name=="CS-Net":
        return CSNet(in_channels=1,out_channels=1,feature_size=24,use_checkpoint=False,compression=True,hr_skip=True)
    raise KeyError(name)

def deep_iterator(name, ps, device, timing):
    model=make_model(name).to(device)
    obj=torch.load(CHECKPOINTS[name],map_location="cpu",weights_only=False)
    state=obj["model"] if name=="CS-Net" else obj
    model.load_state_dict(state,strict=True); model.eval()
    infer=[]; peak=[]
    try:
        for d,c in ps:
            x=torch.from_numpy(load_raw(d)).unsqueeze(0).unsqueeze(0).to(device)
            gt=load_raw(c)
            x=F.pad(x,PAD,mode="reflect")
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0=time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                y=model(x)[:,:,12:52]
            if device.type=="cuda": torch.cuda.synchronize()
            infer.append((time.perf_counter()-t0)*1000)
            peak.append(torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0)
            yield c.name,y.squeeze().float().cpu().numpy(),gt
    finally:
        timing.update({"inference_ms_mean":float(np.mean(infer)) if infer else None,"inference_ms_std":float(np.std(infer)) if infer else None,"peak_vram_gib_max":float(max(peak)) if peak else None})
        del model; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); ap.add_argument('--skip-fdk',action='store_true'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); ps=pairs(args.limit); total=len(ps)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest={"created_at":datetime.now(timezone.utc).isoformat(),"inference_only":True,"samples":total,"seed":SEED,"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=='cuda' else None,"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"checkpoints":{k:str(v) for k,v in CHECKPOINTS.items()},"methods":{}}
    manifest["methods"]["SIRT"]=evaluate_arrays("SIRT",sirt_iterator(ps),total,{"source":"existing paired initial reconstruction; no model"})
    if not args.skip_fdk:
        manifest["methods"]["FDK"]=evaluate_arrays("FDK",fdk_iterator(ps),total,{"geometry":"180 projections, 45 deg, SO=500, OD=500, detector=300x300, spacing=1","noise":"Poisson I0=5e4; RNG reset to independent-test seed"})
    for name in ["3D U-Net","UNETR","Swin-UNETR","CS-Net"]:
        timing={}
        manifest["methods"][name]=evaluate_arrays(name,deep_iterator(name,ps,device,timing),total,{"checkpoint":str(CHECKPOINTS[name]),"timing":timing})
    (OUT/'unified_results.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'unified_results.json')

if __name__=='__main__': main()
