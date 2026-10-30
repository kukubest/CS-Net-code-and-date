"""Select representative PCB GT patches and create a publication-style gallery."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r"G:\CT_RE\CT_DRA")
DATA=Path(r"G:\CT_RE\dataset\Final_Adaptive_GT\independent_test")
OUT=ROOT/'output'/'figures'; TMP=ROOT/'tmp'/'pcb_gallery'; OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
SHAPE=(40,256,256); Z=8

with (DATA/'manifest.csv').open(encoding='utf-8-sig') as f: manifest={r['filename']:r for r in csv.DictReader(f)}

def read_slice(path): return np.fromfile(path,dtype=np.float32).reshape(SHAPE)[Z]
def score(im):
    mask=im>0.9; occ=float(mask.mean())
    edges=float(np.abs(np.diff(mask.astype(np.float32),axis=0)).mean()+np.abs(np.diff(mask.astype(np.float32),axis=1)).mean())
    # Prefer informative but not saturated patches; reward topology and balanced coverage.
    return edges*4.0 - abs(occ-0.22)*0.18

groups={1:[],2:[],4:[]}
for path in sorted(DATA.glob('*.raw')):
    scale=int(manifest[path.name]['scale']); im=read_slice(path); groups[scale].append((score(im),path,im))
for scale in groups: groups[scale].sort(key=lambda x:x[0],reverse=True)

# Diagnostic contact sheet: top candidates per scale.
fig,axs=plt.subplots(3,8,figsize=(12,4.7),dpi=180)
for r,scale in enumerate((1,2,4)):
    for c,(s,path,im) in enumerate(groups[scale][:8]):
        axs[r,c].imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); axs[r,c].axis('off')
        axs[r,c].set_title(f'{scale}x #{c+1}\n{path.stem.split("patch")[-1].split("_")[0]}',fontsize=7)
    axs[r,0].set_ylabel(f'Scale {scale}x',fontsize=9)
fig.tight_layout(pad=.5); fig.savefig(TMP/'candidate_contact_sheet.png',bbox_inches='tight'); plt.close(fig)

# Hand-off defaults selected from the ranked candidates: two complementary samples per scale.
pick_indices={1:(0,3),2:(0,4),4:(0,5)}
selected=[]
for row in range(2):
    for scale in (1,2,4):
        rank=pick_indices[scale][row]
        selected.append((scale,*groups[scale][rank]))

fig,axs=plt.subplots(2,3,figsize=(9.2,5.8),dpi=300)
letters='abcdef'
for i,(scale,s,path,im) in enumerate(selected):
    ax=axs.flat[i]
    ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_linewidth(.7); spine.set_color('#555555')
    ax.set_title(f'({letters[i]})  Scale {scale}×',fontsize=11,pad=6)
fig.subplots_adjust(left=.025,right=.99,top=.94,bottom=.03,wspace=.055,hspace=.14)
final=OUT/'pcb_dataset_representative_samples.png'
fig.savefig(final,dpi=300,bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'pcb_dataset_representative_samples.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

(OUT/'pcb_dataset_representative_samples.txt').write_text('\n'.join(f'({letters[i]}) scale={x[0]} file={x[2].name} score={x[1]:.6f}' for i,x in enumerate(selected)),encoding='utf-8')
print(TMP/'candidate_contact_sheet.png'); print(final)
