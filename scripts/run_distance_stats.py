"""Same-position lesion effect stratified by slice distance (R = 2, 4, 8).

Usage: python run_distance_stats.py <v2 results dir>
Writes <results dir>/distance_stats.json
"""
import sys, numpy as np, pandas as pd, json
from scipy.stats import wilcoxon
D=sys.argv[1]; rng=np.random.default_rng(0)
key=['file','label','row_top','x','w','h']
def load(R):
    if R==4:
        Rg=pd.read_csv(f"{D}/regions_R4.csv").rename(columns={'ssim_les_cg':'les','ssim_sizectl_cg':'ctl'})
        P=pd.read_csv(f"{D}/placebo_R4.csv").rename(columns={'ssim_les_cg':'les','ssim_sizectl_cg':'ctl'})
    else:
        Rg=pd.read_csv(f"{D}/accel_regions_R{R}.csv").rename(columns={'ssim_les':'les','ssim_ctl':'ctl'})
        P=pd.read_csv(f"{D}/accel_placebo_R{R}.csv").rename(columns={'ssim_les':'les','ssim_ctl':'ctl'})
    P['dist']=(P.slice-P.src_slice).abs(); return Rg,P
def boot(df,col):
    files=df.file.unique(); groups={f:df[df.file==f][col].values for f in files}
    b=[np.concatenate([groups[f] for f in rng.choice(files,len(files))]).mean() for _ in range(3000)]
    pm=df.groupby('file')[col].mean()
    p=wilcoxon(pm).pvalue if len(pm)>=5 else np.nan
    return dict(mean=float(df[col].mean()),lo=float(np.percentile(b,2.5)),hi=float(np.percentile(b,97.5)),n=int(len(df)),pts=int(len(pm)),pos=int((pm>0).sum()),neg=int((pm<0).sum()),p=float(p))
out={}
bins=[(1,1),(2,2),(3,3),(4,5),(6,9),(10,99)]
for R in [2,4,8]:
    Rg,P=load(R); out[R]={}
    for lab in ['ALL','Cartilage','Meniscus']:
        out[R][lab]={}
        for lo,hi in bins+[(1,3),(1,99)]:
            Pk=P[(P.dist>=lo)&(P.dist<=hi)].groupby(key+['src_slice']).les.mean().rename('sp').reset_index()
            M=Rg.merge(Pk,left_on=key+['slice'],right_on=key+['src_slice']); M['direct']=M.les-M.sp
            if lab=='Cartilage': M=M[M.label.str.startswith('Cartilage')]
            if lab=='Meniscus': M=M[M.label=='Meniscus Tear']
            if len(M)==0: continue
            out[R][lab][f"{lo}-{hi}"]=boot(M.reset_index(drop=True),'direct')
for R in out:
    for lab in out[R]:
        for b,s in out[R][lab].items():
            print(R,lab,b,f"{s['mean']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] n={s['n']} pts={s['pts']} {s['pos']}+/{s['neg']}- p={s['p']:.4f}")
json.dump(out,open(f"{D}/distance_stats.json",'w'),indent=1)


# ---- Table 1 rows: zero-filled and PSNR at R=4 (separate RNG stream) ----
rng=np.random.default_rng(1)
key=['file','label','row_top','x','w','h']
R=pd.read_csv(f"{D}/regions_R4.csv"); P=pd.read_csv(f"{D}/placebo_R4.csv"); P['dist']=(P.slice-P.src_slice).abs()
def boot(df,col):
    files=df.file.unique(); g={f:df[df.file==f][col].values for f in files}
    b=[np.concatenate([g[f] for f in rng.choice(files,len(files))]).mean() for _ in range(3000)]
    pm=df.groupby('file')[col].mean(); p=wilcoxon(pm).pvalue if len(pm)>=5 else np.nan
    return f"{df[col].mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] n={len(df)} pts={len(pm)} {int((pm>0).sum())}/{int((pm<0).sum())} p={p:.3f}"
for met,lc in [('ssim_cg','ssim_les_cg'),('ssim_zf','ssim_les_zf'),('psnr_cg','psnr_les_cg'),('psnr_zf','psnr_les_zf')]:
  for lo,hi in [(1,1),(1,3),(1,99)]:
    Pk=P[(P.dist>=lo)&(P.dist<=hi)].replace([np.inf,-np.inf],np.nan).groupby(key+['src_slice'])[lc].mean().rename('sp').reset_index()
    M=R.merge(Pk,left_on=key+['slice'],right_on=key+['src_slice']); M['d']=M[lc]-M.sp; M=M.replace([np.inf,-np.inf],np.nan).dropna(subset=['d']).reset_index(drop=True)
    print(met,f"{lo}-{hi}",boot(M,'d'))
