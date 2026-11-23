"""Figures for the paper: registration check, comparators, example.

Requires results/patient_level_stats.json (scripts/run_patient_level_stats.py).
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
# Usage: python make_paper_figures.py <v2 dir> <output figure dir>
D=sys.argv[1]; OUT=sys.argv[2]
INK='#0b0b0b'; SEC='#52514e'; MUT='#898781'; GRID='#e1e0d9'; BASE='#c3c2b7'
BLUE='#2a78d6'; ORANGE='#eb6834'; RAMP={2:'#86b6ef',4:'#2a78d6',8:'#104281'}
plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'axes.edgecolor':BASE,'axes.labelcolor':SEC,'xtick.color':MUT,'ytick.color':MUT,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
def style(ax):
    ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2)

# ---------- registration ----------
r=pd.read_csv(f"{D}/figure_data/fig4_registration_offsets.csv"); r=r[r.label.isin(["Meniscus Tear","Cartilage - Partial Thickness loss/defect"])]
fig,ax=plt.subplots(figsize=(3.3,1.75))
bins=np.arange(0,131,5)
ax.hist(r.offset_corrected,bins=bins,color=BLUE,edgecolor='#fcfcfb',lw=0.8,label='bottom-left origin (correct)')
ax.hist(r.offset_as_given,bins=bins,color=ORANGE,edgecolor='#fcfcfb',lw=0.8,label='top-left origin (incorrect)')
ax.axvline(15,color=SEC,lw=0.8,ls=(0,(3,2))); ax.text(16,ax.get_ylim()[1]*0.92,'15 px',color=SEC,fontsize=6.5,va='top')
ax.set_xlabel('distance from box centre to joint line (pixels)'); ax.set_ylabel('meniscus + cartilage boxes')
ax.legend(frameon=False,loc='upper right'); style(ax)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_registration.pdf"); fig.savefig(f"{OUT}/fig_registration.png",dpi=200)

# ---------- comparators + distance ----------
st=json.load(open(f"{D}/results/patient_level_stats.json"))
def g(s_): return s_['mean'],s_['ci_lo'],s_['ci_hi']
t1=st['table1']['cg_ssim']; cart=st['table2']['cartilage']
comps=['size-matched','appearance-\nmatched','same position,\nall slices','same position,\n$\\leq$3 slices','same position,\nadjacent slice']
keys=['sm','am','sp_all','sp_d3','sp_d1']
P=[g(t1[k]) for k in keys]; C=[g(cart[k]) for k in keys]
fig,axs=plt.subplots(2,1,figsize=(3.3,3.75),gridspec_kw={'height_ratios':[1.1,1]})
ax=axs[0]; y=np.arange(len(comps))[::-1]
for vals,off,col,mk,name in [(P,0.14,BLUE,'o','all findings'),(C,-0.14,ORANGE,'s','cartilage defects')]:
    m=np.array([v[0] for v in vals]); lo=np.array([v[1] for v in vals]); hi=np.array([v[2] for v in vals])
    ax.errorbar(m,y+off,xerr=[m-lo,hi-m],fmt=mk,color=col,ms=4.5,lw=1.2,capsize=0,label=name,mec='#fcfcfb',mew=0.6)
ax.axvline(0,color=SEC,lw=0.8); ax.set_yticks(y); ax.set_yticklabels(comps,fontsize=6.8,color=SEC)
ax.set_xlabel('lesion effect, $\\Delta$SSIM (lesion $-$ comparator)'); ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False,loc='upper left',fontsize=6.5); ax.set_title('(a) CG-SENSE, nominal R = 4, by comparator',fontsize=7.5,color=INK,loc='left')
ax=axs[1]; labels=['1','2','3','4–5','6–9','≥10']; bk=['1','2','3','4-5','6-9','10+']; x=np.arange(6)
for R,mk in [(2,'o'),(4,'s'),(8,'^')]:
    v=[g(st['distance'][str(R)]['all'][k]) for k in bk]; m=np.array([a[0] for a in v]); lo=np.array([a[1] for a in v]); hi=np.array([a[2] for a in v])
    ax.fill_between(x,lo,hi,color=RAMP[R],alpha=0.18,lw=0); ax.plot(x,m,marker=mk,color=RAMP[R],lw=1.5,ms=4.5,mec='#fcfcfb',mew=0.6,label=f'R = {R}')
ax.axhline(0,color=SEC,lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlim(-0.3,5.3)
ax.set_xlabel('slices between lesion and comparator slice'); ax.set_ylabel('$\\Delta$SSIM, lesion $-$ same position')
ax.legend(frameon=False,loc='lower left',ncol=3,fontsize=6.5); style(ax)
ax.set_title('(b) all findings, by comparator distance',fontsize=7.5,color=INK,loc='left')
fig.tight_layout(pad=0.3,h_pad=0.8); fig.savefig(f"{OUT}/fig_comparators.pdf"); fig.savefig(f"{OUT}/fig_comparators.png",dpi=200)

# ---------- example ----------
meta=json.load(open(f"{D}/figure_data/example_meta.json"))
ref=np.load(f"{D}/figure_data/example_ref.npy"); cg=np.load(f"{D}/figure_data/example_cg.npy")
err=np.load(f"{D}/figure_data/example_err.npy"); ss=np.load(f"{D}/figure_data/example_ssim.npy")
b=meta['box_array_indices']; r0,c0,h,w=b['row_top'],b['col_left'],b['h'],b['w']
vmax=np.percentile(ref,99.5)
fig,axs=plt.subplots(1,5,figsize=(7.0,1.7),gridspec_kw={'width_ratios':[1,1,1,1,1]})
panels=[(ref,'Fully sampled reference',dict(cmap='gray',vmin=0,vmax=vmax)),(cg,'CG-SENSE, nominal R = 4',dict(cmap='gray',vmin=0,vmax=vmax)),
        (err,'Absolute error',dict(cmap='magma',vmin=0,vmax=np.percentile(err,99.5))),(ss,'SSIM map',dict(cmap='gray',vmin=0,vmax=1))]
for ax,(im,t,kw) in zip(axs[:4],panels):
    ax.imshow(im,**kw); ax.set_title(t,fontsize=7,color=INK); ax.axis('off')
    ax.add_patch(Rectangle((c0-0.5,r0-0.5),w,h,fill=False,ec='#1baf7a',lw=1.2))
for (rr,cc) in meta['size_matched_controls_row_col']:
    axs[0].add_patch(Rectangle((cc-0.5,rr-0.5),w,h,fill=False,ec='#ffffff',lw=0.9))
for (rr,cc) in meta['appearance_matched_controls_row_col']:
    axs[0].add_patch(Rectangle((cc-0.5,rr-0.5),w,h,fill=False,ec=ORANGE,lw=0.9))
pad=22; z=ref[max(0,r0-pad):r0+h+pad, max(0,c0-pad):c0+w+pad]; zc=cg[max(0,r0-pad):r0+h+pad, max(0,c0-pad):c0+w+pad]
axs[4].imshow(np.hstack([z,np.full((z.shape[0],2),vmax),zc]),cmap='gray',vmin=0,vmax=vmax); axs[4].axis('off')
axs[4].set_title('Zoom: reference | CG-SENSE',fontsize=7,color=INK)
for off in [0,z.shape[1]+2]:
    axs[4].add_patch(Rectangle((min(pad,c0)+off-0.5,min(pad,r0)-0.5),w,h,fill=False,ec='#1baf7a',lw=1.2))
fig.tight_layout(pad=0.2,w_pad=0.3); fig.savefig(f"{OUT}/fig_example.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{OUT}/fig_example.png",dpi=200,bbox_inches="tight",pad_inches=0.02)
print('ok', meta['file'], meta['slice'], r0,c0,h,w)
