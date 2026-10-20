#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PROJECT=Path.home()/'projects'/'nairobi_v2'
STEP18_HIGH=PROJECT/'results/xai_gradcam_v2/seed_42/representative_high_impact_v2/representative_high_impact_metrics_v2.csv'
STEP18_MEDIAN=PROJECT/'results/xai_gradcam_v2/seed_42/representative_median_v2/representative_median_metrics_v2.csv'
STEP19B=PROJECT/'results/xai_semantic_v2/seed_42/19b_coarse_semantic_grouping'
COARSE_CASE_CSV=STEP19B/'coarse_semantic_case_summary_final_v2.csv'
OUT_DIR=PROJECT/'results/xai_semantic_v2/seed_42/19c_semantic_gradcam_quantification'
CASEWISE_CSV=OUT_DIR/'semantic_gradcam_casewise_final_v2.csv'
CASE_META_CSV=OUT_DIR/'semantic_gradcam_case_meta_final_v2.csv'
SELECTION_CSV=OUT_DIR/'semantic_gradcam_selection_summary_final_v2.csv'
EFFECT_CSV=OUT_DIR/'semantic_gradcam_effect_summary_final_v2.csv'
TARGET_CSV=OUT_DIR/'semantic_gradcam_target_summary_final_v2.csv'
AUDIT_JSON=OUT_DIR/'semantic_gradcam_quantification_audit_final_v2.json'
NOTES_TXT=OUT_DIR/'semantic_gradcam_quantification_notes_final_v2.txt'
QC_AREA=OUT_DIR/'qc_area_fraction_all16_final_v2.png'
QC_RESPONSE=OUT_DIR/'qc_gradcam_response_fraction_all16_final_v2.png'
QC_ENRICH=OUT_DIR/'qc_semantic_enrichment_all16_final_v2.png'

GROUPS=['sky','built_structure','road_surface','bare_ground','vegetation','dynamic_objects','other_uncertain']
RGB={'sky':(145,190,225),'built_structure':(203,111,91),'road_surface':(125,129,136),'bare_ground':(194,154,100),'vegetation':(93,154,93),'dynamic_objects':(139,119,170),'other_uncertain':(190,130,170)}
ORDER=['fp_to_correct','fn_to_correct','correct_to_fp','correct_to_fn']
SHORT={'fp_to_correct':'FP→C','fn_to_correct':'FN→C','correct_to_fp':'C→FP','correct_to_fn':'C→FN'}
EFFECT={'fp_to_correct':'helpful','fn_to_correct':'helpful','correct_to_fp':'harmful','correct_to_fn':'harmful'}
TARGET={'fp_to_correct':'non_informal','fn_to_correct':'informal','correct_to_fp':'informal','correct_to_fn':'non_informal'}
ALIASES={'fp_to_correct':'fp_to_correct','s2_fp_to_correct':'fp_to_correct','fn_to_correct':'fn_to_correct','s2_fn_to_correct':'fn_to_correct','correct_to_fp':'correct_to_fp','s2_correct_to_fusion_fp':'correct_to_fp','correct_to_fn':'correct_to_fn','s2_correct_to_fusion_fn':'correct_to_fn'}

def read_csv(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    if not rows: raise ValueError(f'No rows: {path}')
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def resolve(v):
    p=Path(str(v))
    if p.is_file(): return p
    marker='projects/nairobi_v2/'
    if marker in str(p):
        q=PROJECT/str(p).split(marker,1)[1]
        if q.is_file(): return q
    return p

def canon_st(x):
    s=str(x).strip().lower()
    if s not in ALIASES: raise ValueError(f'Unknown subtype: {x}')
    return ALIASES[s]

def canon_k(x):
    s=str(x).replace(' ','')
    if s in {'K1','K=1','1'}: return 'K1'
    if s in {'K>=2','K≥2','>=2','≥2'}: return 'K>=2'
    raise ValueError(f'Unknown K: {x}')

def key(r): return (r['selection'],canon_st(r['subtype']),canon_k(r['k_stratum']),int(float(r['covered_order'])),int(float(r['target_embedding_row'])))

def load_inputs():
    s18={}
    for sel,path in [('high_impact',STEP18_HIGH),('median',STEP18_MEDIAN)]:
        rows=read_csv(path)
        if len(rows)!=8: raise ValueError(f'{sel}: expected 8 Step-18 cases')
        for r in rows:
            rr={'selection':sel,'subtype':canon_st(r['subtype']),'k_stratum':canon_k(r['k_stratum']),'covered_order':int(float(r['covered_order'])),'target_embedding_row':int(float(r['target_embedding_row'])),'heatmap_npy':resolve(r['heatmap_npy'])}
            s18[key(rr)]=rr
    s19={}
    rows=read_csv(COARSE_CASE_CSV)
    if len(rows)!=16: raise ValueError(f'Expected 16 Step-19b cases, found {len(rows)}')
    for r in rows:
        rr={'selection':r['selection'],'subtype':canon_st(r['subtype']),'k_stratum':canon_k(r['k_stratum']),'covered_order':int(float(r['covered_order'])),'target_embedding_row':int(float(r['target_embedding_row'])),'coarse_mask_npy':resolve(r['coarse_mask_npy'])}
        s19[key(rr)]=rr
    if set(s18)!=set(s19): raise ValueError('Step-18 and Step-19b case sets do not match exactly.')
    return s18,s19

def aggregate(rows,fields):
    buckets=defaultdict(list)
    for r in rows: buckets[tuple(r[f] for f in fields)+(r['coarse_group'],)].append(r)
    out=[]
    for kk,b in sorted(buckets.items()):
        base={f:kk[i] for i,f in enumerate(fields)}; g=kk[-1]
        e=np.array([float(r['enrichment']) for r in b if np.isfinite(float(r['enrichment']))]); l=np.array([float(r['log2_enrichment']) for r in b if np.isfinite(float(r['log2_enrichment']))])
        out.append({**base,'coarse_group':g,'n_cases':len(b),'n_valid_enrichment':int(e.size),'mean_area_fraction':float(np.mean([float(r['area_fraction']) for r in b])),'mean_response_fraction':float(np.mean([float(r['gradcam_response_fraction']) for r in b])),'median_enrichment':float(np.median(e)) if e.size else float('nan'),'median_log2_enrichment':float(np.median(l)) if l.size else float('nan')})
    return out

def case_label(m): return f"{'Median' if m['selection']=='median' else 'High'} | {SHORT[m['subtype']]} | {'K1' if m['k_stratum']=='K1' else 'K≥2'}"

def stacked(rows,metas,metric,ylabel,out):
    lookup=defaultdict(dict)
    for r in rows:
        kk=(r['selection'],r['subtype'],r['k_stratum'],int(r['covered_order']),int(r['target_embedding_row'])); lookup[kk][r['coarse_group']]=float(r[metric])
    x=np.arange(len(metas)); bottom=np.zeros(len(metas)); fig,ax=plt.subplots(figsize=(190/25.4,96/25.4),dpi=180)
    for g in GROUPS:
        vals=[]
        for m in metas:
            kk=(m['selection'],m['subtype'],m['k_stratum'],int(m['covered_order']),int(m['target_embedding_row'])); vals.append(lookup[kk].get(g,0))
        vals=np.asarray(vals); ax.bar(x,vals,bottom=bottom,label=g.replace('_',' '),color=np.asarray(RGB[g])/255); bottom+=vals
    ax.set_ylabel(ylabel,fontsize=8); ax.set_xticks(x); ax.set_xticklabels([case_label(m) for m in metas],rotation=62,ha='right',fontsize=5.8); ax.set_ylim(0,1); ax.tick_params(axis='y',labelsize=6.5)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.34),ncol=4,fontsize=6,frameon=False); fig.subplots_adjust(left=.075,right=.995,top=.985,bottom=.42); fig.savefig(out,dpi=300,bbox_inches='tight',pad_inches=.02); plt.close(fig)

def enrich_plot(rows,metas,out):
    lookup=defaultdict(dict)
    for r in rows:
        kk=(r['selection'],r['subtype'],r['k_stratum'],int(r['covered_order']),int(r['target_embedding_row'])); lookup[kk][r['coarse_group']]=float(r['log2_enrichment'])
    mat=np.full((len(metas),len(GROUPS)),np.nan)
    for i,m in enumerate(metas):
        kk=(m['selection'],m['subtype'],m['k_stratum'],int(m['covered_order']),int(m['target_embedding_row']))
        for j,g in enumerate(GROUPS): mat[i,j]=lookup[kk].get(g,np.nan)
    valid=mat[np.isfinite(mat)]; vmax=max(1,float(np.max(np.abs(valid)))) if valid.size else 1
    cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('white'); norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    fig,ax=plt.subplots(figsize=(170/25.4,108/25.4),dpi=180); im=ax.imshow(np.ma.masked_invalid(mat),aspect='auto',cmap=cmap,norm=norm)
    ax.set_xticks(np.arange(len(GROUPS))); ax.set_xticklabels([g.replace('_',' ') for g in GROUPS],rotation=32,ha='right',fontsize=6); ax.set_yticks(np.arange(len(metas))); ax.set_yticklabels([case_label(m) for m in metas],fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i,j]): ax.text(j,i,'N/A',ha='center',va='center',fontsize=4.8)
    cb=fig.colorbar(im,ax=ax,fraction=.032,pad=.028); cb.set_label('log2 enrichment',fontsize=7); cb.ax.tick_params(labelsize=6); fig.subplots_adjust(left=.25,right=.91,top=.985,bottom=.20); fig.savefig(out,dpi=300,bbox_inches='tight',pad_inches=.02); plt.close(fig)

def main():
    print('='*96); print('Nairobi v2 — Step 19c Final Semantic × Grad-CAM Quantification'); print('='*96)
    print('New segmentation inference: NO'); print('Grad-CAM recomputation: NO'); print('Hotspot/confidence threshold: NO'); print('Metrics: area fraction, continuous response fraction, enrichment\n')
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    s18,s19=load_inputs(); ordered=sorted(s18,key=lambda kk:(0 if kk[0]=='median' else 1,ORDER.index(kk[1]),0 if kk[2]=='K1' else 1))
    rows=[]; metas=[]
    for i,kk in enumerate(ordered,1):
        heat=np.squeeze(np.asarray(np.load(s18[kk]['heatmap_npy']))).astype(float); coarse=np.squeeze(np.asarray(np.load(s19[kk]['coarse_mask_npy']))).astype(int)
        if heat.shape!=(128,500) or coarse.shape!=heat.shape: raise ValueError(f'Alignment failure {kk}: {heat.shape}, {coarse.shape}')
        if not np.all(np.isfinite(heat)) or heat.min()<-1e-5: raise ValueError(f'Invalid Grad-CAM {kk}')
        heat=np.clip(heat,0,None); total=float(heat.sum())
        if total<=0: raise ValueError(f'Zero Grad-CAM response {kk}')
        unique_ids = set(int(x) for x in np.unique(coarse))
        valid_ids = set(range(len(GROUPS)))
        if not unique_ids.issubset(valid_ids):
            raise ValueError(f'Unexpected coarse semantic IDs for {kk}: {sorted(unique_ids - valid_ids)}')

        metrics={}
        for gi,g in enumerate(GROUPS):
            m=coarse==gi; area=float(m.mean()); resp=float(heat[m].sum()/total); enr=resp/area if area>0 else float('nan'); log2=math.log2(enr) if np.isfinite(enr) and enr>0 else float('nan'); metrics[g]=(area,resp,enr,log2)
            rows.append({'selection':kk[0],'subtype':kk[1],'effect_role':EFFECT[kk[1]],'target_direction':TARGET[kk[1]],'k_stratum':kk[2],'covered_order':kk[3],'target_embedding_row':kk[4],'coarse_group':g,'area_fraction':area,'gradcam_response_fraction':resp,'enrichment':enr,'log2_enrichment':log2})

        area_sum = sum(metrics[g][0] for g in GROUPS)
        response_sum = sum(metrics[g][1] for g in GROUPS)
        if not np.isclose(area_sum, 1.0, atol=1e-10):
            raise ValueError(f'Area fractions do not sum to 1 for {kk}: {area_sum}')
        if not np.isclose(response_sum, 1.0, atol=1e-8):
            raise ValueError(f'Grad-CAM response fractions do not sum to 1 for {kk}: {response_sum}')
        dom=max(GROUPS,key=lambda g:metrics[g][1]); metas.append({'selection':kk[0],'subtype':kk[1],'effect_role':EFFECT[kk[1]],'target_direction':TARGET[kk[1]],'k_stratum':kk[2],'covered_order':kk[3],'target_embedding_row':kk[4],'dominant_response_group':dom,'dominant_response_fraction':metrics[dom][1],'dynamic_objects_response_fraction':metrics['dynamic_objects'][1],'other_uncertain_response_fraction':metrics['other_uncertain'][1],'core_semantic_response_fraction':1-metrics['dynamic_objects'][1]-metrics['other_uncertain'][1]})
        print(f"[{i:02d}/16] {kk[0]} | {kk[1]} | {kk[2]} | dominant={dom} ({100*metrics[dom][1]:.1f}%)")
    write_csv(CASEWISE_CSV,rows); write_csv(CASE_META_CSV,metas); write_csv(SELECTION_CSV,aggregate(rows,['selection'])); write_csv(EFFECT_CSV,aggregate(rows,['effect_role'])); write_csv(TARGET_CSV,aggregate(rows,['target_direction']))
    stacked(rows,metas,'area_fraction','Image area fraction',QC_AREA); stacked(rows,metas,'gradcam_response_fraction','Grad-CAM response fraction',QC_RESPONSE); enrich_plot(rows,metas,QC_ENRICH)
    mean_dyn=float(np.mean([m['dynamic_objects_response_fraction'] for m in metas])); mean_other=float(np.mean([m['other_uncertain_response_fraction'] for m in metas])); mean_core=float(np.mean([m['core_semantic_response_fraction'] for m in metas]))
    AUDIT_JSON.write_text(json.dumps({'status':'PASSED','step':'19c','n_cases':16,'population_level_claim_supported':False,'new_segmentation_inference':False,'gradcam_recomputed':False,'hotspot_threshold':False,'confidence_threshold':False,'mean_dynamic_objects_response_fraction':mean_dyn,'mean_other_uncertain_response_fraction':mean_other,'mean_core_semantic_response_fraction':mean_core,'interpretation_scope':'descriptive interpretation of the same 16 frozen representative cases only'},indent=2),encoding='utf-8')
    NOTES_TXT.write_text('Step 19c: continuous pixel-wise coarse semantic × directional Grad-CAM quantification.\n',encoding='utf-8')
    print('\nSTATUS: PASSED'); print(f'Mean dynamic-object response fraction: {mean_dyn:.4f}'); print(f'Mean other/uncertain response fraction: {mean_other:.4f}'); print(f'Mean core interpretable response fraction: {mean_core:.4f}'); print(f'Output: {OUT_DIR}')

if __name__=='__main__': main()
