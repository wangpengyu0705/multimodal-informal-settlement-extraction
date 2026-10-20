#!/usr/bin/env python3
from __future__ import annotations

import csv, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path.home() / 'projects' / 'nairobi_v2'
MODEL_CONFIG = PROJECT / 'models/pretrained/segformer_b2_ade20k_v2/config.json'
STEP19A = PROJECT / 'results/xai_semantic_v2/seed_42/19a_fine_semantic_segmentation'
FINE_CASE_CSV = STEP19A / 'fine_semantic_case_summary_final_v2.csv'

OUT_DIR = PROJECT / 'results/xai_semantic_v2/seed_42/19b_coarse_semantic_grouping'
MASK_DIR = OUT_DIR / 'coarse_masks'
COLOR_DIR = OUT_DIR / 'coarse_segmentation_color'
OVERLAY_DIR = OUT_DIR / 'coarse_segmentation_overlay'
CASE_CSV = OUT_DIR / 'coarse_semantic_case_summary_final_v2.csv'
FRACTION_CSV = OUT_DIR / 'coarse_semantic_fractions_final_v2.csv'
MAPPING_JSON = OUT_DIR / 'coarse_semantic_mapping_final_v2.json'
AUDIT_JSON = OUT_DIR / 'coarse_semantic_audit_final_v2.json'
NOTES_TXT = OUT_DIR / 'coarse_semantic_notes_final_v2.txt'
QC_MEDIAN = OUT_DIR / 'qc_coarse_semantic_median_final_v2.png'
QC_HIGH = OUT_DIR / 'qc_coarse_semantic_high_impact_final_v2.png'

COARSE_GROUPS = ['sky','built_structure','road_surface','bare_ground','vegetation','dynamic_objects','other_uncertain']
LABEL_SETS = {
    'sky': {'sky'},
    'built_structure': {'building','wall','fence','house','skyscraper','hovel','tower','bridge','column','railing','door','windowpane','awning','booth'},
    'road_surface': {'road','sidewalk','path','runway','stairs','stairway','step','escalator'},
    'bare_ground': {'earth','sand','dirt track','land','rock'},
    'vegetation': {'tree','grass','plant','flower','palm'},
    'dynamic_objects': {'person','car','bus','truck','van','bicycle','minibike','airplane','boat','ship'},
}
RGB = {
    'sky': (145,190,225), 'built_structure': (203,111,91),
    'road_surface': (125,129,136), 'bare_ground': (194,154,100),
    'vegetation': (93,154,93), 'dynamic_objects': (139,119,170),
    'other_uncertain': (190,130,170),
}
ALPHA = 0.8
SUBTYPE_ORDER = ['fp_to_correct','fn_to_correct','correct_to_fp','correct_to_fn']
SUBTYPE_LABEL = {
    'fp_to_correct':'S2 FP → Fusion correct', 'fn_to_correct':'S2 FN → Fusion correct',
    'correct_to_fp':'S2 correct → Fusion FP', 'correct_to_fn':'S2 correct → Fusion FN',
}

def read_csv(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows):
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

def load_id2label():
    cfg=json.loads(MODEL_CONFIG.read_text(encoding='utf-8'))
    return {int(k):str(v).strip().lower() for k,v in cfg['id2label'].items()}

def group_for(label):
    x=str(label).strip().lower()
    for g,labels in LABEL_SETS.items():
        if x in labels: return g
    return 'other_uncertain'

def render(mask):
    out=np.zeros((*mask.shape,3),dtype=np.uint8)
    for gi,g in enumerate(COARSE_GROUPS): out[mask==gi]=RGB[g]
    return out

def overlay(img, seg):
    return np.clip((1-ALPHA)*img.astype(np.float32)+ALPHA*seg.astype(np.float32),0,255).astype(np.uint8)

def stem(r):
    k='k1' if r['k_stratum']=='K1' else 'kge2'
    return f"{r['selection']}__{r['subtype']}__{k}__order{int(float(r['covered_order']))}__emb{int(float(r['target_embedding_row']))}"

def build_qc(selection, rows):
    lookup={(r['subtype'],r['k_stratum']):r for r in rows if r['selection']==selection}
    fig=plt.figure(figsize=(190/25.4,112/25.4),dpi=180)
    gs=fig.add_gridspec(6,5,width_ratios=[0.84,1,1,1,1],height_ratios=[0.18,0.20,1,1,1,1],left=.035,right=.995,top=.975,bottom=.045,wspace=.07,hspace=.22)
    ax=fig.add_subplot(gs[0,1:3]); ax.axis('off'); ax.text(.5,.45,'K = 1',ha='center',va='center',fontsize=9,fontweight='bold')
    ax=fig.add_subplot(gs[0,3:5]); ax.axis('off'); ax.text(.5,.45,'K ≥ 2',ha='center',va='center',fontsize=9,fontweight='bold')
    for j,t in enumerate(['Original','Coarse semantic overlay','Original','Coarse semantic overlay'],1):
        ax=fig.add_subplot(gs[1,j]); ax.axis('off'); ax.text(.5,.5,t,ha='center',va='center',fontsize=7.2)
    for i,st in enumerate(SUBTYPE_ORDER):
        row=i+2; axl=fig.add_subplot(gs[row,0]); axl.axis('off'); axl.text(.98,.5,f'({chr(97+i)}) {SUBTYPE_LABEL[st]}',ha='right',va='center',fontsize=6.9,fontweight='bold',wrap=True)
        for kidx,ks in enumerate(['K1','K>=2']):
            r=lookup[(st,ks)]
            original=np.asarray(Image.open(resolve(r['model_input_png'])).convert('RGB'))
            ov=np.asarray(Image.open(resolve(r['coarse_overlay_png'])).convert('RGB'))
            c0=1+2*kidx
            for col,img in [(c0,original),(c0+1,ov)]:
                ax=fig.add_subplot(gs[row,col]); ax.imshow(img); ax.set_axis_off()
    out=QC_MEDIAN if selection=='median' else QC_HIGH
    fig.savefig(out,dpi=300,bbox_inches='tight',pad_inches=.015); plt.close(fig)

def main():
    print('='*96); print('Nairobi v2 — Step 19b Final Coarse Semantic Grouping'); print('='*96)
    print('New SegFormer inference: NO'); print('Grad-CAM quantification: NO'); print('Fine ADE20K labels -> fixed coarse semantic groups\n')
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    for d in [MASK_DIR,COLOR_DIR,OVERLAY_DIR]: d.mkdir(parents=True,exist_ok=True)
    id2label=load_id2label(); rows=read_csv(FINE_CASE_CSV)
    if len(rows)!=16: raise ValueError(f'Expected 16 Step-19a cases, found {len(rows)}')
    cases=[]; fractions=[]; observed={}
    for i,r in enumerate(rows,1):
        fine_path=resolve(r['mask_npy']); img_path=resolve(r['model_input_png'])
        fine=np.squeeze(np.asarray(np.load(fine_path))).astype(np.int64)
        img=np.asarray(Image.open(img_path).convert('RGB'))
        if fine.shape!=(128,500) or img.shape[:2]!=fine.shape: raise ValueError(f'Grid mismatch: {fine.shape}, {img.shape[:2]}')
        coarse=np.full(fine.shape,COARSE_GROUPS.index('other_uncertain'),dtype=np.uint8)
        for cid in np.unique(fine):
            cid=int(cid); label=id2label.get(cid,f'class_{cid}'); g=group_for(label); coarse[fine==cid]=COARSE_GROUPS.index(g); observed[cid]={'label':label,'coarse_group':g}

        unique_coarse = set(int(x) for x in np.unique(coarse))
        valid_ids = set(range(len(COARSE_GROUPS)))
        if not unique_coarse.issubset(valid_ids):
            raise ValueError(f'Unexpected coarse semantic IDs: {sorted(unique_coarse - valid_ids)}')
        s=stem(r); mp=MASK_DIR/f'{s}__coarse_mask.npy'; cp=COLOR_DIR/f'{s}__coarse_segmentation.png'; op=OVERLAY_DIR/f'{s}__coarse_overlay.png'
        np.save(mp,coarse); seg=render(coarse); Image.fromarray(seg).save(cp); Image.fromarray(overlay(img,seg)).save(op)
        dfrac={}
        for gi,g in enumerate(COARSE_GROUPS):
            cnt=int(np.sum(coarse==gi)); frac=cnt/float(coarse.size); dfrac[g]=frac
            fractions.append({'selection':r['selection'],'subtype':r['subtype'],'k_stratum':r['k_stratum'],'covered_order':int(float(r['covered_order'])),'target_embedding_row':int(float(r['target_embedding_row'])),'coarse_group':g,'pixel_count':cnt,'area_fraction':frac})

        if not np.isclose(sum(dfrac.values()), 1.0, atol=1e-10):
            raise ValueError(
                f"Coarse area fractions do not sum to 1 for case {r['selection']}/"
                f"{r['subtype']}/{r['k_stratum']}: {sum(dfrac.values())}"
            )
        cases.append({'selection':r['selection'],'subtype':r['subtype'],'k_stratum':r['k_stratum'],'covered_order':int(float(r['covered_order'])),'target_embedding_row':int(float(r['target_embedding_row'])),'model_input_png':str(img_path),'fine_mask_npy':str(fine_path),'coarse_mask_npy':str(mp),'coarse_segmentation_png':str(cp),'coarse_overlay_png':str(op),'grid_width':500,'grid_height':128,'grid_alignment':'PASSED',**{f'area_{g}':dfrac[g] for g in COARSE_GROUPS}})
        print(f"[{i:02d}/16] {r['selection']} | {r['subtype']} | {r['k_stratum']}")
    write_csv(CASE_CSV,cases); write_csv(FRACTION_CSV,fractions)
    mapping={'status':'FROZEN','coarse_groups':COARSE_GROUPS,'mapping_rule':{g:sorted(v) for g,v in LABEL_SETS.items()},'fallback':'all unmapped ADE20K labels -> other_uncertain','manual_relabeling':False,'dynamic_objects_note':'Retained but not treated as core built-environment evidence.','other_uncertain_note':'Implausible/rare labels such as table remain other_uncertain.','observed_ade20k_labels':{str(k):v for k,v in sorted(observed.items())}}
    MAPPING_JSON.write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    build_qc('median',cases); build_qc('high_impact',cases)
    AUDIT_JSON.write_text(json.dumps({'status':'PASSED','step':'19b','n_cases':16,'new_segformer_inference':False,'gradcam_quantification':False,'manual_relabeling':False,'all_grid_alignment_passed':True},indent=2),encoding='utf-8')
    NOTES_TXT.write_text('Step 19b: fixed coarse semantic grouping only; no Grad-CAM quantification.\n',encoding='utf-8')
    print('\nSTATUS: PASSED'); print(f'Output: {OUT_DIR}')

if __name__=='__main__': main()
