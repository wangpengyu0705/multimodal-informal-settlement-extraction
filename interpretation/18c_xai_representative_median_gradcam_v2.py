#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from pathlib import Path
import csv, json, math

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT = Path.home() / 'projects' / 'nairobi_v2'
K = 7
BASE_DIM = 2048
FEATURE_DIM = 256
N_COVERED = 5744
FUSION_THRESHOLD = 0.59
TARGET_HEIGHT, TARGET_WIDTH = 128, 500

# Independent panorama-completeness QC.
NEAR_BLACK_MAX = 12
MAX_BORDER_CONNECTED_BLACK_FRACTION = 0.05

SUBTYPES = [
    's2_fp_to_correct',
    's2_fn_to_correct',
    's2_correct_to_fusion_fp',
    's2_correct_to_fusion_fn',
]
SUBTYPE_LABELS = {
    's2_fp_to_correct': 'S2 FP → Fusion correct',
    's2_fn_to_correct': 'S2 FN → Fusion correct',
    's2_correct_to_fusion_fp': 'S2 correct → Fusion FP',
    's2_correct_to_fusion_fn': 'S2 correct → Fusion FN',
}
EXPECTED_DIRECTION = {
    's2_fp_to_correct': -1,
    's2_fn_to_correct': +1,
    's2_correct_to_fusion_fp': +1,
    's2_correct_to_fusion_fn': -1,
}

FREEZE_MANIFEST = PROJECT / 'results/final_protocol_v2/seed_42/final_protocol_manifest_v2.json'
FINAL_TEST_AUDIT = PROJECT / 'results/final_test_v2/seed_42/final_test_audit_v2.json'
STEP16_AUDIT = PROJECT / 'results/xai_modality_v2/seed_42/xai_modality_audit_v2.json'
STEP17_AUDIT = PROJECT / 'results/xai_loo_v2/seed_42/loo_audit_v2.json'
STEP16_CELLWISE = PROJECT / 'results/xai_modality_v2/seed_42/xai_modality_delta_cellwise_v2.csv'
STEP17_CELLWISE = PROJECT / 'results/xai_loo_v2/seed_42/loo_cellwise_summary_v2.csv'
COVERED_CELL_INDEX = PROJECT / 'results/final_test_v2/seed_42/covered_test_cell_index_v2.csv'

X_TEST_S2 = PROJECT / 'scratch/features/s2_inputs/X_test_s2_v2.npy'
S2_ENCODER = PROJECT / 'models/s2_only_v2/seed_42/s2_encoder_best_v2.keras'
GSV_ENCODER = PROJECT / 'models/gsv_mean_pool_v2/seed_42/gsv_encoder_mean_pool_best_v2.keras'
FUSION_MODEL = PROJECT / 'models/fusion_mean_pool_v2/seed_42/fusion_mean_pool_best_v2.keras'
FUSION_SCALER = PROJECT / 'results/fusion_mean_pool_v2/seed_42/fusion_mean_pool_standardization_train_only_v2.npz'
GSV_EMBEDDINGS = PROJECT / 'scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embeddings_v2.npy'
GSV_EMBED_INDEX = PROJECT / 'scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embedding_index_v2.csv'
GSV_ROOT = PROJECT / 'scratch/data/gsv/images'

OUT_DIR = PROJECT / 'results/xai_gradcam_v2/seed_42/representative_median_v2'
CASE_DIR = OUT_DIR / 'cases'
SELECTION_CSV = OUT_DIR / 'representative_median_selection_v2.csv'
QC_CSV = OUT_DIR / 'panorama_completeness_qc_v2.csv'
METRICS_CSV = OUT_DIR / 'representative_median_metrics_v2.csv'
PANEL_PNG = OUT_DIR / 'gradcam_representative_median_panel_v2.png'
AUDIT_JSON = OUT_DIR / 'representative_median_audit_v2.json'
AUDIT_TXT = OUT_DIR / 'representative_median_audit_v2.txt'

@tf.keras.utils.register_keras_serializable(package='NairobiV2')
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        z, slot_mask = inputs
        mask = tf.cast(slot_mask, z.dtype)[..., None]
        return tf.math.divide_no_nan(tf.reduce_sum(z * mask, axis=1), tf.reduce_sum(mask, axis=1))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows: {path}')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def i(row, key): return int(float(str(row[key]).strip()))
def f(row, key): return float(str(row[key]).strip())

def ensure_clean_start():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    protected = [SELECTION_CSV, QC_CSV, METRICS_CSV, PANEL_PNG, AUDIT_JSON, AUDIT_TXT]
    existing = [p for p in protected if p.exists()]
    if existing or any(CASE_DIR.iterdir()):
        raise FileExistsError('Step18b outputs already exist; refusing overwrite')

def verify_state():
    if read_json(FREEZE_MANIFEST).get('status') != 'FROZEN': raise ValueError('Protocol not frozen')
    if read_json(FINAL_TEST_AUDIT).get('status') != 'PASSED': raise ValueError('Final test not passed')
    if read_json(STEP16_AUDIT).get('status') != 'PASSED': raise ValueError('Step16 not passed')
    if read_json(STEP17_AUDIT).get('status') != 'PASSED': raise ValueError('Step17 not passed')

def load_index():
    rows = read_csv(COVERED_CELL_INDEX)
    if len(rows) != N_COVERED: raise ValueError('Covered index count mismatch')
    out = {}
    for r in rows:
        order = i(r, 'covered_order')
        slots = [i(r, f'slot_{j}_embedding_row') for j in range(1, K+1)]
        out[order] = {
            'patch_order': i(r,'patch_order'), 'cell_row': i(r,'cell_row'), 'cell_col': i(r,'cell_col'),
            'global_cell_row': i(r,'global_cell_row'), 'global_cell_col': i(r,'global_cell_col'),
            'label': i(r,'label'), 'slot_count': i(r,'slot_count'), 'slot_rows': slots,
        }
    return out

def load_step16():
    out = {}
    for r in read_csv(STEP16_CELLWISE):
        o = i(r,'covered_order')
        out[o] = {
            'slot_count': i(r,'slot_count'), 'label': i(r,'label'),
            's2_probability': f(r,'s2_probability'), 's2_prediction': i(r,'s2_prediction'),
            'fusion_probability': f(r,'fusion_probability'), 'fusion_prediction': i(r,'fusion_prediction'),
            'delta_gsv_logit': f(r,'delta_gsv_logit'),
            'subtype': r['prediction_change_subtype'],
        }
    return out

def load_step17():
    out = {}
    for r in read_csv(STEP17_CELLWISE):
        o = i(r,'covered_order')
        out[o] = {
            'slot_count': i(r,'slot_count'), 'subtype': r['prediction_change_subtype'],
            'target_embedding_row': i(r,'most_influential_embedding_row'),
            'signed_loo': f(r,'most_influential_signed_impact_logit'),
            'abs_loo': f(r,'most_influential_abs_impact_logit'),
            'flip': i(r,'loo_any_prediction_flip'),
        }
    return out

def resolve_image(row):
    vals = [row.get('resolved_image_path',''), row.get('resolved_relative_path',''), row.get('image_path_relative_to_source','')]
    for s in vals:
        s = str(s).strip()
        if not s: continue
        p = Path(s)
        if p.is_file(): return p
        q = GSV_ROOT / s
        if q.is_file(): return q
    raise FileNotFoundError(f"Cannot resolve embedding_row={row.get('embedding_row')}")

def load_embedding_meta():
    out = {}
    for r in read_csv(GSV_EMBED_INDEX):
        er = i(r,'embedding_row'); p = resolve_image(r)
        out[er] = {'path': p, 'filename': p.name}
    return out

def decode_native(path):
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None,None,3])
    return img.numpy()

def border_connected_black_fraction(rgb):
    mask = np.all(rgb <= NEAR_BLACK_MAX, axis=2)
    h,w = mask.shape
    seen = np.zeros_like(mask, dtype=bool); q = deque()
    for x in range(w):
        for y in (0,h-1):
            if mask[y,x] and not seen[y,x]: seen[y,x]=True; q.append((y,x))
    for y in range(h):
        for x in (0,w-1):
            if mask[y,x] and not seen[y,x]: seen[y,x]=True; q.append((y,x))
    while q:
        y,x=q.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and mask[yy,xx] and not seen[yy,xx]:
                seen[yy,xx]=True; q.append((yy,xx))
    return float(seen.mean())

def build_pools(index, s16, s17):
    pools = {}
    for subtype in SUBTYPES:
        sign = EXPECTED_DIRECTION[subtype]
        k1=[]; km=[]
        for o,r in s16.items():
            if r['subtype']!=subtype or r['slot_count']!=1: continue
            d=r['delta_gsv_logit']
            if d==0 or int(math.copysign(1,d))!=sign: continue
            active=[x for x in index[o]['slot_rows'] if x>=0]
            if len(active)!=1: raise ValueError('K1 slot mismatch')
            k1.append({'subtype':subtype,'k_stratum':'K=1','covered_order':o,'target_embedding_row':active[0],
                       'signed_effect':d,'abs_effect':abs(d),'effect_name':'|DeltaGSV|'})
        for o,r in s17.items():
            if r['subtype']!=subtype or r['slot_count']<2: continue
            d=r['signed_loo']
            if d==0 or int(math.copysign(1,d))!=sign: continue
            km.append({'subtype':subtype,'k_stratum':'K>=2','covered_order':o,'target_embedding_row':r['target_embedding_row'],
                       'signed_effect':d,'abs_effect':r['abs_loo'],'effect_name':'max |LOO impact|'})
        pools[(subtype,'K=1')]=k1; pools[(subtype,'K>=2')]=km
    return pools

def select_median_original_logic_with_qc(pools, meta):
    """
    Preserve the ORIGINAL Step-18 median-strength representative-case logic,
    adding only an independent panorama-completeness QC.

    For each prediction-transition subtype and K stratum:

      1) Start from ALL direction-consistent candidates.
      2) Compute the ORIGINAL subgroup median absolute effect:
           K=1   -> median |DeltaGSV|
           K>=2  -> median max |LOO impact|
         This median is computed BEFORE panorama QC, matching the original
         representative-case selection definition.
      3) Apply independent panorama-completeness QC.
      4) From the QC-passing candidates, select the case whose absolute effect
         is closest to the ORIGINAL subgroup median.

    The panorama/Grad-CAM visual appearance is never used to choose the case.
    """
    qc_cache = {}
    qc_rows = []
    selected = []

    def qc(er):
        if er in qc_cache:
            return qc_cache[er]

        # load_embedding_meta() stores the resolved path under key "path".
        path = meta[er]["path"]
        frac = border_connected_black_fraction(decode_native(path))
        passed = frac <= MAX_BORDER_CONNECTED_BLACK_FRACTION

        rec = {
            "embedding_row": er,
            "filename": meta[er]["filename"],
            "image_path": str(path),
            "border_connected_near_black_fraction": frac,
            "threshold": MAX_BORDER_CONNECTED_BLACK_FRACTION,
            "qc_passed": int(passed),
        }
        qc_cache[er] = rec
        qc_rows.append(rec)
        return rec

    for subtype in SUBTYPES:
        for kstr in ("K=1", "K>=2"):
            candidates = list(pools[(subtype, kstr)])
            if not candidates:
                raise ValueError(f"Empty candidate pool: {subtype} {kstr}")

            # ORIGINAL target median: compute before panorama QC.
            original_median_abs_effect = float(
                np.median([float(c["abs_effect"]) for c in candidates])
            )

            eligible = []
            for c in candidates:
                q = qc(c["target_embedding_row"])
                if q["qc_passed"]:
                    cc = dict(c)
                    cc["panorama_black_fraction"] = q[
                        "border_connected_near_black_fraction"
                    ]
                    eligible.append(cc)

            if not eligible:
                raise ValueError(
                    f"No panorama-QC-passing candidate: {subtype} {kstr}"
                )

            # Preserve the original median target; only restrict the final
            # selectable pool by image-completeness QC.
            chosen = min(
                eligible,
                key=lambda c: (
                    abs(float(c["abs_effect"]) - original_median_abs_effect),
                    c["covered_order"],
                ),
            )
            chosen = dict(chosen)
            chosen.update({
                "candidate_pool_n_before_qc": len(candidates),
                "candidate_pool_n_after_qc": len(eligible),
                "original_subgroup_median_abs_effect_before_qc":
                    original_median_abs_effect,
                "distance_to_original_median": abs(
                    float(chosen["abs_effect"]) - original_median_abs_effect
                ),
                "selection_rule": (
                    "closest to the original pre-QC subgroup median "
                    "direction-consistent absolute effect among panoramas "
                    "passing independent completeness QC"
                ),
            })
            selected.append(chosen)

    if len(selected) != 8:
        raise ValueError(f"Expected 8 median-strength cases, got {len(selected)}")

    return selected, qc_rows


def load_scaler():
    x=np.load(FUSION_SCALER,allow_pickle=False); out={k:x[k].astype(np.float32) for k in x.files}; x.close(); return out

def build_s2_grid():
    x=np.load(X_TEST_S2,allow_pickle=False)
    enc=tf.keras.models.load_model(S2_ENCODER,compile=False)
    return enc.predict(x,batch_size=16,verbose=0).astype(np.float32)

def load_raw_exact(path):
    raw=tf.io.read_file(str(path)); img=tf.io.decode_image(raw,channels=3,expand_animations=False); img.set_shape([None,None,3])
    resized=tf.image.resize(img,[TARGET_HEIGHT,TARGET_WIDTH],method='bilinear',antialias=True); resized=tf.cast(resized,tf.float32)
    pre=tf.keras.applications.resnet50.preprocess_input(tf.identity(resized))
    rgb=tf.clip_by_value(resized,0,255).numpy().astype(np.uint8)
    return pre[None,...],rgb

def fusion_logit(model,s2z,gsvz):
    x=tf.concat([s2z,gsvz],axis=-1)
    x=model.get_layer('fusion_dense_128')(x,training=False)
    x=model.get_layer('fusion_dropout')(x,training=False)
    out=model.get_layer('informal_probability')
    logit=tf.linalg.matmul(x,out.kernel)
    if out.use_bias: logit=tf.nn.bias_add(logit,out.bias)
    return logit

def norm_cam(x):
    x=np.maximum(np.asarray(x,dtype=np.float32),0); m=float(x.max())
    return x/m if m>0 else x

def gradcam_case(case,index,s16,meta,embeddings,s2grid,scaler,gsv_encoder,fusion_model,resnet):
    o=case['covered_order']; idx=index[o]; er=case['target_embedding_row']; slot=idx['slot_rows'].index(er)
    direction=EXPECTED_DIRECTION[case['subtype']]
    pre,rgb=load_raw_exact(meta[er]['path'])
    s2f=s2grid[idx['patch_order'],idx['cell_row'],idx['cell_col']][None,:]
    s2z=tf.constant(((s2f-scaler['s2_mean'])/scaler['s2_std']).astype(np.float32))
    cell=np.zeros((1,K,BASE_DIM),np.float32); mask=np.zeros((1,K),np.float32)
    for j,row in enumerate(idx['slot_rows']):
        if row>=0: cell[0,j]=embeddings[row]; mask[0,j]=1
    cached=np.asarray(embeddings[er],np.float32)
    probe=resnet(pre,training=False); raw_probe=tf.reduce_mean(probe,axis=[1,2]).numpy()[0].astype(np.float32)
    cos=float(np.dot(raw_probe,cached)/(np.linalg.norm(raw_probe)*np.linalg.norm(cached)))
    with tf.GradientTape() as tape:
        conv=resnet(pre,training=False); tape.watch(conv); raw_emb=tf.reduce_mean(conv,axis=[1,2])
        cell_t=tf.tensor_scatter_nd_update(tf.constant(cell),tf.constant([[0,slot]],tf.int32),raw_emb)
        rep=gsv_encoder({'gsv_embeddings_k7':cell_t,'gsv_slot_mask':tf.constant(mask)},training=False)
        gsvz=(rep-tf.constant(scaler['gsv_mean'][None,:]))/tf.constant(scaler['gsv_std'][None,:])
        ilogit=fusion_logit(fusion_model,s2z,gsvz); score=ilogit[0,0] if direction>0 else -ilogit[0,0]
    grads=tape.gradient(score,conv); weights=tf.reduce_mean(grads,axis=[1,2])[0]
    cam=tf.reduce_sum(conv[0]*weights[None,None,:],axis=-1); cam=norm_cam(cam.numpy())
    cam=tf.image.resize(cam[...,None],[TARGET_HEIGHT,TARGET_WIDTH],method='bilinear',antialias=True).numpy()[...,0]; cam=norm_cam(cam)
    raw_prob=float(tf.sigmoid(ilogit)[0,0].numpy()); frozen=s16[o]['fusion_probability']
    return {'rgb':rgb,'heatmap':cam,'embedding_cosine':cos,'raw_prob':raw_prob,'frozen_prob':frozen,
            'prob_diff':abs(raw_prob-frozen),'target_direction':'informal' if direction>0 else 'non-informal',
            'delta_gsv_logit':s16[o]['delta_gsv_logit'],'label':s16[o]['label'],'s2_probability':s16[o]['s2_probability']}

def save_case(case,res):
    name=f"{case['subtype']}_{case['k_stratum'].replace('>=','ge').replace('=','eq')}_order{case['covered_order']}"
    d=CASE_DIR/name; d.mkdir(parents=True,exist_ok=False)
    orig=d/'model_input_rgb_v2.png'; overlay=d/'gradcam_directional_overlay_v2.png'; heat=d/'gradcam_directional_heatmap_v2.npy'
    plt.imsave(orig,res['rgb'])
    fig,ax=plt.subplots(figsize=(12,3)); ax.imshow(res['rgb']); ax.imshow(res['heatmap'],cmap='viridis',alpha=.45,vmin=0,vmax=1); ax.axis('off'); fig.tight_layout(pad=0); fig.savefig(overlay,dpi=220,bbox_inches='tight',pad_inches=0); plt.close(fig)
    np.save(heat,res['heatmap'].astype(np.float32),allow_pickle=False)
    return str(orig),str(overlay),str(heat)

def make_panel(selected,results):
    fig,axes=plt.subplots(4,4,figsize=(18,10),squeeze=False)
    for ri,subtype in enumerate(SUBTYPES):
        for pi,kstr in enumerate(('K=1','K>=2')):
            case=next(c for c in selected if c['subtype']==subtype and c['k_stratum']==kstr); res=results[(subtype,kstr)]
            a0=axes[ri,pi*2]; a1=axes[ri,pi*2+1]
            a0.imshow(res['rgb']); a0.axis('off'); a0.set_title(f'{kstr} — original',fontsize=9)
            a1.imshow(res['rgb']); a1.imshow(res['heatmap'],cmap='viridis',alpha=.45,vmin=0,vmax=1); a1.axis('off')
            eff=f"ΔGSV={res['delta_gsv_logit']:.2f}" if kstr=='K=1' else f"ΔGSV={res['delta_gsv_logit']:.2f}, LOO={case['signed_effect']:.2f}"
            a1.set_title(f"{res['target_direction']} support\n{eff}",fontsize=9)
        axes[ri,0].set_ylabel(SUBTYPE_LABELS[subtype],fontsize=9,rotation=90)
    sm=ScalarMappable(norm=Normalize(0,1),cmap='viridis'); sm.set_array([])
    cb=fig.colorbar(sm,ax=axes.ravel().tolist(),fraction=.018,pad=.012); cb.set_label('Directional Grad-CAM response\n(within-image normalized)',fontsize=9); cb.set_ticks([0,.5,1]); cb.set_ticklabels(['Low','Medium','High'])
    fig.suptitle('Representative median-strength Fusion-targeted GSV Grad-CAM cases',fontsize=12)
    fig.subplots_adjust(left=.08,right=.94,top=.92,bottom=.04,wspace=.08,hspace=.34)
    fig.savefig(PANEL_PNG,dpi=240,bbox_inches='tight'); plt.close(fig)

def main():
    ensure_clean_start(); verify_state()
    print('='*92); print('Nairobi v2 — Step 18c: representative median-strength Grad-CAM'); print('='*92)
    print('Training: NO | Threshold tuning: NO | Model modification: NO')
    print(f'Panorama QC: border-connected RGB<={NEAR_BLACK_MAX}; exclude > {MAX_BORDER_CONNECTED_BLACK_FRACTION:.0%}')
    index=load_index(); s16=load_step16(); s17=load_step17(); meta=load_embedding_meta()
    pools=build_pools(index,s16,s17); selected,qc_rows=select_median_original_logic_with_qc(pools,meta)
    write_csv(SELECTION_CSV,selected); write_csv(QC_CSV,qc_rows)
    print('[1] Original-median representative selection + independent panorama QC: PASSED')
    embeddings=np.load(GSV_EMBEDDINGS,allow_pickle=False,mmap_mode='r'); s2grid=build_s2_grid(); scaler=load_scaler()
    gsv_encoder=tf.keras.models.load_model(GSV_ENCODER,custom_objects={'MaskedMeanPooling':MaskedMeanPooling},compile=False)
    fusion_model=tf.keras.models.load_model(FUSION_MODEL,compile=False)
    resnet=tf.keras.applications.ResNet50(weights='imagenet',include_top=False,pooling=None,input_shape=(TARGET_HEIGHT,TARGET_WIDTH,3)); resnet.trainable=False
    results={}; metrics=[]
    for case in selected:
        res=gradcam_case(case,index,s16,meta,embeddings,s2grid,scaler,gsv_encoder,fusion_model,resnet)
        if res['embedding_cosine']<.999: raise ValueError('Embedding reconstruction mismatch')
        if res['prob_diff']>.01: raise ValueError('Fusion reconstruction mismatch')
        orig,overlay,heat=save_case(case,res); results[(case['subtype'],case['k_stratum'])]=res
        metrics.append({**case,'subtype_label':SUBTYPE_LABELS[case['subtype']],'label':res['label'],'s2_probability':res['s2_probability'],
            'fusion_probability_frozen':res['frozen_prob'],'delta_gsv_logit':res['delta_gsv_logit'],'target_direction':res['target_direction'],
            'embedding_cosine_raw_vs_cached':res['embedding_cosine'],'raw_vs_frozen_probability_abs_diff':res['prob_diff'],
            'image_filename':meta[case['target_embedding_row']]['filename'],'image_path':str(meta[case['target_embedding_row']]['path']),
            'original_png':orig,'overlay_png':overlay,'heatmap_npy':heat})
        print(f"  {case['subtype']} {case['k_stratum']}: order={case['covered_order']} effect={case['signed_effect']:.4f} QCblack={case['panorama_black_fraction']:.4%}")
    write_csv(METRICS_CSV,metrics); make_panel(selected,results)
    audit={'status':'PASSED','training_performed':False,'threshold_tuning_performed':False,'model_modified':False,
        'selection_name':'representative median-strength cases','cases':8,
        'k1_rule':'closest to original pre-QC subgroup median direction-consistent |DeltaGSV| among panorama-QC-passing candidates',
        'k_ge_2_rule':'closest to original pre-QC subgroup median direction-consistent max |LOO impact| among panorama-QC-passing candidates',
        'panorama_qc':{'near_black_definition':f'all RGB channels <= {NEAR_BLACK_MAX}','measure':'border-connected near-black pixel fraction','exclusion_threshold':MAX_BORDER_CONNECTED_BLACK_FRACTION,'uses_prediction_or_gradcam':False},
        'figure_note':'Heatmaps are normalized independently within each image; color intensity is for within-image spatial interpretation, not cross-image quantitative comparison.',
        'outputs':{'selection_csv':str(SELECTION_CSV),'qc_csv':str(QC_CSV),'metrics_csv':str(METRICS_CSV),'panel_png':str(PANEL_PNG),'case_dir':str(CASE_DIR)}}
    AUDIT_JSON.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    lines=['='*92,'Nairobi v2 — Representative median-strength Grad-CAM audit','='*92,'STATUS: PASSED','Training: NO','Threshold tuning: NO','Model modification: NO','',
        'Selection: 4 transition subtypes x {K=1,K>=2} = 8 cases','K=1: closest to original pre-QC subgroup median |DeltaGSV| after panorama QC','K>=2: closest to original pre-QC subgroup median max |LOO impact| after panorama QC','Grad-CAM appearance used for selection: NO','',
        f'Panorama QC: border-connected near-black fraction; exclude > {MAX_BORDER_CONNECTED_BLACK_FRACTION:.2%}','',
        'Figure: explicit Low→High colorbar; heatmaps normalized within each image.','',f'Panel: {PANEL_PNG}',f'Selection: {SELECTION_CSV}',f'QC: {QC_CSV}',f'Metrics: {METRICS_CSV}']
    AUDIT_TXT.write_text('\n'.join(lines),encoding='utf-8'); print('\n'+'\n'.join(lines))

if __name__=='__main__': main()
