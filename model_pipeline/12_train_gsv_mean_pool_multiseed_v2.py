#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv, json, os, random, sys, time
import numpy as np
import tensorflow as tf

PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = int(os.environ.get("GSV_MEAN_SEED", "42"))
BATCH_SIZE = 128
MAX_EPOCHS = 100
LR = 1e-4
EARLY_STOP_PATIENCE = 12
LR_PATIENCE = 5
DROPOUT = 0.30
K = 7
BASE_DIM = 2048
REP_DIM = 256
EXPECTED_IMAGES = 34207
EXPECTED_TRAIN_CELLS = 15481
EXPECTED_VAL_CELLS = 4902
EXPECTED_TRAIN_NEG = 12920
EXPECTED_TRAIN_POS = 2561
EXPECTED_VAL_NEG = 4037
EXPECTED_VAL_POS = 865

EMBED_PATH = PROJECT / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embeddings_v2.npy"
CELL_INDEX_PATH = PROJECT / "results/gsv_only_v2/seed_42/gsv_train_val_cell_index_v2.csv"
MODEL_DIR = PROJECT / f"models/gsv_mean_pool_v2/seed_{SEED}"
RESULT_DIR = PROJECT / f"results/gsv_mean_pool_v2/seed_{SEED}"
CKPT_DIR = PROJECT / f"scratch/checkpoints/gsv_mean_pool_v2/seed_{SEED}"
BEST_WEIGHTS = CKPT_DIR / "best.weights.h5"
FULL_MODEL = MODEL_DIR / "gsv_only_mean_pool_best_v2.keras"
ENCODER_MODEL = MODEL_DIR / "gsv_encoder_mean_pool_best_v2.keras"
HISTORY_CSV = RESULT_DIR / "training_history_v2.csv"
THRESHOLD_CSV = RESULT_DIR / "val_threshold_scan_v2.csv"
VAL_PROBS = RESULT_DIR / "val_probabilities_v2.npy"
AUDIT_JSON = RESULT_DIR / "gsv_mean_pool_training_audit_v2.json"
AUDIT_TXT = RESULT_DIR / "gsv_mean_pool_training_audit_v2.txt"

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def to_int(v, name):
    v = str(v).strip()
    if not v: raise ValueError(f"Empty {name}")
    return int(float(v))

def ensure_clean_start():
    MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULT_DIR.mkdir(parents=True, exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [FULL_MODEL, ENCODER_MODEL, BEST_WEIGHTS, VAL_PROBS, AUDIT_JSON]
    existing = [p for p in protected if p.exists()]
    if existing: raise FileExistsError("Protected outputs already exist; refusing overwrite:\n" + "\n".join(map(str, existing)))

def load_embeddings():
    if not EMBED_PATH.is_file(): raise FileNotFoundError(EMBED_PATH)
    x = np.load(EMBED_PATH, allow_pickle=False)
    if x.shape != (EXPECTED_IMAGES, BASE_DIM): raise ValueError(f"Embedding shape {x.shape}")
    if x.dtype != np.float32: raise ValueError(f"Embedding dtype {x.dtype}")
    if not np.isfinite(x).all(): raise ValueError("Non-finite embeddings")
    return x

def read_cell_records():
    if not CELL_INDEX_PATH.is_file(): raise FileNotFoundError(CELL_INDEX_PATH)
    recs = {"train": [], "val": []}
    with CELL_INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = str(row["split"]).strip().lower()
            if split not in recs: continue
            rec = {
                "order": to_int(row["cell_order_within_split"], "cell_order_within_split"),
                "label": to_int(row["label"], "label"),
                "slot_count": to_int(row["slot_count"], "slot_count"),
                "slot_rows": [to_int(row[f"slot_{i+1}_embedding_row"], f"slot_{i+1}_embedding_row") for i in range(K)],
            }
            if rec["label"] not in {0,1}: raise ValueError(f"Invalid label {rec['label']}")
            recs[split].append(rec)
    for split, expected in [("train", EXPECTED_TRAIN_CELLS), ("val", EXPECTED_VAL_CELLS)]:
        recs[split].sort(key=lambda r:r["order"])
        if len(recs[split]) != expected: raise ValueError(f"{split} count {len(recs[split])} != {expected}")
        if [r["order"] for r in recs[split]] != list(range(expected)): raise ValueError(f"{split} order mismatch")
        for r in recs[split]:
            active = sum(int(v >= 0) for v in r["slot_rows"])
            if active != r["slot_count"] or not 1 <= active <= K: raise ValueError("slot_count mismatch")
    return recs

def records_to_arrays(records, embeddings):
    n = len(records)
    x = np.zeros((n,K,BASE_DIM), np.float32); m = np.zeros((n,K), np.float32); y = np.zeros((n,1), np.float32)
    for i,r in enumerate(records):
        y[i,0] = r["label"]
        for s,er in enumerate(r["slot_rows"]):
            if er < 0: continue
            if not 0 <= er < embeddings.shape[0]: raise ValueError(f"Invalid embedding row {er}")
            x[i,s] = embeddings[er]; m[i,s] = 1.0
    if not np.isfinite(x).all(): raise ValueError("Non-finite cell tensor")
    return x,m,y

@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        z, slot_mask = inputs
        mask = tf.cast(slot_mask, z.dtype)[...,None]
        return tf.math.divide_no_nan(tf.reduce_sum(z*mask, axis=1), tf.reduce_sum(mask, axis=1))

@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class PositiveWeightedBCE(tf.keras.losses.Loss):
    def __init__(self, positive_weight, name="positive_weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs); self.positive_weight=float(positive_weight)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred), message="PositiveWeightedBCE requires identical y_true/y_pred shapes")
        eps=tf.keras.backend.epsilon(); y_pred=tf.clip_by_value(y_pred, eps, 1.0-eps)
        loss=-(self.positive_weight*y_true*tf.math.log(y_pred)+(1.0-y_true)*tf.math.log(1.0-y_pred))
        return tf.reduce_mean(loss, axis=-1)
    def get_config(self):
        cfg=super().get_config(); cfg.update({"positive_weight":self.positive_weight}); return cfg

def build_models():
    emb_in=tf.keras.Input((K,BASE_DIM), name="gsv_embeddings_k7"); mask_in=tf.keras.Input((K,), name="gsv_slot_mask")
    z=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(REP_DIM,activation="relu",kernel_initializer="he_normal"), name="shared_projection_2048_to_256")(emb_in)
    z=tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(DROPOUT), name="shared_projection_dropout")(z)
    pooled=MaskedMeanPooling(name="masked_mean_pool")([z,mask_in])
    rep=tf.keras.layers.LayerNormalization(name="gsv_representation_256")(pooled)
    h=tf.keras.layers.Dropout(DROPOUT, name="gsv_head_dropout")(rep)
    prob=tf.keras.layers.Dense(1,activation="sigmoid",name="gsv_probability")(h)
    full=tf.keras.Model([emb_in,mask_in],prob,name="nairobi_v2_gsv_only_mean_pool")
    enc=tf.keras.Model([emb_in,mask_in],rep,name="nairobi_v2_gsv_mean_pool_encoder")
    return full,enc

def confusion(y,p,t):
    pred=p>=t; tp=int(np.sum((pred==1)&(y==1))); fp=int(np.sum((pred==1)&(y==0))); tn=int(np.sum((pred==0)&(y==0))); fn=int(np.sum((pred==0)&(y==1)))
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.; iou=tp/(tp+fp+fn) if tp+fp+fn else 0.; acc=(tp+tn)/len(y)
    nprec=tn/(tn+fn) if tn+fn else 0.; nrec=tn/(tn+fp) if tn+fp else 0.; nf1=2*nprec*nrec/(nprec+nrec) if nprec+nrec else 0.
    return {"threshold":float(t),"tp":tp,"fp":fp,"tn":tn,"fn":fn,"informal_precision":precision,"informal_recall":recall,"informal_f1":f1,"informal_iou":iou,"macro_f1":0.5*(f1+nf1),"accuracy":acc}

def threshold_scan(y,p):
    rows=[confusion(y,p,float(t)) for t in np.round(np.arange(0.05,0.9501,0.005),3)]
    best=max(rows,key=lambda r:(r["informal_f1"],r["macro_f1"],-abs(r["threshold"]-0.5),-r["threshold"]))
    return rows,best

def pr_auc(y,p):
    m=tf.keras.metrics.AUC(curve="PR"); m.update_state(y.astype(np.float32),p.astype(np.float32)); return float(m.result().numpy())

def main():
    seed_everything(SEED); ensure_clean_start()
    print("="*88); print("Nairobi v2 — final GSV mean-pooling multiseed training"); print("="*88)
    print("Python:",sys.executable); print("TensorFlow:",tf.__version__); print("Physical GPUs:",tf.config.list_physical_devices("GPU")); print("Seed:",SEED); print("TEST DATA LOADED/EVALUATED: NO")
    emb=load_embeddings(); recs=read_cell_records(); xtr,mtr,ytr=records_to_arrays(recs["train"],emb); xva,mva,yva=records_to_arrays(recs["val"],emb); del emb
    trn=int(np.sum(ytr[:,0]==0)); trp=int(np.sum(ytr[:,0]==1)); van=int(np.sum(yva[:,0]==0)); vap=int(np.sum(yva[:,0]==1))
    if (trn,trp)!=(EXPECTED_TRAIN_NEG,EXPECTED_TRAIN_POS): raise ValueError((trn,trp))
    if (van,vap)!=(EXPECTED_VAL_NEG,EXPECTED_VAL_POS): raise ValueError((van,vap))
    pw=trn/trp
    probe=PositiveWeightedBCE(pw).call(tf.constant([[0.],[1.]],tf.float32),tf.constant([[0.25],[0.75]],tf.float32))
    if tuple(probe.shape)!=(2,): raise ValueError(probe.shape)
    print("Input integrity: PASSED"); print("Train X/mask/y:",xtr.shape,mtr.shape,ytr.shape); print("Val X/mask/y:",xva.shape,mva.shape,yva.shape)
    print("Train class counts:",trn,trp); print("Val class counts:",van,vap); print("Positive-class weight:",pw); print("Custom weighted-BCE shape contract: PASSED"); print("GSV-only population: GSV-COVERED VALID CELLS ONLY")
    model,encoder=build_models(); model.compile(optimizer=tf.keras.optimizers.Adam(LR),loss=PositiveWeightedBCE(pw),metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy",threshold=0.5),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.AUC(name="pr_auc",curve="PR")]); model.summary()
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(BEST_WEIGHTS),monitor="val_pr_auc",mode="max",save_best_only=True,save_weights_only=True,verbose=1),tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc",mode="max",patience=EARLY_STOP_PATIENCE,restore_best_weights=True,verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc",mode="max",patience=LR_PATIENCE,factor=0.5,min_lr=1e-6,verbose=1)]
    t0=time.time(); hist=model.fit({"gsv_embeddings_k7":xtr,"gsv_slot_mask":mtr},ytr,validation_data=({"gsv_embeddings_k7":xva,"gsv_slot_mask":mva},yva),batch_size=BATCH_SIZE,epochs=MAX_EPOCHS,callbacks=callbacks,shuffle=True,verbose=2); elapsed=time.time()-t0
    model.load_weights(BEST_WEIGHTS)
    hrows=[]
    for i in range(len(hist.history["loss"])):
        row={"epoch":i+1}; row.update({k:float(v[i]) for k,v in hist.history.items()}); hrows.append(row)
    write_rows(HISTORY_CSV,hrows)
    probs=model.predict({"gsv_embeddings_k7":xva,"gsv_slot_mask":mva},batch_size=BATCH_SIZE,verbose=0).astype(np.float32)
    if probs.shape!=(EXPECTED_VAL_CELLS,1) or not np.isfinite(probs).all(): raise ValueError(f"Bad val probs {probs.shape}")
    np.save(VAL_PROBS,probs,allow_pickle=False); yv=yva[:,0].astype(np.int8); mean_pr=pr_auc(yv,probs[:,0])
    scans,best=threshold_scan(yv,probs[:,0]); write_rows(THRESHOLD_CSV,scans); m05=confusion(yv,probs[:,0],0.5)
    model.save(FULL_MODEL); encoder.save(ENCODER_MODEL); best_epoch=int(np.argmax(hist.history["val_pr_auc"])+1)
    payload={"status":"PASSED","experiment":"Final GSV mean-pooling multiseed training","population":"GSV-covered valid train/validation cells only","seed":SEED,"trainable_parameters":int(model.count_params()),"best_epoch":best_epoch,"validation_pr_auc":mean_pr,"selected_threshold":best,"test_data_loaded":False,"test_data_evaluated":False,"elapsed_seconds":elapsed}; atomic_json(AUDIT_JSON,payload)
    lines=["="*88,"Nairobi v2 — Final GSV Mean-Pooling Multiseed Training Audit","="*88,f"Seed: {SEED}","Population: GSV-COVERED VALID CELLS ONLY",f"Train cells: {EXPECTED_TRAIN_CELLS}",f"Validation cells: {EXPECTED_VAL_CELLS}",f"Trainable parameters: {model.count_params()}",f"Best epoch: {best_epoch}",f"Validation PR-AUC: {mean_pr}",f"F1 @ 0.5: {m05['informal_f1']}",f"Selected validation threshold: {best['threshold']}",f"Selected-threshold precision: {best['informal_precision']}",f"Selected-threshold recall: {best['informal_recall']}",f"Selected-threshold F1: {best['informal_f1']}",f"Selected-threshold IoU: {best['informal_iou']}",f"Selected-threshold macro-F1: {best['macro_f1']}","","TEST DATA LOADED: NO","TEST DATA EVALUATED: NO","","GSV MEAN-POOLING MULTISEED TRAINING: PASSED"]
    AUDIT_TXT.write_text("\n".join(lines),encoding="utf-8"); print(); print("\n".join(lines)); print(); print("Saved full model:",FULL_MODEL); print("Saved encoder:",ENCODER_MODEL); print("Saved validation probabilities:",VAL_PROBS)

if __name__ == "__main__": main()
