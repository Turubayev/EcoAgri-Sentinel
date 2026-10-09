"""Cloud-aware field features and a field-disjoint classification baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FEATURES = ['ndvi', 'ndmi', 'vv_db', 'vh_db']

def normalized_difference(a, b, valid=None):
    a,b = np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    ok = np.isfinite(a)&np.isfinite(b)&(a>=0)&(b>=0)&(np.abs(a+b)>1e-8)
    if valid is not None: ok &= np.broadcast_to(np.asarray(valid,bool), a.shape)
    return np.divide(a-b,a+b,out=np.full(a.shape,np.nan),where=ok)

def field_features(red, nir, swir, scl, vv_db, vh_db, min_clear_fraction=.5):
    """Sentinel-2 reflectance (same scale), SCL, co-registered S1 dB pixels.

    Clear SCL classes: vegetation(4), bare soil(5), water(6). Reject no-data,
    saturated pixels, shadows, clouds, cirrus, snow and unclassified pixels.
    """
    arrays = [np.asarray(a) for a in (red,nir,swir,scl,vv_db,vh_db)]
    if not arrays[0].size or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('All bands must share a nonempty registered grid')
    if not 0 <= min_clear_fraction <= 1: raise ValueError('Invalid clear fraction')
    red,nir,swir,scl,vv,vh = arrays
    clear = np.isin(scl,[4,5,6]) & np.isfinite(red)&np.isfinite(nir)&np.isfinite(swir)&(red>=0)&(nir>=0)&(swir>=0)
    fraction = float(clear.mean())
    ndvi,ndmi = normalized_difference(nir,red,clear),normalized_difference(nir,swir,clear)
    if fraction < min_clear_fraction or not np.isfinite(ndvi).any() or not np.isfinite(ndmi).any():
        return {'status':'insufficient_optical_data','clear_fraction':fraction,'features':None}
    if not np.isfinite(vv).any() or not np.isfinite(vh).any():
        return {'status':'insufficient_radar_data','clear_fraction':fraction,'features':None}
    return {'status':'ready','clear_fraction':fraction, 'features':[float(np.nanmedian(v)) for v in (ndvi,ndmi,vv,vh)]}

def train_baseline(rows, test_year, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
    if not isinstance(rows,list) or len(rows)<8: raise ValueError('Need at least 8 field observations')
    x = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([str(r['crop']) for r in rows])
    fields = np.array([str(r['field_id']) for r in rows]); years = np.array([int(r['year']) for r in rows])
    if not np.isfinite(x).all() or not all(fields): raise ValueError('Invalid feature or field ID')
    test = years >= test_year
    # Exclude *every* observation from test fields, even observations in earlier years.
    train = (years < test_year) & ~np.isin(fields, fields[test])
    if train.sum()<4 or test.sum()<2 or len(np.unique(y[train]))<2:
        raise ValueError('Need multiple crop classes and separate fields/years for test')
    if not set(y[test]) <= set(y[train]): raise ValueError('Test contains an unseen crop class')
    scale = StandardScaler().fit(x[train]); model = LogisticRegression(random_state=seed,max_iter=1000).fit(scale.transform(x[train]),y[train])
    pred = model.predict(scale.transform(x[test]))
    # Portable JSON weights; loading never executes a pickle.
    artifact = {'schema':1,'features':FEATURES,'classes':model.classes_.tolist(), 'mean':scale.mean_.tolist(),
                'scale':scale.scale_.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),
                'training_fields':sorted(set(fields[train])),'test_fields':sorted(set(fields[test])),
                'test_year':test_year,'data_kind':'synthetic' if all(r.get('data_kind')=='synthetic' for r in rows) else 'user_supplied_unverified'}
    metrics = {'macro_f1':float(f1_score(y[test],pred,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y[test],pred)),
               'confusion_matrix':confusion_matrix(y[test],pred,labels=model.classes_).tolist(), 'classes':model.classes_.tolist(),
               'n_train':int(train.sum()),'n_test':int(test.sum()),'field_overlap':0,'data_kind':artifact['data_kind'],'operational_validation':False}
    return artifact, metrics

def predict(artifact, features):
    if artifact.get('schema')!=1 or artifact.get('features')!=FEATURES: raise ValueError('Invalid model schema')
    x = np.asarray(features,float)
    mean,scale = np.asarray(artifact['mean']),np.asarray(artifact['scale'])
    coef,bias = np.asarray(artifact['coef']),np.asarray(artifact['intercept'])
    classes = artifact['classes']
    if x.shape!=(4,) or mean.shape!=(4,) or scale.shape!=(4,) or not np.isfinite(x).all() or not (scale>0).all():
        raise ValueError('Invalid input or scaling')
    if coef.ndim != 2 or coef.shape[1:] != (4,) or bias.shape!=(len(coef),) or len(classes)<2 or len(coef) != (1 if len(classes)==2 else len(classes)):
        raise ValueError('Invalid classifier weights')
    if not np.isfinite(np.concatenate([mean,scale,coef.ravel(),bias])).all(): raise ValueError('Nonfinite model')
    z=(x-mean)/scale
    logits=coef@z+bias
    if len(classes)==2 and len(logits)==1: logits=np.array([0.,logits[0]])
    probs=np.exp(logits-logits.max());probs/=probs.sum()
    # Conservative heuristic, not a calibrated OOD detector.
    out_of_range=bool((np.abs(z)>5).any())
    return {'crop':None if out_of_range else classes[int(probs.argmax())], 'status':'review_out_of_range' if out_of_range else 'prediction',
            'probabilities':dict(zip(classes,map(float,probs))),'review_required':True}

def main():
    p=argparse.ArgumentParser(description='Field-disjoint crop baseline')
    p.add_argument('rows',type=Path);p.add_argument('--test-year',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        model,metrics=train_baseline(json.loads(a.rows.read_text(encoding='utf-8')),a.test_year)
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
        (a.out/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(json.dumps(metrics,indent=2))
    except (ValueError,TypeError,KeyError,OSError) as e: p.error(str(e))

if __name__=='__main__':main()
