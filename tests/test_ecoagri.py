import numpy as np
import pytest
from ecoagri import field_features, normalized_difference, predict, train_baseline

def test_clouds_do_not_become_zero_ndvi():
    r=field_features([.1,.1],[.6,.6],[.2,.2],[9,9],[-10,-11],[-15,-16])
    assert r['status']=='insufficient_optical_data' and r['features'] is None

def test_indices_and_missing():
    assert normalized_difference(.6,.2)==pytest.approx(.5)
    assert np.isnan(normalized_difference(0,0))
    assert np.isnan(normalized_difference(-1,2))
    r=field_features([.2],[.6],[.3],[4],[-10],[-15])
    assert r['features'][0]==pytest.approx(.5)

def test_grid_validation():
    with pytest.raises(ValueError):field_features([1],[1,2],[1],[4],[1],[1])

def rows():
    return [dict(field_id=f'{split}-{i}',year=year,crop='a' if i%2 else 'b',ndvi=.2+.5*(i%2),ndmi=.1+.3*(i%2),vv_db=-10+2*(i%2),vh_db=-16+3*(i%2))
            for split,year in [('train',2024),('test',2025)] for i in range(8)]

def test_field_purge_and_roundtrip():
    r=rows();r.append(r[8]|{'year':2023})
    model,metrics=train_baseline(r,2025)
    assert set(model['training_fields']).isdisjoint(model['test_fields'])
    assert metrics['n_train']==8 and metrics['n_test']==8
    assert predict(model,[.7,.4,-8,-13])['crop']=='a'
    assert predict(model,[10,.4,-8,-13])['crop'] is None

def test_no_valid_split():
    with pytest.raises(ValueError):train_baseline(rows(),2030)
    with pytest.raises(ValueError):train_baseline(rows(),2024)
