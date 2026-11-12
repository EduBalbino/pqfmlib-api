"""Check the SIA notebook's in-memory classical search against its pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def test_classical_search_reuses_preprocessing_without_changing_scores():
    path = Path(__file__).resolve().parents[2] / 'experiments/sia/QIMED_MoE_Demanda_Pressao_pqfmlib_api.ipynb'
    notebook = json.loads(path.read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'def tune_classic(' in ''.join(cell['source']))
    namespace = dict(ColumnTransformer=ColumnTransformer, Pipeline=Pipeline,
                     SimpleImputer=SimpleImputer, StandardScaler=StandardScaler,
                     OneHotEncoder=OneHotEncoder, XGBClassifier=XGBClassifier,
                     ParameterGrid=ParameterGrid, average_precision_score=average_precision_score,
                     np=np, pd=pd, time=__import__('time'), N_JOBS=1, SEED=42,
                     TARGETS={'classificacao': 'target'}, timed_function=lambda _: lambda fn: fn)
    exec(compile(source, '<sia-classical-cell>', 'exec'), namespace)
    namespace['make_estimator'] = lambda task, params: XGBClassifier(
        n_estimators=12, tree_method='hist', n_jobs=1, random_state=42, **params)
    data = pd.DataFrame({'number': [0., 1., np.nan, 3., 4., 5., 6., 7., 8., 9., 10., 11.],
                         'region': ['a', 'b'] * 6, 'target': [0, 1, 0, 1] * 3})
    protocol = {'features_numericas': ['number'], 'features_categoricas': ['region']}
    fold = {'inner_train': np.arange(8), 'inner_validation': np.arange(8, 12)}
    grid = {'max_depth': [1, 2], 'learning_rate': [.1, .2]}
    features = ['number', 'region']
    expected = []
    for params in ParameterGrid(grid):
        model = namespace['make_pipeline']('classificacao', params, protocol)
        model.fit(data.iloc[fold['inner_train']][features], data.iloc[fold['inner_train']]['target'])
        expected.append(average_precision_score(data.iloc[fold['inner_validation']]['target'],
                        model.predict_proba(data.iloc[fold['inner_validation']][features])[:, 1]))
    make_preprocessor = namespace['make_preprocessor']
    calls = []
    def counted_preprocessor(protocol):
        calls.append(1)
        return make_preprocessor(protocol)
    namespace['make_preprocessor'] = counted_preprocessor
    best, table = namespace['tune_classic'](data, fold, protocol, 'classificacao', grid)
    np.testing.assert_array_equal(table['score'].to_numpy(), expected)
    assert best == list(ParameterGrid(grid))[int(np.argmax(expected))]
    assert len(calls) == 1
