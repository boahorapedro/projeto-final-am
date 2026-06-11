import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xrfm import xRFM

class XRFMClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, iters=3, bandwidth=10.0, exponent=1.0, reg=1e-3, random_state=42):
        self.iters = int(iters)
        self.bandwidth = float(bandwidth)
        self.exponent = float(exponent)
        self.reg = float(reg)
        self.random_state = random_state
        self.model = None
        self.le_ = LabelEncoder()
        self.ohe_ = OneHotEncoder(sparse_output=False)
        self.classes_ = None

    def _preprocess_X(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            return X.fillna(0).values.astype(np.float32)
        return np.nan_to_num(X).astype(np.float32)

    def _to_numpy(self, data):
        """Converte de forma segura o retorno do xRFM para NumPy array, tratando tensores ou ndarrays."""
        if hasattr(data, 'cpu'):
            return data.cpu().numpy()
        if hasattr(data, 'detach'):
            return data.detach().numpy()
        return np.asarray(data)

    def fit(self, X, y):
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        # Transforma o alvo para o formato multi-coluna binário aceito pela matemática interna
        y_ohe = self.ohe_.fit_transform(y_encoded.reshape(-1, 1)).astype(np.float32)
        X_num = self._preprocess_X(X)

        X_train, X_val, y_train_ohe, y_val_ohe = train_test_split(
            X_num, y_ohe, test_size=0.2, random_state=self.random_state, stratify=y_encoded
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train_t = torch.tensor(X_train, device=device)
        y_train_t = torch.tensor(y_train_ohe, device=device)
        X_val_t = torch.tensor(X_val, device=device)
        y_val_t = torch.tensor(y_val_ohe, device=device)

        rfm_params = {
            'model': {
                'kernel': 'l1_kermac',
                'bandwidth': self.bandwidth,
                'exponent': self.exponent,
                'diag': False,
                'bandwidth_mode': 'constant',
            },
            'fit': {
                'reg': self.reg,
                'iters': self.iters,
                'verbose': False,
                'early_stop_rfm': True,
            }
        }

        self.model = xRFM(
            rfm_params=rfm_params,
            device=device,
            tuning_metric='auc' 
        )

        self.model.fit(X_train_t, y_train_t, X_val_t, y_val_t)
        return self

    def predict_proba(self, X):
        X_num = self._preprocess_X(X)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X_num, device=device)
        
        # Coleta os valores brutos da predição usando a função auxiliar segura
        try:
            raw_data = self.model.predict_proba(X_t)
        except AttributeError:
            raw_data = self.model.predict(X_t)
            
        probas = self._to_numpy(raw_data)
        
        # Filtros e ajustes para estabilidade das métricas no Optuna
        probas = np.nan_to_num(probas, nan=0.0, posinf=1.0, neginf=0.0)
        probas = np.clip(probas, 1e-7, 1.0)
        
        row_sums = probas.sum(axis=1, keepdims=True)
        probas = probas / row_sums

        if len(self.classes_) == 2 and probas.shape[1] == 1:
            probas_2d = np.zeros((probas.shape[0], 2))
            probas_2d[:, 1] = probas[:, 0]
            probas_2d[:, 0] = 1.0 - probas[:, 0]
            return probas_2d
            
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        preds_num = np.argmax(probas, axis=1)
        return self.le_.inverse_transform(preds_num)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

def build_group_model(seed: int = 42, params: dict = None):
    if params is None:
        params = {}
    params['random_state'] = seed
    return XRFMClassifierWrapper(**params)