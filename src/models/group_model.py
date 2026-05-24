"""Placeholder do modelo principal do grupo.

Cada grupo deve substituir `build_group_model` pelo wrapper sklearn-compatível
do modelo atribuído. A lista de modelos atribuíveis está descrita no README.

Padrão esperado: a função retorna um estimador com .fit(X, y) e .predict_proba(X).
Se o modelo não tem API sklearn, envolva em sklearn.base.BaseEstimator.
"""

from __future__ import annotations


def build_group_model(seed: int = 42):
    """Substitua o corpo desta função pelo modelo do seu grupo.

    Exemplos (descomente o que se aplicar ao seu grupo):

    # 1) TabPFN-2.5
    # from tabpfn import TabPFNClassifier
    # return TabPFNClassifier(random_state=seed, device='auto')

    # 2) TabICL v2
    # from tabicl import TabICLClassifier
    # return TabICLClassifier(random_state=seed)

    # 3) TabM (via pytabkit)
    # from pytabkit import TabM_Classifier
    # return TabM_Classifier(random_state=seed, n_jobs=-1)

    # 4) ModernNCA (via TALENT toolkit)
    # consultar https://github.com/LAMDA-Tabular/TALENT para wrapper

    # 5) RealMLP
    # from pytabkit import RealMLP_TD_S_Classifier
    # return RealMLP_TD_S_Classifier(random_state=seed)

    # 6) xRFM
    # from xrfm import xRFMClassifier
    # return xRFMClassifier(random_state=seed)

    # 7) Mambular
    # from deeptab import MambularClassifier
    # return MambularClassifier(random_state=seed)

    # 8) FT-Transformer
    # from pytabkit import FTT_TD_Classifier
    # return FTT_TD_Classifier(random_state=seed)

    # 9) EBM
    # from interpret.glassbox import ExplainableBoostingClassifier
    # return ExplainableBoostingClassifier(random_state=seed)

    # 10) Trompt (via pytorch-frame)
    # consultar https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py

    raise NotImplementedError(
        "Implemente o modelo do seu grupo em src/models/group_model.py"
    )
