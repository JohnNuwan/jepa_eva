"""E.V.A — Evolving Virtual Asset.

Pipeline souverain de trading évolutif :
- Bloc A : ingestion, normalisation, encodeur JEPA PyTorch (GPU 0),
- Bloc B : pont DLPack, planificateur TD-MPC2, arène génétique JAX (GPU 1),
- Bloc C : assainisseur d'actions et disjoncteur drawdown.
"""

from .arene_genetique import JaxGeneticArena, ResultatEvaluation
from .assainisseur_actions import ActionSanitizer, DecisionSanitisee, LimitesMoneyManagement
from .disjoncteur_drawdown import DrawdownDisconnecter, PositionOuverte, RapportDisjonction
from .encodeur_jepa import MomentumTarget, TimeJEPAEncoder
from .normalisation import DynamicNormalizer, RunningLayerNorm
from .pont_jax import JAXTransitionBridge, pont_defaut
from .planificateur_tdmpc2 import TDMPC2Planner, initialiser_world_model

__all__ = [
    "ActionSanitizer",
    "DecisionSanitisee",
    "DrawdownDisconnecter",
    "DynamicNormalizer",
    "JaxGeneticArena",
    "JAXTransitionBridge",
    "LimitesMoneyManagement",
    "MomentumTarget",
    "PositionOuverte",
    "RapportDisjonction",
    "ResultatEvaluation",
    "RunningLayerNorm",
    "TDMPC2Planner",
    "TimeJEPAEncoder",
    "initialiser_world_model",
    "pont_defaut",
]
