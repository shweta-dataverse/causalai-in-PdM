"""Actionable Root Cause Analysis framework for predictive maintenance.

Pipeline stages (one module each):

    preprocessing   Module 1  Data understanding & preprocessing
    features        Module 2  Feature engineering & health indicator
    discovery       Module 3  Causal discovery (PCMCI+)
    validation      Module 4  Causal graph validation
    scm             Module 5  SCM fitting (mechanism & noise estimation)
    rca             Module 6  Root cause analysis
    intervention    Module 7  Intervention analysis (do-operations)
    counterfactual  Module 8  Counterfactual XAI (abduction/action/prediction)
    dashboard       Module 9  Actionable RCA dashboard

Prototyped on NASA C-MAPSS; the stage interfaces are dataset-agnostic so the
same pipeline can be re-targeted at other industrial time-series datasets.
"""

from . import (
    artifacts,
    counterfactual,
    dashboard,
    discovery,
    features,
    intervention,
    preprocessing,
    rca,
    scm,
    validation,
)

__all__ = [
    "preprocessing",
    "features",
    "discovery",
    "validation",
    "scm",
    "rca",
    "intervention",
    "counterfactual",
    "dashboard",
    "artifacts",
]
