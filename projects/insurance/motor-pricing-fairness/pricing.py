"""Frequency x severity pricing models and three ways to treat a protected attribute.

Models (scikit-learn only):
  glm   Poisson GLM for claim frequency, Gamma GLM for claim size (binned ages, one-hot categories)
  gbm   gradient boosted trees with Poisson and Gamma losses

Prices, following Lindholm, Richman, Tsanakas & Wüthrich (2022), "Discrimination-free insurance pricing":
  aware                the model sees the protected attribute D (here: driver age)
  unaware              D is dropped, but other features (bonus-malus, vehicle, region) can stand in for it
  discrimination_free  fit with D, then average the prediction over the portfolio's distribution of D:
                       price(x) = sum_d mu(x, d) * P(D = d). D has no direct effect, and proxies
                       can't bring it back, because the model already explains D's part with D itself.
Unaware and discrimination-free prices are scaled so the total premium on the training set matches aware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from data import CATEGORICAL, FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

PROTECTED = "DrivAge"


def _glm_features(cols: list[str]) -> ColumnTransformer:
    num_binned = [c for c in ["VehAge", "DrivAge"] if c in cols]
    cats = [c for c in CATEGORICAL + ["VehPower"] if c in cols]
    return ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=10, strategy="quantile", encode="onehot", subsample=None), num_binned),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cats),
        ("bonus", StandardScaler(), ["BonusMalus"]),
        ("density", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["Density"]),
    ])


def _gbm(loss: str, cols: list[str], **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features=[c for c in cols if c in CATEGORICAL], random_state=0, **kw)


def _x(df: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    x = df[cols].copy()
    if kind == "gbm":
        for c in cols:
            if c in CATEGORICAL:
                x[c] = x[c].astype("category")
    return x


@dataclass
class FreqSev:
    """One frequency model and one severity model on the same feature set."""
    kind: str               # "glm" or "gbm"
    cols: list[str]

    def fit(self, train: pd.DataFrame) -> FreqSev:
        if self.kind == "glm":
            self.freq = make_pipeline(_glm_features(self.cols), PoissonRegressor(alpha=1e-4, solver="newton-cholesky"))
            self.sev = make_pipeline(_glm_features(self.cols), GammaRegressor(alpha=1e-2, solver="newton-cholesky"))
        else:
            self.freq = _gbm("poisson", self.cols)
            self.sev = _gbm("gamma", self.cols, max_depth=3)
        step = self.freq.steps[-1][0] + "__sample_weight" if self.kind == "glm" else "sample_weight"
        self.freq.fit(_x(train, self.cols, self.kind), train.ClaimNb / train.Exposure, **{step: train.Exposure})
        claims = train[train.ClaimNb > 0]
        step = self.sev.steps[-1][0] + "__sample_weight" if self.kind == "glm" else "sample_weight"
        self.sev.fit(_x(claims, self.cols, self.kind), claims.ClaimAmount / claims.ClaimNb, **{step: claims.ClaimNb})
        return self

    def frequency(self, df: pd.DataFrame) -> np.ndarray:
        return self.freq.predict(_x(df, self.cols, self.kind))

    def severity(self, df: pd.DataFrame) -> np.ndarray:
        return self.sev.predict(_x(df, self.cols, self.kind))

    def pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claim cost per policy-year."""
        return self.frequency(df) * self.severity(df)


def protected_distribution(train: pd.DataFrame) -> pd.Series:
    """P(D = d), weighted by exposure."""
    return train.groupby(PROTECTED).Exposure.sum() / train.Exposure.sum()


def discrimination_free(model: FreqSev, df: pd.DataFrame, p_d: pd.Series) -> np.ndarray:
    """sum_d mu(x, d) P(D = d): predict every policy at every value of D and average."""
    out = np.zeros(len(df))
    x = df.copy()
    for d, p in p_d.items():
        x[PROTECTED] = d
        out += p * model.pure_premium(x)
    return out


def fit_prices(train: pd.DataFrame, kind: str) -> dict:
    aware = FreqSev(kind, FEATURES).fit(train)
    unaware = FreqSev(kind, [c for c in FEATURES if c != PROTECTED]).fit(train)
    p_d = protected_distribution(train)
    w = train.Exposure
    total = (aware.pure_premium(train) * w).sum()
    k_un = total / (unaware.pure_premium(train) * w).sum()
    k_df = total / (discrimination_free(aware, train, p_d) * w).sum()
    return {"aware": aware, "unaware": unaware, "p_d": p_d, "scale": {"unaware": k_un, "discrimination_free": k_df}}


def price(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "aware": models["aware"].pure_premium(df),
        "unaware": models["unaware"].pure_premium(df) * models["scale"]["unaware"],
        "discrimination_free": discrimination_free(models["aware"], df, models["p_d"])
        * models["scale"]["discrimination_free"],
    }, index=df.index)
