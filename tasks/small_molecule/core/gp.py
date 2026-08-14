"""Gaussian-process surrogate for Vina-style SMILES → score regression.

The single public class :class:`GPSurrogate` wraps a gpytorch ``ExactGP``
whose covariance kernel is selected by ``GPConfig.impl``:

* ``"fingerprint+tanimoto"`` — RDKit Morgan fingerprints + the Tanimoto
  similarity kernel from the ``gauche`` package.
* ``"smiles-strkernel"`` — raw SMILES strings + the subsequence string
  kernel from ``gauche``, with the alphabet auto-built from the training
  set.

The outer API is identical for both impls: ``fit(smiles, scores)`` trains
the GP hyperparameters with Adam under a Cholesky-jitter ladder, and
``predict(smiles)`` returns ``(mean, std)`` for each SMILES. When every
jitter attempt fails the surrogate falls back to **prior mode**
(hyperparameters left at initialization, no observed data) so that
downstream BO loops always have a callable surrogate.

Public surface (re-exported from :mod:`tasks.small_molecule.core.__init__``):

- :class:`GPSurrogate` -- the only class the caller instantiates.

Everything else (``GPConfig``, ``GPImpl``, the internal featurizers and
GP models, the jitter ladder helpers) is importable via
``tasks.small_molecule.core.gp.<name>`` but is intentionally not part of the canonical
public surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Union

import numpy as np
import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gauche.kernels.string_kernels.sskkernel import SubsequenceStringKernel

LOGGER = logging.getLogger(__name__)


__all__ = ["GPSurrogate"]


GPImpl = Literal["fingerprint+tanimoto", "smiles-strkernel"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPConfig:
    """Hyperparameters and feature choices for :class:`GPSurrogate`.

    The jitter ladder ``[min_jitter, min_jitter * jitter_multiplier,
    min_jitter * jitter_multiplier^2, …]`` is enumerated at every Adam
    step. As soon as one ``loss.backward()`` succeeds the inner loop
    exits and the outer step counter advances. If the inner loop runs
    out of attempts on any step the training is aborted and the
    surrogate falls back to **prior mode**.

    ``smiles-strkernel`` does not require a pre-declared alphabet; it
    is built lazily on the first ``fit()`` call from the union of
    characters in the training SMILES and reused thereafter.
    """

    impl: GPImpl = "fingerprint+tanimoto"

    # Numerical-stability ladder (gpytorch.settings.cholesky_jitter).
    min_jitter: float = 1e-6
    jitter_multiplier: float = 10.0
    max_jitter: float = 1e-1

    # Adam hyperparameters.
    learning_rate: float = 0.1
    fit_n_itersteps: int = 100

    # Hardware / RNG.
    device: str = "cuda"
    seed: int = 0

    # Featurization.
    fp_radius: int = 2
    fp_n_bits: int = 2048
    smiles_maxlen: int = 80

    # Output normalization.
    standardize_y: bool = True


# ---------------------------------------------------------------------------
# Featurizers
# ---------------------------------------------------------------------------


def _smiles_to_fingerprints(
    smiles_list: Iterable[str],
    radius: int,
    n_bits: int,
) -> torch.Tensor:
    """RDKit Morgan fingerprints as a ``float32`` tensor ``(n, n_bits)``.

    Invalid SMILES yield an all-zero fingerprint row; the kernel still
    defines a similarity (zero to every other fingerprint), so the GP
    can fit and predict on the valid rows without crashing.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")  # suppress RDKit parse warnings on bad SMILES

    out = np.zeros((0, n_bits), dtype=np.float32)
    rows: list[np.ndarray] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles or ""))
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        bit_string = fp.ToBitString()
        arr = np.frombuffer(bit_string.encode("ascii"), dtype=np.uint8) - ord("0")
        rows.append(arr.astype(np.float32))
    if rows:
        out = np.stack(rows, axis=0)
    return torch.from_numpy(out)


def _build_smiles_alphabet(smiles_list: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    """Build the alphabet + integer index from the training SMILES.

    Index 0 is reserved for padding / unseen characters. The alphabet
    preserves first-seen insertion order so two ``fit()`` calls on the
    same training set always produce identical kernels.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for smiles in smiles_list:
        for ch in str(smiles or ""):
            if ch not in seen_set:
                seen_set.add(ch)
                seen.append(ch)
    index = {ch: i + 1 for i, ch in enumerate(seen)}
    return seen, index


def _smiles_to_strings(
    smiles_list: Iterable[str],
    index: dict[str, int],
    maxlen: int,
) -> torch.Tensor:
    """Integer-encoded SMILES as an ``int64`` tensor ``(n, maxlen)``.

    Unseen characters and positions beyond ``maxlen`` encode to 0
    (the padding / unknown slot). Right-pad with 0 if the string is
    shorter than ``maxlen``.
    """
    out = np.zeros((0, maxlen), dtype=np.int64)
    rows: list[np.ndarray] = []
    for smiles in smiles_list:
        text = str(smiles or "")
        encoded = np.zeros(maxlen, dtype=np.int64)
        for j, ch in enumerate(text[:maxlen]):
            encoded[j] = index.get(ch, 0)
        rows.append(encoded)
    if rows:
        out = np.stack(rows, axis=0)
    return torch.from_numpy(out)


def _smiles_alphabet_embds(alphabet: list[str]) -> torch.Tensor:
    """One-hot embedding matrix of shape ``(len(alphabet) + 1, len(alphabet))``.

    Row ``i+1`` corresponds to character ``alphabet[i]``; row 0 is the
    padding/unknown slot (zero vector). Matches the layout produced by
    ``gauche.kernels.string_kernels.sskkernel.build_one_hot``.
    """
    dim = len(alphabet)
    embs = torch.zeros((dim + 1, dim), dtype=torch.float32)
    for i in range(dim):
        embs[i + 1, i] = 1.0
    return embs


# ---------------------------------------------------------------------------
# Internal GP models
# ---------------------------------------------------------------------------


class _TanimotoGPModel(gpytorch.models.ExactGP):
    """Exact GP with Morgan-FP Tanimoto kernel. ``train_x`` is float32 ``(n, n_bits)``."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            TanimotoKernel(),
            outputscale_constraint=gpytorch.constraints.Positive(),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _StringGPModel(gpytorch.models.ExactGP):
    """Exact GP with the subsequence string kernel. ``train_x`` is int64 ``(n, maxlen)``."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        embds: torch.Tensor,
        index: dict[str, int],
        alphabet: list[str],
        maxlen: int,
        device: torch.device,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        base_kernel = SubsequenceStringKernel(
            embds, index, alphabet=alphabet, maxlen=maxlen,
        )
        # The kernel's __init__ auto-detects CUDA via ``tensor_kwargs``;
        # rewrite the device to match the requested device so ``.to()`` works.
        base_kernel.tensor_kwargs["device"] = device
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=gpytorch.constraints.Positive(),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


# ---------------------------------------------------------------------------
# Jitter ladder
# ---------------------------------------------------------------------------


def _jitter_ladder(min_jitter: float, multiplier: float, max_jitter: float) -> list[float]:
    """Geometric ladder ``[min_jitter, min_jitter*M, min_jitter*M^2, …]`` capped at ``max_jitter``.

    Edge cases:
    * ``min_jitter > max_jitter`` → ``[]`` (no usable jitter values).
    * ``multiplier <= 1.0`` → ``[min_jitter]`` (the ladder would never grow;
      we return a single attempt rather than looping forever).
    """
    if multiplier <= 1.0:
        return [float(min_jitter)]
    ladder: list[float] = []
    value = float(min_jitter)
    while value <= float(max_jitter) * (1.0 + 1e-12):
        ladder.append(value)
        value *= float(multiplier)
    return ladder


def _destandardize(
    mean: torch.Tensor,
    var: torch.Tensor,
    y_mean: Optional[float],
    y_std: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse ``_normalize_y``: map (μ, σ²) on standardized y back to original scale."""
    if y_mean is None or y_std is None:
        return mean, var
    return mean * y_std + y_mean, var * (y_std ** 2)


def _canonicalize_and_dedup(
    smiles: list[str],
    scores: list[float],
) -> tuple[list[str], list[float]]:
    """Canonicalize, group by canonical SMILES, average duplicate scores.

    Returns ``(unique_smiles, averaged_scores)`` preserving first-seen
    order. Skips empty / unparseable SMILES with a WARNING. Repeated
    evaluations of the same canonical molecule (common in BO loops that
    rescore frontier candidates across iterations) collapse into a
    single row with the arithmetic mean of the supplied scores — the
    simplest deterministic reducer.

    Validation uses RDKit's ``MolFromSmiles`` directly because
    ``canonicalize_smiles`` returns the input string unchanged for
    unparseable SMILES (silent pass-through), which would let
    unparseable inputs pollute the training set.
    """
    from tasks.small_molecule.core.docking import canonicalize_smiles
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    skipped = 0
    for raw, raw_score in zip(smiles, scores):
        text = str(raw or "").strip()
        if not text:
            skipped += 1
            continue
        if Chem.MolFromSmiles(text) is None:
            skipped += 1
            continue
        canon = canonicalize_smiles(text)
        if canon not in grouped:
            grouped[canon] = []
            order.append(canon)
        grouped[canon].append(float(raw_score))
    if skipped:
        LOGGER.warning(
            "GPSurrogate.fit: skipped %d invalid/empty SMILES during canonicalization.",
            skipped,
        )
    averaged = [sum(grouped[c]) / len(grouped[c]) for c in order]
    return order, averaged


def _dedup_by_feature_row(
    features: torch.Tensor,
    scores: list[float],
) -> tuple[torch.Tensor, list[float]]:
    """Drop rows whose feature vector already appeared earlier.

    Different SMILES occasionally hash to identical fingerprints when
    ``n_bits`` is small (rare in production with 2048 bits, common in
    tests with 64 bits). Identical rows would make the kernel matrix
    rank-deficient and Cholesky would fail. Returns the first occurrence
    of each unique row plus its score.
    """
    seen: set[tuple] = set()
    keep_rows: list[int] = []
    keep_scores: list[float] = []
    n_total = features.shape[0]
    for i in range(n_total):
        key = tuple(features[i].detach().cpu().numpy().tolist())
        if key in seen:
            continue
        seen.add(key)
        keep_rows.append(i)
        keep_scores.append(scores[i])
    if len(keep_rows) < n_total:
        LOGGER.warning(
            "GPSurrogate.fit: dropped %d rows whose feature vector duplicates an earlier one.",
            n_total - len(keep_rows),
        )
    if not keep_rows:
        return features, scores
    keep_idx = torch.tensor(keep_rows, dtype=torch.long, device=features.device)
    return features.index_select(0, keep_idx), keep_scores


def _normalize_y(
    scores: list[float],
    standardize: bool,
    device: torch.device,
) -> tuple[torch.Tensor, Optional[float], Optional[float]]:
    """Standardize ``scores`` to (μ=0, σ=1) when ``standardize`` is True.

    Returns ``(y_tensor, y_mean, y_std)``; the mean and std are ``None``
    when standardization is disabled. Falls back to un-normalized data
    when ``standardize=True`` but ``std == 0`` (all scores equal) or
    non-finite, with a WARNING logged.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if not standardize:
        return (
            torch.tensor(arr, dtype=torch.float32, device=device),
            None,
            None,
        )
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0.0 or not np.isfinite(std):
        LOGGER.warning(
            "standardize_y=True but y has zero or non-finite std (std=%s); "
            "leaving y un-normalized.", std,
        )
        return (
            torch.tensor(arr, dtype=torch.float32, device=device),
            0.0,
            1.0,
        )
    return (
        torch.tensor((arr - mean) / std, dtype=torch.float32, device=device),
        mean,
        std,
    )


# ---------------------------------------------------------------------------
# Public surrogate
# ---------------------------------------------------------------------------


class GPSurrogate:
    """Unified GP surrogate over SMILES → Vina-score regression.

    Choose the kernel via :attr:`GPConfig.impl`:

    * ``"fingerprint+tanimoto"`` (default) -- RDKit Morgan fingerprints
      + Tanimoto similarity kernel.
    * ``"smiles-strkernel"`` -- raw SMILES + the ``gauche`` subsequence
      string kernel with an alphabet auto-built from the training set.

    ``fit`` trains the GP hyperparameters with Adam under a Cholesky-
    jitter ladder. If every jitter attempt fails on any step the
    surrogate falls back to **prior mode** (no observed data,
    hyperparameters left at initialization) and :attr:`in_prior_mode`
    becomes ``True``. ``predict`` always works once :attr:`is_fitted`
    is ``True``; in prior mode it returns the GP prior (mean from
    ``ConstantMean``, variance from the kernel on the diagonal).
    """

    def __init__(self, config: Optional[GPConfig] = None) -> None:
        self.config: GPConfig = config if config is not None else GPConfig()
        self.device: torch.device = self._resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)

        self.model: Optional[gpytorch.models.ExactGP] = None
        self.likelihood: Optional[GaussianLikelihood] = None
        self.train_x_feats: Optional[torch.Tensor] = None
        self.train_y: Optional[torch.Tensor] = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None

        # Lazy alphabet for the string kernel; built on first fit().
        self._alphabet: Optional[list[str]] = None
        self._alphabet_index: Optional[dict[str, int]] = None
        self._alphabet_embds: Optional[torch.Tensor] = None

        self._fitted: bool = False
        self._in_prior_mode: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def in_prior_mode(self) -> bool:
        """``True`` if ``fit`` aborted to prior mode (Cholesky past ``max_jitter``)."""
        return self._in_prior_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        text = str(spec or "cuda").strip().lower()
        if text == "cpu":
            return torch.device("cpu")
        if text.startswith("cuda") and torch.cuda.is_available():
            return torch.device(text)
        if text.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA unavailable; falling back to CPU (requested %s).", spec)
            return torch.device("cpu")
        # Unrecognized → honor as-is (lets tests pass "cpu" / "cuda:0")
        return torch.device(text)

    def _featurize_train(self, smiles_list: list[str]) -> torch.Tensor:
        if self.config.impl == "fingerprint+tanimoto":
            feats = _smiles_to_fingerprints(
                smiles_list, radius=self.config.fp_radius, n_bits=self.config.fp_n_bits
            )
            return feats.to(device=self.device, dtype=torch.float32)
        if self.config.impl == "smiles-strkernel":
            if self._alphabet is None or self._alphabet_index is None:
                self._alphabet, self._alphabet_index = _build_smiles_alphabet(smiles_list)
                self._alphabet_embds = _smiles_alphabet_embds(self._alphabet).to(self.device)
            else:
                # Extend alphabet if new characters appeared; rebuild embds.
                extended = False
                for smiles in smiles_list:
                    for ch in str(smiles or ""):
                        if ch not in self._alphabet_index:
                            self._alphabet.append(ch)
                            self._alphabet_index[ch] = len(self._alphabet)
                            extended = True
                if extended:
                    self._alphabet_embds = _smiles_alphabet_embds(self._alphabet).to(self.device)
            assert self._alphabet_index is not None and self._alphabet_embds is not None
            feats = _smiles_to_strings(
                smiles_list, index=self._alphabet_index, maxlen=self.config.smiles_maxlen
            )
            return feats.to(device=self.device, dtype=torch.int64)
        raise ValueError(f"Unknown impl: {self.config.impl!r}")

    def _featurize_predict(self, smiles_list: list[str]) -> torch.Tensor:
        if self.config.impl == "fingerprint+tanimoto":
            feats = _smiles_to_fingerprints(
                smiles_list, radius=self.config.fp_radius, n_bits=self.config.fp_n_bits
            )
            return feats.to(device=self.device, dtype=torch.float32)
        if self.config.impl == "smiles-strkernel":
            assert self._alphabet_index is not None, (
                "predict() before fit() is not supported for smiles-strkernel"
            )
            feats = _smiles_to_strings(
                smiles_list, index=self._alphabet_index, maxlen=self.config.smiles_maxlen
            )
            return feats.to(device=self.device, dtype=torch.int64)
        raise ValueError(f"Unknown impl: {self.config.impl!r}")

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> gpytorch.models.ExactGP:
        likelihood = GaussianLikelihood().to(self.device)
        if self.config.impl == "fingerprint+tanimoto":
            model = _TanimotoGPModel(train_x, train_y, likelihood).to(self.device)
        elif self.config.impl == "smiles-strkernel":
            assert (
                self._alphabet is not None
                and self._alphabet_index is not None
                and self._alphabet_embds is not None
            ), "alphabet not built; call fit() before building the string model"
            model = _StringGPModel(
                train_x,
                train_y,
                likelihood,
                embds=self._alphabet_embds,
                index=self._alphabet_index,
                alphabet=self._alphabet,
                maxlen=self.config.smiles_maxlen,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown impl: {self.config.impl!r}")
        return model

    def _normalize_y(self, scores: list[float]) -> torch.Tensor:
        tensor, y_mean, y_std = _normalize_y(
            scores, standardize=self.config.standardize_y, device=self.device,
        )
        self._y_mean, self._y_std = y_mean, y_std
        return tensor

    @staticmethod
    def _destandardize(
        mean: torch.Tensor, var: torch.Tensor, y_mean: Optional[float], y_std: Optional[float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _destandardize(mean, var, y_mean, y_std)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles: list[str],
        scores: list[float],
        *,
        verbose: bool = False,
    ) -> "GPSurrogate":
        """Fit GP hyperparameters on ``(smiles, scores)`` pairs.

        Trains for ``config.fit_n_itersteps`` Adam iterations. Each step
        tries the Cholesky jitter ladder ``[min_jitter, min_jitter*M, …]``
        (capped at ``max_jitter``); if the step succeeds with any jitter
        value the outer counter advances, otherwise training aborts to
        prior mode.

        Returns ``self`` for chaining. Sets :attr:`is_fitted` to ``True``
        in both success and prior-mode fallback paths.
        """
        smiles = list(smiles)
        scores = list(scores)
        if not smiles:
            raise ValueError("GPSurrogate.fit requires at least one SMILES.")
        if len(smiles) != len(scores):
            raise ValueError(
                f"smiles length ({len(smiles)}) does not match scores length ({len(scores)})."
            )

        # Reset prior-mode flags; each fit() is a fresh attempt.
        self._fitted = False
        self._in_prior_mode = False

        # Canonicalize + dedup SMILES (handles repeated BO-loop evaluations
        # of the same molecule and "OCC" vs "CCO" identity issues).
        smiles, scores = _canonicalize_and_dedup(smiles, scores)
        if not smiles:
            raise ValueError(
                "GPSurrogate.fit: every supplied SMILES was empty or unparseable."
            )

        # Featurize.
        train_x = self._featurize_train(smiles)

        # Dedup by feature row (handles the rare case where two distinct
        # SMILES hash to identical fingerprints with small n_bits).
        train_x, scores = _dedup_by_feature_row(train_x, scores)

        # Normalize targets using the deduped scores.
        train_y = self._normalize_y(scores)

        # Build (or rebuild) the model. set_train_data is called by ExactGP
        # already during construction; the likelihood created inside the model
        # is the one we keep and reference as ``self.likelihood``.
        self.model = self._build_model(train_x, train_y)
        self.likelihood = self.model.likelihood
        self.train_x_feats = train_x
        self.train_y = train_y

        # Adam over the GP hyperparameters. ``model.parameters()`` already
        # includes the likelihood's parameters transitively (ExactGP stores
        # its likelihood as ``self.likelihood`` and exposes its parameters),
        # so listing them again would produce "duplicate parameters" warnings.
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        ladder = _jitter_ladder(
            self.config.min_jitter,
            self.config.jitter_multiplier,
            self.config.max_jitter,
        )
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()
        for step in range(self.config.fit_n_itersteps):
            succeeded = False
            for trial in ladder:
                try:
                    with gpytorch.settings.cholesky_jitter(float_value=trial):
                        optimizer.zero_grad()
                        output = self.model(self.train_x_feats)
                        loss = -mll(output, self.train_y)
                        loss.backward()
                        optimizer.step()
                    succeeded = True
                    break
                except Exception as exc:  # NotPSDError, RuntimeError, generic gpytorch numerical issues
                    if verbose:
                        LOGGER.debug(
                            "GP step %d jitter=%.3e failed (%s: %s); trying next.",
                            step, trial, type(exc).__name__, exc,
                        )
                    continue
            if not succeeded:
                LOGGER.warning(
                    "GPSurrogate training aborted at step %d/%d: Cholesky failed "
                    "for every jitter value in %s (max_jitter=%g). Falling back to "
                    "prior mode (no observed data).",
                    step + 1, self.config.fit_n_itersteps, ladder, self.config.max_jitter,
                )
                self._fitted = True
                self._in_prior_mode = True
                # In prior mode, observed data contributes nothing → drop it.
                self.train_x_feats = None
                self.train_y = None
                self.model.eval()
                self.likelihood.eval()
                return self

        self._fitted = True
        self._in_prior_mode = False
        self.model.eval()
        self.likelihood.eval()
        return self

    def predict(
        self,
        smiles: list[str],
        *,
        return_tensor: bool = False,
    ) -> Union[
        tuple[np.ndarray, np.ndarray],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Posterior (or prior) predictive on a batch of SMILES.

        Returns ``(mean, std)`` where each element is shape ``(n,)``.
        In prior mode the mean is the GP prior mean (constant from
        ``ConstantMean``) and the variance comes from the kernel
        diagonal on the test inputs only.

        Raises ``RuntimeError`` if :attr:`is_fitted` is ``False``.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "GPSurrogate.predict called before fit(); call fit(smiles, scores) first."
            )
        smiles = list(smiles)
        if not smiles:
            empty = np.zeros((0,), dtype=np.float32)
            return (empty, empty)

        test_x = self._featurize_predict(smiles)

        with torch.no_grad():
            if self._in_prior_mode or self.train_x_feats is None:
                # Prior predictive: no data contribution.
                prior = self.model(test_x)
                mean = prior.mean
                var = prior.variance
            else:
                output = self.model(test_x)
                observed = self.likelihood(output)
                mean = observed.mean
                var = observed.variance

            mean, var = self._destandardize(mean, var, self._y_mean, self._y_std)
            std = torch.sqrt(torch.clamp(var, min=0.0))
            if not return_tensor:
                mean = mean.detach().cpu().numpy()
                std = std.detach().cpu().numpy()
        return mean, std
