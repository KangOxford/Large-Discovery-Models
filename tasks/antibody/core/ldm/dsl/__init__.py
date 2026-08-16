"""core/ldm/dsl: Trust Region DSL for LLM-controlled Bayesian Optimisation.

This subpackage is PRIVATE to core/ldm. The only public re-exports happen in
``core/ldm/__init__.py`` via :class:`tasks.antibody.core.ldm.SearchSpaceAtom` and
:class:`tasks.antibody.core.ldm.BiasAtom`. Concrete atoms (LocalSearch, NeighborSampling, ...)
Not, MaxCysteine, ...) MUST NOT be imported directly from outside ``core/ldm``.
"""