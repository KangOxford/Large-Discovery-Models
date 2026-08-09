"""bo/ldm/dsl: Trust Region DSL for LLM-controlled Bayesian Optimisation.

This subpackage is PRIVATE to bo/ldm. The only public re-exports happen in
``bo/ldm/__init__.py`` via :class:`bo.ldm.SearchSpaceAtom` and
:class:`bo.ldm.BiasAtom`. Concrete atoms (LocalSearch, NeighborSampling, ...)
Not, MaxCysteine, ...) MUST NOT be imported directly from outside ``bo/ldm``.
"""