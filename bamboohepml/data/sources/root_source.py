"""ROOT data source implementation.

Loads data from ROOT files via uproot.  Supports:
  - Direct full podio-style branch paths (e.g. ``Collection/Collection.field``).
  - Optional ``branch_magic`` alias mapping for backward compatibility.
  - ``file_magic`` for virtual variables derived from filename patterns.
  - Fractional ``load_range`` for train / val / test splitting.
  - Glob / directory resolution for multi-file datasets.
"""

from __future__ import annotations

import math
import re
from typing import Any

import awkward as ak

from ..logger import _logger
from .base import DataSource


class ROOTDataSource(DataSource):
    """ROOT file data source backed by uproot."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_branches(self, branches: list[str]) -> ak.Array:
        """Load requested branches from all ROOT files.

        The method automatically separates *file_magic* virtual variables
        (computed from filename patterns) from real branches that must be
        read from the TTree.

        For real branches the loader first checks whether uproot can find
        the branch directly (supporting full podio paths like
        ``EcalBarrelCollection/EcalBarrelCollection.energy``).  If a
        ``branch_magic`` alias mapping is configured, short names are
        resolved through uproot's alias mechanism.

        Args:
            branches: Branch names (or full podio paths) to load.

        Returns:
            Concatenated awkward array spanning all files.
        """
        import uproot

        from ..tools import _concat

        table: list[ak.Array] = []
        branches = list(branches)

        for filepath in self._file_paths:
            try:
                with uproot.open(filepath) as f:
                    treename = self._resolve_treename(f, filepath)
                    tree = f[treename]

                    start, stop = self._entry_range(tree)

                    # Partition branches into file_magic vars vs real.
                    file_magic_vars: set[str] = set()
                    if self.config.file_magic is not None:
                        file_magic_vars = set(self.config.file_magic.keys())
                    real_branches = [
                        b for b in branches if b not in file_magic_vars
                    ]

                    # Load real branches.
                    outputs = self._load_real_branches(
                        tree, real_branches, start, stop
                    )

                    # Attach file_magic virtual variables.
                    outputs = self._attach_file_magic(
                        outputs, filepath, file_magic_vars
                    )

                    # Clean up dummy field if present.
                    if "__dummy__" in outputs.fields:
                        outputs = outputs[
                            [
                                fld
                                for fld in outputs.fields
                                if fld != "__dummy__"
                            ]
                        ]

                    table.append(outputs)

            except Exception as e:
                _logger.error(f"Error reading {filepath}: {e}")
                import traceback

                _logger.error(traceback.format_exc())

        if len(table) == 0:
            raise RuntimeError(
                f"Loaded zero records from {self._file_paths}."
            )

        return _concat(table)

    def get_available_branches(self) -> list[str]:
        """Return the list of branches available in the first ROOT file."""
        import uproot

        if not self._file_paths:
            return []
        filepath = self._file_paths[0]
        try:
            with uproot.open(filepath) as f:
                treename = self._resolve_treename(f, filepath)
                return list(f[treename].keys())
        except Exception as e:
            _logger.error(f"Error listing branches: {e}")
            return []

    def get_num_events(self) -> int | None:
        """Return total number of events across all files."""
        import uproot

        total = 0
        for filepath in self._file_paths:
            try:
                with uproot.open(filepath) as f:
                    treename = self._resolve_treename(f, filepath)
                    total += f[treename].num_entries
            except Exception:
                continue
        return total if total > 0 else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_treename(self, f: Any, filepath: str) -> str:
        """Determine the TTree name to use."""
        treename = self.config.treename
        if treename is not None:
            return treename
        treenames = {
            k.split(";")[0]
            for k, v in f.items()
            if getattr(v, "classname", "") == "TTree"
        }
        if len(treenames) == 1:
            return treenames.pop()
        raise RuntimeError(
            f"Multiple trees found in {filepath}: {treenames}. "
            f"Specify `treename` explicitly."
        )

    def _entry_range(
        self, tree: Any
    ) -> tuple[int | None, int | None]:
        """Compute start / stop indices from fractional load_range."""
        load_range = self.config.load_range
        if load_range is None:
            return None, None
        start = math.trunc(load_range[0] * tree.num_entries)
        stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        return start, stop

    def _load_real_branches(
        self,
        tree: Any,
        real_branches: list[str],
        start: int | None,
        stop: int | None,
    ) -> ak.Array:
        """Load real (non-virtual) branches from the TTree.

        Strategy:
        1.  Branches available directly in the tree are loaded as-is.
            This includes full podio paths (``Collection/Collection.field``).
        2.  Missing branches are looked up in ``branch_magic``; if found
            uproot aliases are used.
        3.  Remaining missing branches are silently skipped (they may be
            computed downstream by FeatureGraph).
        """
        import numpy as np

        if not real_branches:
            n = self._dummy_length(tree, start, stop)
            return ak.Array({"__dummy__": np.zeros(n)})

        tree_keys = set(tree.keys())
        branch_magic = self.config.branch_magic or {}

        direct: list[str] = []
        aliases: dict[str, str] = {}

        for name in real_branches:
            if name in tree_keys:
                # Directly available (works for full podio paths).
                direct.append(name)
            elif name in branch_magic:
                # Legacy alias mapping.
                aliases[name] = branch_magic[name]
            else:
                _logger.debug(f"Skipping unavailable branch: {name}")

        all_load = list(aliases.keys()) + direct
        if not all_load:
            import numpy as np

            n = self._dummy_length(tree, start, stop)
            return ak.Array({"__dummy__": np.zeros(n)})

        return tree.arrays(
            all_load,
            aliases=aliases if aliases else None,
            entry_start=start,
            entry_stop=stop,
        )

    def _attach_file_magic(
        self,
        outputs: ak.Array,
        filepath: str,
        file_magic_vars: set[str],
    ) -> ak.Array:
        """Add virtual variables derived from filename patterns."""
        if self.config.file_magic is None:
            return outputs

        for var, value_dict in self.config.file_magic.items():
            matched_value = 0
            for fn_pattern, value in value_dict.items():
                if re.search(fn_pattern, filepath):
                    matched_value = value
                    break
            outputs[var] = matched_value

        return outputs

    @staticmethod
    def _dummy_length(
        tree: Any, start: int | None, stop: int | None
    ) -> int:
        """Compute dummy array length when no real branches are loaded."""
        if start is not None and stop is not None:
            return stop - start
        return tree.num_entries
