"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from climada.engine.option_appraisal.MCDM.category import (
    CategorizedObject,
    CategoryLike,
    CategorySpace,
)
from climada.engine.option_appraisal.MCDM.constants import DEFAULT_CRITERION_BASE_WEIGHT
from climada.engine.option_appraisal.MCDM.weights import WeightedItemZeroOne

LOGGER = logging.getLogger(__name__)


class Criterion(CategorizedObject, WeightedItemZeroOne):
    def __init__(
        self,
        name: str,
        categories: CategoryLike | Sequence[CategoryLike] | None = None,
        space: CategorySpace | None = None,
        data: pd.Series | None = None,
        obj_maximize: bool = True,
        base_weight: float = DEFAULT_CRITERION_BASE_WEIGHT,
    ) -> None:
        CategorizedObject.__init__(self, name, categories, space)
        WeightedItemZeroOne.__init__(self, base_weight)
        if data is None:
            data = pd.Series()

        self.data = data
        self.data.name = name
        self.obj_maximize = obj_maximize

    def __repr__(self, indent=4) -> str:
        """
        Provides a custom, formatted, multi-line string representation
        of the Criterion object for better readability.
        """
        indent_space = " " * indent
        # Format the Categories Set
        # Use a list comprehension to format each category on a new line
        formatted_categories = ",\n".join(
            f"{cat.__repr__(indent=indent*2)}"
            for cat in sorted(
                self.categories, key=lambda c: c.name
            )  # Sorting for consistency
        )

        return (
            f"Criterion(\n"
            f"{indent_space}name='{self.name}',\n"
            f"{indent_space}categories={{\n{formatted_categories}\n{indent_space}}},\n"
            f"{indent_space}obj_maximise={self.obj_maximize}\n"
            f"{indent_space}base_weight={self.weight}\n"
            f"{indent_space}categories_effective_weight={self.category_effective_weight}\n"
            f")"
        )

    def effective_weight(self, categories_influence=0.5):
        return (
            self.weight * (1 - categories_influence)
            + self.category_effective_weight * categories_influence
        )

    @property
    def category_effective_weight(self):
        return np.array([cat.effective_weight for cat in self.categories]).prod() ** (
            1 / len(self.categories)
        )

    @property
    def categories_weight_profile(self):
        return {cat.name: cat.weight for cat in self.categories}
