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

from typing import Generic, TypeVar
from warnings import warn

from climada.engine.option_appraisal.MCDM.constants import (
    DEFAULT_ITEM_WEIGHT,
    IMPORTANCE_MATCH,
)


class WeightedItem:
    """Mixin providing a validated weight attribute.

    Parameters
    ----------
    weight : float or str or None
        Initial weight. Strings are resolved via ``IMPORTANCE_MATCH``.
        ``None`` defaults to ``DEFAULT_WEIGHT``.
    """

    def __init__(self, weight=None) -> None:
        self.weight = weight

    @property
    def weight(self) -> float:
        """Direct weight of this item, in [0, 1]."""
        return self._weight

    @weight.setter
    def weight(self, value):
        if value is None:
            value = DEFAULT_ITEM_WEIGHT
        if isinstance(value, str):
            try:
                value = IMPORTANCE_MATCH[value]
            except KeyError as err:
                err.add_note(
                    f"Importance '{value}' is not defined. "
                    f"Must be one of {list(IMPORTANCE_MATCH.keys())}"
                )
                raise
        if not 0.0 <= value:
            raise ValueError(f"Weight must be positive, got {value}.")
        self._weight = value


class WeightedItemZeroOne(WeightedItem):
    """Mixin providing a validated weight attribute.

    Parameters
    ----------
    weight : float or str or None
        Initial weight. Strings are resolved via ``IMPORTANCE_MATCH``.
        ``None`` defaults to ``DEFAULT_WEIGHT``.
    """

    def __init__(self, weight=None) -> None:
        WeightedItem.__init__(self, weight)

    @property
    def weight(self) -> float:
        """Direct weight of this item, in [0, 1]."""
        return self._weight

    @weight.setter
    def weight(self, value):
        if value is None:
            value = DEFAULT_ITEM_WEIGHT
        if isinstance(value, str):
            try:
                value = IMPORTANCE_MATCH[value]
            except KeyError as err:
                err.add_note(
                    f"Importance '{value}' is not defined. "
                    f"Must be one of {list(IMPORTANCE_MATCH.keys())}"
                )
                raise
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {value}.")
        self._weight = value


T = TypeVar("T", bound="BudgetWeightItem")


class BudgetedSet(Generic[T]):
    """Named set of budget-weighted items.

    Parameters
    ----------
    name : str
        Name of the set.
    members : set of T
        Items sharing the budget.
    """

    def __init__(self, name, members: dict[str, T] | None = None) -> None:
        self.name = name
        self.members = members if members else {}

    @property
    def total_budget(self) -> float:
        return sum(m.weight for m in self.members.values())


class BudgetWeightItem(WeightedItem):
    """Mixin providing an allocated budget attribute,
    relative to a `BudgetedSet`

    Parameters
    ----------
    weight : float or str or None
        Initial weight. Strings are resolved via ``IMPORTANCE_MATCH``.
        ``None`` defaults to ``DEFAULT_WEIGHT``.
    """

    def __init__(self, budget_set: BudgetedSet, name: str, weight=None) -> None:
        WeightedItem.__init__(self, weight)
        if name in budget_set.members:
            warn(f"{name} is already in the BudgetedSet {budget_set.name}. Overwriting")
        self.name = name
        budget_set.members[name] = self
        self.budget_set = budget_set

    @property
    def allocated_budget(self) -> float:
        """Direct weight of this item, in [0, 1]."""
        return self.weight / self.budget_set.total_budget
