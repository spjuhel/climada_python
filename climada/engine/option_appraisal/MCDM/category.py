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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Optional, cast
from warnings import warn

from climada.engine.option_appraisal.MCDM.constants import DEFAULT_CATEGORY_WEIGHT
from climada.engine.option_appraisal.MCDM.weights import (
    BudgetedSet,
    BudgetWeightItem,
    WeightedItemZeroOne,
)

LOGGER = logging.getLogger(__name__)


class CriteriaCategory(BudgetWeightItem):
    """
    Represents a criteria category in a dynamic, multiple-parent hierarchy.

    This class manages the structure and relationships of criteria, allowing
    for runtime definition and retrieval of categories. It implements logic
    to check for ancestral relationships, respecting multiple parent links.

    Attributes
    ----------
    _registry : dict[str, CriteriaCategory]
        A class-level dictionary serving as a global lookup for all defined
        CriteriaCategory instances, keyed by their name.
    name : str
        The unique name of the criteria category.
    parents : set[CriteriaCategory]
        The set of direct parent categories this criteria inherits from.
    children : set[CriteriaCategory]
        The set of direct child categories that inherit from this criteria.
    """

    _registry: ClassVar[dict[str, "CriteriaCategory"]] = {}

    def __init__(
        self,
        name: str,
        category_type: str,
        weight: float | None = None,
        criteria: "Sequence[Criterion] | None" = None,
        space: "CategorySpace | None" = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initializes a new CriteriaCategory.

        Parameters
        ----------
        name : str
            The unique name of the category.
        parents : Optional[Union[CategoryLike, Sequence[CategoryLike]]], optional
            The parent criteria(s) this category inherits from. Can be a single
            name/object or a list of names/objects. By default, None.

        Raises
        ------
        ValueError
            If a category with the given name already exists in the registry.
        """
        self._space = space if space is not None else CategorySpace.get_default_space()

        if category_type not in self._space._cat_type_registry:
            self._space.add_category_type(category_type)

        if name in self._space.categories and not overwrite:
            raise ValueError(
                f"CriteriaCategory {name} is already in the space. Set overwrite=true to overwrite."
            )

        self.category_type: str = category_type
        self.criteria = criteria if criteria else []
        BudgetWeightItem.__init__(
            self, self._space.get_category_type(category_type), name, weight
        )

    @property
    def effective_weight(self):
        """Hierarchical weight: max of own weight and all ancestors' weights.

        Distinct from ``weight``, which is this category's direct assignment.
        """
        try:
            return self.allocated_budget
        except ZeroDivisionError:
            return 0

    @property
    def space(self):
        return self._space

    def __eq__(self, other: object) -> bool:
        """
        Defines equality based solely on the category name.
        """
        # 1. Check if the other object is an instance of CriteriaCategory
        if not isinstance(other, CriteriaCategory):
            return NotImplemented  # Defer to the other object's __eq__

        # 2. Compare names
        return self.name == other.name

    def __hash__(self) -> int:
        """
        Defines the hash based solely on the category name.
        Required for objects used in sets or as dictionary keys.
        """
        return hash(self.name)

    def __repr__(self, indent=0) -> str:
        indent_space = " " * indent
        return f"""{indent_space}* {self.name} :: weight(effective budget)={self.weight}({self.effective_weight}), metatype={self.category_type}"""


CategoryLike = str | CriteriaCategory


class CategoryType(BudgetedSet["CriteriaCategory"]):
    """Budgeted set whose members are criteria categories."""


class CategorySpace:
    """Manages a dedicated, isolated registry of CriteriaCategory objects."""

    _default_space: Optional["CategorySpace"] = None

    def __init__(self):
        self._cat_registry: dict[str, CriteriaCategory] = {}
        self._cat_type_registry: dict[str, CategoryType] = {}

    def __contains__(self, item: object) -> bool:
        """
        Allows checking if a category is in this space using 'category in space'.

        It checks if the item is a CriteriaCategory object and if it is
        present in this space's internal registry.
        """
        if isinstance(item, str):
            return item in self._cat_registry or item in self._cat_type_registry
        if isinstance(item, CategoryType):
            return item.name in self._cat_type_registry

        if isinstance(item, CriteriaCategory):
            return item.name in self._cat_registry

        return False

    def get_category_type(self, key):
        return self._cat_type_registry[key]

    def get_category(self, key):
        return self._cat_registry[key]

    def set_weight(self, cat, weight):
        self._cat_registry[cat].weight = weight

    def reset_weights(self, weight=0.0):
        for cat in self._cat_registry:
            self._cat_registry[cat].weight = weight

    @classmethod
    def get_default_space(cls) -> "CategorySpace":
        """Returns the default CategorySpace instance, creating it if necessary."""
        if cls._default_space is None:
            cls._default_space = CategorySpace()
        return cls._default_space

    @property
    def categories(self) -> list[CriteriaCategory]:
        return sorted(self._cat_registry.values(), key=lambda x: x.category_type)

    @property
    def categories_weights(self) -> dict[str, float]:
        return {cat.name: cat.weight for cat in self._cat_registry.values()}

    @property
    def categories_budgets(self) -> dict[str, float]:
        return {cat.name: cat.allocated_budget for cat in self._cat_registry.values()}

    def clear(self):
        self._cat_type_registry.clear()
        self._cat_registry.clear()

    def add_category_type(self, category_type):
        self._cat_type_registry[category_type] = CategoryType(category_type)

    def add_category(
        self,
        name: str,
        category_type: str,
        weight: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """Create and register a new category in this space.

        Parameters
        ----------
        name : str
        parent_cats : str or CriteriaCategory or list, optional
        category_type : str, optional
        weight : float or str, optional
        overwrite : bool, optional
        """
        self._cat_registry[name] = CriteriaCategory(
            name,
            category_type=category_type,
            weight=weight,
            overwrite=overwrite,
            space=self,
        )

    @property
    def category_types(self):
        return list(self._cat_type_registry.keys())

    def select_categories_by_type(
        self, category_types: str | Iterable[str]
    ) -> "list[CriteriaCategory]":
        if isinstance(category_types, str):
            category_types = [category_types]

        return [
            member
            for catset in self._cat_type_registry.values()
            if catset.name in category_types
            for member in catset.members.values()
        ]


class CategorizedObject:
    """
    An object that uses composition to belong to one or more CriteriaCategories.

    The object maintains a set of direct criteria links. Checks against the
    hierarchy are delegated to the CriteriaCategory system.

    Attributes
    ----------
    name : str
        The name or identifier of the object.
    categories : Sequence[CriteriaCategory]
        The set of CriteriaCategory objects this instance is directly assigned to.
    """

    def __init__(
        self,
        name: str,
        categories: (
            CategoryLike | Sequence[CategoryLike] | Sequence[CriteriaCategory] | None
        ) = None,
        space: CategorySpace | None = None,
    ) -> None:
        """
        Initializes the CategorizedObject.

        Parameters
        ----------
        name : str
            The name or identifier of the object.
        categories : Optional[Union[str, List[str]]], optional
            The name(s) of the initial categories to assign. By default, None.
        """
        self.name: str = name
        self._categories: set[CriteriaCategory] = set()
        self._space = space if space is not None else CategorySpace.get_default_space()
        if categories:
            self.add_categories(categories)

    @property
    def space(self):
        return self._space

    @property
    def categories(self):
        return self._categories

    @property
    def category_space(self):
        return self._space

    def add_categories(
        self,
        categories: CategoryLike | Sequence[CategoryLike] | Sequence[CriteriaCategory],
    ) -> None:
        """
        Adds one or more criteria categories to the object by name.

        Parameters
        ----------
        category_names : Union[str, List[str]]
            The name(s) of the criteria categories to add.

        Raises
        ------
        ValueError
            If a category name does not exist in the CriteriaCategory registry.
        """
        if not isinstance(categories, Sequence):
            categories = [categories]

        for cat_to_add in categories:
            if not isinstance(cat_to_add, CriteriaCategory):
                cat = self.category_space.get_category(cat_to_add)
                if not cat:
                    # Enforce that categories must be defined globally before being assigned to an object
                    raise ValueError(
                        f"CriteriaCategory '{cat_to_add}' is not defined. Create it first with `CriteriaCategory.add_category()`"
                    )
                self.categories.add(cat)
            else:
                self.categories.add(cat_to_add)

    def has_category(self, category_name: CategoryLike) -> bool:
        """
        Checks if the object belongs to the specified category or any of its
        subcategories in the hierarchy.

        Parameters
        ----------
        category_name : str
            The name of the criteria category to check against.

        Returns
        -------
        bool
            True if the object is directly or indirectly a member of the
            target category, False otherwise.
        """
        if isinstance(category_name, str):
            target_category = self.category_space.get(category_name)
            if not target_category:
                return False

        elif isinstance(category_name, CriteriaCategory):
            target_category = category_name
        else:
            raise TypeError(f"{category_name} is not a string or a CriteriaCategory")

        for category in self.categories:
            if category is target_category:
                return True
        return False

    def __repr__(self) -> str:
        cat_names = sorted([c.name for c in self.categories])
        return f"<CategorizedObject: {self.name} | Criteria: {', '.join(cat_names)}>"


def create_criteria_category(
    name: str,
    category_type: str,
    space: CategorySpace | None = None,
    overwrite: bool = False,
) -> CriteriaCategory:
    """
    Convenience function to simplify dynamic creation of CriteriaCategory objects.

    Parameters
    ----------
    name : str
        The unique name of the new criteria category.
    parent_names : Optional[Union[str, List[str]]], optional
        The name(s) of the parent criteria. By default, None.

    Returns
    -------
    CriteriaCategory
        The newly created criteria category object.
    """
    return CriteriaCategory(
        name,
        category_type=category_type,
        space=space,
        overwrite=overwrite,
    )
