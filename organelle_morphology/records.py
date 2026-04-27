from __future__ import annotations

from abc import ABC
from dataclasses import fields
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar
from collections import defaultdict
from typing import List, Type
from organelle_morphology.util import numpy_to_python

import logging
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from organelle_morphology.project import Project


def construct_path(loader, node):
    return Path(loader.construct_scalar(node))


def represent_path(dumper: yaml.Dumper, path: Path):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:python/object/apply:Path", str(path)
    )


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:Path",
    construct_path,
    Loader=yaml.SafeLoader,
)

yaml.add_representer(PosixPath, represent_path, Dumper=yaml.SafeDumper)
yaml.add_representer(WindowsPath, represent_path, Dumper=yaml.SafeDumper)


def represent_property_block(dumper, obj):
    """Represent PropertyBlock objects with their class tag"""
    # Get the class name and module for YAML reconstruction
    class_name = obj.__class__.__name__
    module_name = obj.__class__.__module__

    # Create the tag for reconstruction
    tag = f"tag:yaml.org,2002:python/object/apply:{module_name}.{class_name}"

    # Represent the object as a sequence of key-value pairs
    data = numpy_to_python(obj.to_dict())
    return dumper.represent_mapping(tag, data)


class PropertyBlock(ABC):
    """Base class for (Meta)Data containers"""

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @staticmethod
    def yaml_constructor(loader, node, properties_class):
        """Constructor for PropertyBlock objects"""
        values = loader.construct_pairs(node)
        return properties_class(**dict(values))

    def yaml_representor(self):
        yaml.add_representer(
            self.__class__, represent_property_block, Dumper=yaml.SafeDumper
        )


AnyPropertyBlock = TypeVar("AnyPropertyBlock", bound=PropertyBlock)


class Record:
    """Record class to collect statistical data together with metadata
    throughout the project.

    Attributes:
        data: Dataclass holding the data
        meta: Dataclass holding the metadata
        name: name of data
    """

    def __init__(self, data: AnyPropertyBlock, meta: AnyPropertyBlock):
        """
        Args:
            data: dataclass inheriting from PropertyBlock, contains the data
            meta: dataclass inheriting from PropertyBlock, contains the metadata
        """
        self.data = data
        self.meta = meta
        self.name = type(data).__name__

    def to_dict(self):
        return {"data": self.data, "meta": self.meta, "name": self.name}

    def save_yaml(self, filename: Path):
        to_save = self.to_dict()
        self.data.yaml_representor()
        self.meta.yaml_representor()
        with open(filename, "w") as f:
            yaml.safe_dump(to_save, f)

    @classmethod
    def from_yaml(cls, filename: Path):
        with open(filename) as f:
            yaml_dict = yaml.safe_load(f)
        if not all([k in yaml_dict for k in ["data", "meta", "name"]]):
            raise ValueError(f"YAML file could not be parsed: {filename}")

        return cls(data=yaml_dict["data"], meta=yaml_dict["meta"])

    def __eq__(self, value: object, /) -> bool:
        if data := getattr(value, "data"):
            if data != self.data:
                return False
        if meta := getattr(value, "meta"):
            if meta != self.meta:
                return False
        if name := getattr(value, "name"):
            if name != self.name:
                return False
        return True


class RecordRegistry:
    """
    Central registry for managing all statistical and morphological Records in a Project.
    Provides fast lookups by record type and organelle ID.
    """

    def __init__(self, project: "Project"):
        self.project = project
        self.logger = logging.getLogger(__name__)

        # Internal storage
        self._all_records: List[Record] = []
        self._records_by_type: dict[str, List[Record]] = defaultdict(list)
        self._records_by_id: dict[str, List[Record]] = defaultdict(list)

    def add(self, record: Record) -> None:
        """
        Add a new Record to the registry and index it.
        """
        self._all_records.append(record)

        # Index by the class name of the data (e.g., "McsProperties", "ProfileData")
        record_type = type(record.data).__name__
        self._records_by_type[record_type].append(record)

        # Index by organelle ID if available in the metadata
        if hasattr(record.meta, "organelle_id") and record.meta.organelle_id:
            self._records_by_id[record.meta.organelle_id].append(record)

    def get_all(self) -> List[Record]:
        """Return all stored records."""
        return self._all_records

    def get_by_type(self, property_type: Type[PropertyBlock] | str) -> List[Record]:
        if isinstance(property_type, str):
            return self._records_by_type.get(property_type, [])
        return [r for r in self._all_records if isinstance(r.data, property_type)]

    def get_by_organelle(self, organelle_id: str) -> List[Record]:
        """Retrieve all records associated with a specific organelle ID."""
        return self._records_by_id.get(organelle_id, [])

    def clear(self) -> None:
        """Wipes all records from the registry."""
        self._all_records.clear()
        self._records_by_type.clear()
        self._records_by_id.clear()
        self.logger.debug("Record registry cleared.")

    def summary(self) -> dict[str, int]:
        """
        Generate meta-statistics about the currently stored records.
        Replaces the old Project.get_stat_stats() method.

        Returns:
            dict: Number of each Record type available.
        """
        desc = "Collected records in registry:\n"
        stat_count = defaultdict(int)
        desc += f"{'Total number of records':<25}: {len(self._all_records)}\n"

        for record in self._all_records:
            stat_count[record.name] += 1

        for stat_name, count in stat_count.items():
            desc += f"{stat_name:<23}: {count}\n"

        self.logger.info(desc)
        return dict(stat_count)
