from __future__ import annotations

from abc import ABC
from dataclasses import astuple, fields
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar
from collections import defaultdict
from typing import List, Type
import uuid
import numpy as np

import logging
import yaml
from typing import TYPE_CHECKING

from organelle_morphology.util import numpy_to_python

if TYPE_CHECKING:
    from organelle_morphology.project import Project

logger = logging.getLogger(__name__)


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


def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:
        return NotImplemented


def dc_eq(dc1, dc2) -> bool:
    """Checks if two dataclasses which hold numpy arrays are equal"""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented  # better than False
    t1 = astuple(dc1)
    t2 = astuple(dc2)
    return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


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
    """Base class for (Meta)Data containers

    Should be subclassed into a dataclass.
    If the dataclass should hold np arrays, use
    `@dataclass(eq=False)`


    For data belonging to an organelle, set the metadata field
    `organelle_id` to the organelle id.
    """

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

    def __eq__(self, other):
        return dc_eq(self, other)


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

        array_paths = []

        # Process data fields for numpy arrays
        data_fields = fields(to_save["data"])
        data_dict = {}
        for field in data_fields:
            obj = getattr(to_save["data"], field.name)
            if isinstance(obj, np.ndarray):
                # Save array to npz file
                array_filename = filename.with_name(f"{filename.stem}_{field.name}.npz")
                np.savez_compressed(array_filename, array=obj)
                data_dict[field.name] = array_filename
                array_paths.append(array_filename)
            else:
                data_dict[field.name] = obj

        data_class = to_save["data"].__class__
        processed_data = data_class(**data_dict)

        processed_dict = {
            "data": processed_data,
            "meta": self.meta,
            "name": self.name,
        }

        processed_data.yaml_representor()
        self.meta.yaml_representor()

        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "w") as f:
            yaml.safe_dump(processed_dict, f)

    @classmethod
    def from_yaml(cls, filename: Path):
        with open(filename) as f:
            yaml_dict = yaml.safe_load(f)

        # Restore numpy arrays from files
        # Process data fields
        if hasattr(yaml_dict["data"], "__dict__"):
            data_fields = fields(yaml_dict["data"])
            for field in data_fields:
                obj = getattr(yaml_dict["data"], field.name)
                if isinstance(obj, Path) and obj.suffix == ".npz":
                    if not obj.exists():
                        obj = filename.parent / obj.name
                    try:
                        data = np.load(obj)
                        restored_array = data["array"]
                        setattr(yaml_dict["data"], field.name, restored_array)
                    except Exception as e:
                        logger.warning(f"Restoring {obj} failed!\n{e}")
                        pass  # Keep original value if restoration fails

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

        # Index by the class name of the data (e.g., "McsData", "ProfileData")
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
        Replaces the old Project.get_record_stats() method.

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

    def save_all_to_yaml(self) -> None:
        """
        Saves all records in the registry as YAML files.
        """
        for rec in self._all_records:
            dir: Path = self.project.path / "analysis" / rec.name
            dir.mkdir(exist_ok=True, parents=True)
            rec.save_yaml(dir / f"{uuid.uuid4()}.yaml")

        self.logger.info(f"Saved {len(self._all_records)} records in project directory")

    def load_all_from_yaml(self) -> None:
        """
        Loads records from a YAML file and adds them to the registry.
        """

        files = list((self.project.path / "analysis").rglob("*.yaml"))

        logger.info(f"Found {len(files)} record files.")
        count = 0
        for file in files:
            try:
                self.add(Record.from_yaml(file))
                count += 1
            except ValueError as e:
                logger.warning(e)
        self.logger.info(f"Successfully loaded {count} records")
