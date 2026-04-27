from __future__ import annotations

from abc import ABC
from dataclasses import astuple, fields
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar
import numpy as np
import logging

import yaml

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


def represent_properties(dumper, obj):
    """Represent Properties objects with their class tag"""
    # Get the class name and module for YAML reconstruction
    class_name = obj.__class__.__name__
    module_name = obj.__class__.__module__

    # Create the tag for reconstruction
    tag = f"tag:yaml.org,2002:python/object/apply:{module_name}.{class_name}"

    # Represent the object as a sequence of key-value pairs
    data = obj.to_dict()
    return dumper.represent_mapping(tag, data)


class Properties(ABC):
    """Base class for (Meta)Data containers

    Should be subclassed into a dataclass.
    If the dataclass should hold np arrays, use
    `@dataclass(eq=False)`
    """

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @staticmethod
    def yaml_constructor(loader, node, properties_class):
        """Constructor for Properties objects"""
        values = loader.construct_pairs(node)
        return properties_class(**dict(values))

    def yaml_representor(self):
        yaml.add_representer(
            self.__class__, represent_properties, Dumper=yaml.SafeDumper
        )

    def __eq__(self, other):
        return dc_eq(self, other)


AnyProperty = TypeVar("AnyProperty", bound=Properties)


class Stats:
    """Stats class to collect statistical data together with metadata
    throughout the project.

    Attributes:
        data: Dataclass holding the data
        meta: Dataclass holding the metadata
        name: name of data
    """

    def __init__(self, data: AnyProperty, meta: AnyProperty):
        """
        Args:
            data: dataclass inheriting from Properties, contains the data
            meta: dataclass inheriting from Properties, contains the metadata
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

    @classmethod
    def from_yaml_old(cls, filename: Path):
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
