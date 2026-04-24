from __future__ import annotations

from abc import ABC
from dataclasses import fields
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar

import yaml


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


def represent_properties(dumper, obj):
    """Represent PropertyBlock objects with their class tag"""
    # Get the class name and module for YAML reconstruction
    class_name = obj.__class__.__name__
    module_name = obj.__class__.__module__

    # Create the tag for reconstruction
    tag = f"tag:yaml.org,2002:python/object/apply:{module_name}.{class_name}"

    # Represent the object as a sequence of key-value pairs
    data = obj.to_dict()
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
            self.__class__, represent_properties, Dumper=yaml.SafeDumper
        )


AnyProperty = TypeVar("AnyProperty", bound=PropertyBlock)


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
