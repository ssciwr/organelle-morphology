# The list of registered organelle (factory) subclasses
organelles = {}
organelle_factories = {}


def organelle_types() -> list[str]:
    """The list of organelles currently implemented.

    The strings used here to encode the organelles are expected in
    various APIs when refering to a specific organelle.
    """
    return organelles.keys()


class Organelle:
    def __init__(self, source, source_label: int, organelle_id: str):
        """The organelle base class implementing generic geometric properties.

        Note that instances of Organelle typically are not instantiated directly,
        but through the corresponding subclass of OrganelleFactory.

        :param source:
            The data source instance holding the data for this organelle.
        :type source: organelle_morphology.source.DataSource

        :param source_label:
            The label used in the original data to identify this organelle.

        :param organelle_id:
            The string ID that is used to refer to this organelle.
        """
        self.source = source
        self.source_label = source_label
        self.organelle_id = organelle_id

    def __init_subclass__(cls, name=None):
        """Register a given subclass in the global dictionary 'organelles'"""
        if name is not None:
            organelles[name] = cls


class OrganelleFactory:
    """A factory class for organelles.

    This is used to construct all instances of organelles.
    """

    def __init_subclass__(cls, name: str = None):
        """Register a given subclass in the global dictionary 'organelle_factories'"""
        if name is not None:
            organelle_factories[name] = cls
            cls._name = name

    @classmethod
    def construct(cls, source: str, labels: list[int] = []):
        """A trivial factory for organelle instances.

        It constructs an instance per label. The construction process for each
        organelle is independent of all others. Other organelles can subclass
        this to implement a construction process that e.g. takes into account
        all organelle instances.
        """
        for label in labels:
            yield organelles[cls._name](
                source=source,
                source_label=label,
                organelle_id=f"{cls._name}_{str(label).zfill(4)}",
            )


class Mitochondrium(Organelle, name="mito"):
    pass


class EndoplasmicReticulum(Organelle, name="er"):
    pass


class EndoplasmicReticulumFactory(OrganelleFactory, name="er"):
    pass
