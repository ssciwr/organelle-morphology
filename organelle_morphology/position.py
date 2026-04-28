from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from organelle_morphology.analysis import Analysis
from organelle_morphology.source import DataSource
import logging
import dask.array as da

from organelle_morphology.records import PropertyBlock, Record
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class PositionMeta(PropertyBlock):
    source: str
    dimensionality: int
    bin_resolution: tuple[float, float, float]
    marginal_axis_2d: Optional[int] = None
    axis_1d: Optional[int] = None
    rot_angle: Optional[float] = None
    rot_axes: Optional[tuple[int, int]] = None


@dataclass
class PositionProperties(PropertyBlock):
    density: np.ndarray

    def get_plot(self):
        if len(self.density.shape) == 3:
            raise NotImplementedError()
        elif len(self.density.shape) == 2:
            return plt.imshow(self.density)
        elif len(self.density.shape) == 1:
            return plt.plot(self.density)

    def plot(self):
        self.get_plot()
        plt.show()


class Position_Analysis(Analysis):
    """Analysis of organell density along an axis"""

    def __init__(self, project):
        super().__init__(project=project, property_type=PositionProperties)

    def get_dataframe(self):
        """Dataframe describing available data

        Dataframe is build from the metadata of all Position_Analysis Stats

        Returns:
            Dataframe of the metadata of all collected position analysis calculations
        """
        self.update_project_stats()
        return pd.DataFrame([s.meta for s in self.own_stats])

    def density3D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
    ) -> da.Array:
        """Calculate a 3d heatmap of all organelles in the source.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometers
        """
        res = bin_resolution
        cache_key = f"density3d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}"
        if cache_key not in source.cache:
            binsize = (np.array(bin_resolution) // source.resolution).astype(int)
            n_missing = (binsize - (source.data.shape % binsize)) % binsize

            data = source.data.astype(bool)
            data = da.pad(data, [(0, i) for i in n_missing])

            logger.debug(f"Density3D calculation, binsize {binsize}")
            density = da.coarsen(
                np.mean,
                data,
                {a: b for a, b in zip((0, 1, 2), binsize)},
                trim_excess=False,
            ).compute()
            source.cache[cache_key] = density

            record = Record(
                PositionProperties(density=density),
                PositionMeta(
                    source=source.org_name,
                    dimensionality=3,
                    bin_resolution=bin_resolution,
                ),
            )
            self.project.registry.add(record)

        return source.cache[cache_key]

    def density2D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
        marginal_axis: int = 0,
        rot_angle: float = 0.0,
        rot_axes: tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        """Calculate the density of organelles in 2d.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometer
            marginal_axis: axis which will be averaged over.
            rot_axes: Two axes defining a plane in which the input image can
                be rotated.
            rot_angle: Rotation angle to rotate the input image.
        """

        if marginal_axis not in [0, 1, 2]:
            raise ValueError("Marginal axis must be between 0 and 2")
        res = bin_resolution
        cache_key = f"density2d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_{marginal_axis}_{rot_angle:.2f}_{rot_axes[0]}-{rot_axes[1]}"
        if cache_key not in source.cache:
            density = self.density3D(source, bin_resolution)
            density = rotate(density, rot_angle, rot_axes, reshape=True)
            density_2D = np.mean(density, axis=marginal_axis)
            source.cache[cache_key] = density_2D

            record = Record(
                PositionProperties(density=density_2D),
                PositionMeta(
                    source=source.org_name,
                    dimensionality=2,
                    bin_resolution=bin_resolution,
                    marginal_axis_2d=marginal_axis,
                    rot_angle=rot_angle,
                    rot_axes=rot_axes,
                ),
            )
            self.project.registry.add(record)
        return source.cache[cache_key]

    def density1D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
        axis: int = 0,
        rot_angle: float = 0.0,
        rot_axes: tuple[int, int] = (0, 1),
    ):
        """Calculate the density of organelles along an axis.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometers
            axis: axis along which the density will be calculated
            rot_angle: Rotation angle to rotate the input image.
            rot_axes: Two axes defining a plane in which the input image can
                be rotated.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be between 0 and 2")
        res = bin_resolution
        cache_key = f"density1d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_{axis}_{rot_angle:.2f}_{rot_axes[0]}-{rot_axes[1]}"
        if cache_key not in source.cache:
            marginal_axes = [0, 1, 2]
            marginal_axes.remove(axis)
            density2d = self.density2D(
                source=source,
                bin_resolution=bin_resolution,
                marginal_axis=marginal_axes[1],
                rot_angle=rot_angle,
                rot_axes=rot_axes,
            )
            density_1D = np.mean(density2d, axis=marginal_axes[0])
            source.cache[cache_key] = density_1D

            record = Record(
                PositionProperties(density=density_1D),
                PositionMeta(
                    source=source.org_name,
                    dimensionality=2,
                    bin_resolution=bin_resolution,
                    axis_1d=axis,
                    rot_angle=rot_angle,
                    rot_axes=rot_axes,
                ),
            )
            self.project.registry.add(record)
        return source.cache[cache_key]
