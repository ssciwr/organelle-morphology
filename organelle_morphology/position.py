from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import rotate
from organelle_morphology.analysis import Analysis
from organelle_morphology.source import DataSource
import logging
import dask.array as da

from organelle_morphology.statistics import Properties, Stats

logger = logging.getLogger(__name__)


@dataclass
class PositionMeta(Properties):
    source: str
    normalizing_sources: list[str]
    centroids: Optional[list[float]]  # per source
    dimensionality: int
    bin_resolution: tuple[float, float, float]
    marginal_axis_2d: Optional[int] = None
    axis_1d: Optional[int] = None
    rot_angle: Optional[float] = None
    rot_axes: Optional[tuple[int, int]] = None


@dataclass
class PositionProperties(Properties):
    density: list


class Position_Analysis(Analysis):
    """Analysis of organell density along an axis"""

    def __init__(self, project):
        super().__init__(project=project, property_type=PositionProperties)

    def get_dataframe(self):
        return super().get_dataframe()

    def density3D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
        normalizing_sources: Optional[list[DataSource]],
    ) -> da.Array:
        """Calculate a 3d heatmap of all organelles in the source.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometers
        """
        res = bin_resolution
        norm_sources_name = ""
        if normalizing_sources is not None:
            norm_sources_name = "_".join([s.org_name for s in normalizing_sources])
        cache_key = (
            f"density3d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_norm-{norm_sources_name}"
        )
        if cache_key not in source.cache:
            binsize = (np.array(bin_resolution) // source.resolution).astype(int)
            n_missing = source.data.shape % binsize

            data = source.data.astype(bool)
            data = da.pad(data, [(0, i) for i in n_missing])
            center = None
            if normalizing_sources is not None:
                center = np.mean(
                    [s.global_coarse_centroid for s in normalizing_sources]
                )
                logger.debug(f"Normalizing position onto centroid: {center}")
                data -= center

            logger.debug(f"Density3D calculation, binsize {binsize}")
            density = da.coarsen(
                np.mean,
                data,
                {a: b for a, b in zip((0, 1, 2), binsize)},
                trim_excess=False,
            ).compute()
            source.cache[cache_key] = density

            Stats(
                PositionProperties(density=density),
                PositionMeta(
                    source=source.org_name,
                    normalizing_sources=norm_sources_name,
                    centroids=center,
                    bin_resolution=bin_resolution,
                    dimensionality=3,
                ),
            )
            # TODO: save stats, test

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
            source.cache[cache_key] = np.mean(density, axis=marginal_axis)
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
            source.cache[cache_key] = np.mean(density2d, axis=marginal_axes[0])
        return source.cache[cache_key]
