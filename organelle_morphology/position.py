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
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


@dataclass
class PositionMeta(PropertyBlock):
    source: str
    dimensionality: int
    bin_resolution: tuple[float, float, float]
    marginal_axis_2d: Optional[int] = None
    axis_1d: Optional[int] = None
    rot_angle: Optional[float] = None
    rot_axis: Optional[int] = None
    axes_labels: Optional[list[str]] = None


@dataclass
class PositionProperties(PropertyBlock):
    density: np.ndarray


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
        return pd.DataFrame([s.meta for s in self.own_records])

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

        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=3,
            bin_resolution=bin_resolution,
            axes_labels=["x", "y", "z"],
        )
        if pos_meta not in [r.meta for r in self.own_records]:
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

            density = source.cache[cache_key]
            record = Record(
                PositionProperties(density=density),
                pos_meta,
            )
            self.project.registry.add(record)

        return source.cache[cache_key]

    def density2D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
        marginal_axis: int = 0,
        rot_angle: float = 0.0,
        rot_axis: int = 2,
    ) -> np.ndarray:
        """Calculate the density of organelles in 2d.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometer
            marginal_axis: axis which will be averaged over.
            rot_axis: Axes around which the 3d volume can be rotated
            rot_angle: Rotation angle to rotate the input image.
        """

        if marginal_axis not in [0, 1, 2]:
            raise ValueError("Marginal axis must be between 0 and 2")
        res = bin_resolution
        cache_key = (
            f"density2d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_"
            f"{marginal_axis}_{rot_angle:.2f}_{rot_axis}"
        )
        axes_labels = None
        if rot_angle == 0:
            axes_labels = [0, 1, 2]
            axes_labels.remove(marginal_axis)
            axes_names = ["x", "y", "z"]
            axes_labels = [axes_names[i] for i in axes_labels]

        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=2,
            bin_resolution=bin_resolution,
            marginal_axis_2d=marginal_axis,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            axes_labels=axes_labels,
        )
        if pos_meta not in [r.meta for r in self.own_records]:
            if cache_key not in source.cache:
                density = self.density3D(source, bin_resolution)
                rot_axes = [0, 1, 2]
                rot_axes.remove(rot_axis)
                density = rotate(density, rot_angle, rot_axes, reshape=True)
                density_2D = np.mean(density, axis=marginal_axis)
                source.cache[cache_key] = density_2D

            density_2D = source.cache[cache_key]
            record = Record(
                PositionProperties(density=density_2D),
                pos_meta,
            )
            self.project.registry.add(record)
        return source.cache[cache_key]

    def density1D(
        self,
        source: DataSource,
        bin_resolution: tuple[float, float, float],
        axis: int = 0,
        rot_angle: float = 0.0,
        rot_axis: int = 2,
    ):
        """Calculate the density of organelles along an axis.

        Args:
            source: The source containing the organelle of interest
            bin_resolution: binning resolution in micrometers
            axis: axis along which the density will be calculated
            rot_angle: Rotation angle to rotate the input image.
            rot_axis: Axes around which the 3d volume can be rotated
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be between 0 and 2")
        res = bin_resolution
        cache_key = f"density1d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_{axis}_{rot_angle:.2f}_{rot_axis}"

        axes_labels = None
        if rot_angle == 0:
            axes_names = ["x", "y", "z"]
            axes_labels = [axes_names[axis]]
        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=1,
            bin_resolution=bin_resolution,
            axis_1d=axis,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            axes_labels=axes_labels,
        )

        if pos_meta not in [r.meta for r in self.own_records]:
            if cache_key not in source.cache:
                marginal_axes = [0, 1, 2]
                marginal_axes.remove(axis)
                density2d = self.density2D(
                    source=source,
                    bin_resolution=bin_resolution,
                    marginal_axis=marginal_axes[1],
                    rot_angle=rot_angle,
                    rot_axis=rot_axis,
                )
                density_1D = np.mean(density2d, axis=marginal_axes[0])
                source.cache[cache_key] = density_1D

            density_1D = source.cache[cache_key]
            record = Record(
                PositionProperties(density=density_1D),
                pos_meta,
            )
            self.project.registry.add(record)
        return source.cache[cache_key]

    def plot_density(self, record: Record, ax=None):
        """Plot density from a Position_Analysis record.

        Args:
            record: Record containing PositionProperties and PositionMeta
            ax: matplotlib axes object to plot on (optional, creates new figure if None)

        Returns:
            matplotlib axes object with the plot
        """
        if ax is None:
            _, ax = plt.subplots()
        else:
            _ = ax.figure

        density = record.data.density

        if len(density.shape) == 3:
            if not (isinstance(ax, plt.Axes) and ax.name == "3d"):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            nonzero_mask = density != 0
            if not nonzero_mask.any():
                ax.set_title(f"3D Density Plot - {record.meta.source}")
                return ax

            coords = np.where(nonzero_mask)
            x, y, z = coords[2], coords[1], coords[0]
            densities = density[nonzero_mask]

            ax.scatter(x, y, z, c=densities, cmap="viridis", s=20)
            ax.set_title(f"3D Density Plot - {record.meta.source}")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(labels[2])
                ax.set_ylabel(labels[1])
                ax.set_zlabel(labels[0])
            return ax

        elif len(density.shape) == 2:
            ax.imshow(density)
            ax.set_title(f"2D Density Plot - {record.meta.source}")
            ax.axis("auto")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(labels[1])
                ax.set_ylabel(labels[0])
            return ax

        elif len(density.shape) == 1:
            ax.plot(density)
            ax.set_title(f"1D Density Plot - {record.meta.source}")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(labels[0])
            ax.set_ylabel("Mean density")
            return ax
        else:
            raise ValueError(f"Unsupported density shape: {density.shape}")

    def plot_multiple_densities(
        self,
        records: list[Record],
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        sharex=False,
        sharey=False,
    ):
        """Create subplots for multiple density records.

        Args:
            records: List of Record objects to plot
            nrows: Number of rows for subplots
            ncols: Number of columns for subplots

        Returns:
            matplotlib figure and axes objects
        """
        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(len(records))))
            nrows = int(np.ceil(len(records) / ncols))
        elif nrows is None and ncols is not None:
            nrows = int(np.ceil(len(records) / ncols))
        elif ncols is None and nrows is not None:
            ncols = int(np.ceil(len(records) / nrows))
        assert ncols is not None and nrows is not None

        if ncols * nrows < len(records):
            logger.warning(
                "Not enough axes for all records, "
                f"plotting only {ncols * nrows} first records"
            )

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3 * nrows),
            sharex=sharex,
            sharey=sharey,
        )

        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, record in enumerate(records):
            if i < len(axes):
                if record.meta.dimensionality == 3:
                    # For 3D plots, ensure we have 3D axes
                    if not (isinstance(axes[i], plt.Axes) and axes[i].name == "3d"):
                        # Replace with 3D axis
                        axes[i].remove()
                        axes[i] = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
                self.plot_density(record, axes[i])

        fig.tight_layout()
        return fig, axes
