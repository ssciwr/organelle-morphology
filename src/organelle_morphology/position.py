import logging
from dataclasses import dataclass
from typing import Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate

from organelle_morphology.analysis import Analysis
from organelle_morphology.records import PropertyBlock, Record
from organelle_morphology.source import DataSource

logger = logging.getLogger(__name__)


def _density_bin_sizes(
    source: DataSource, bin_resolution: tuple[float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Per-axis density bin sizes in voxels and bin widths in micrometers.

    A bin spans floor(bin_resolution / voxel resolution) voxels per axis,
    at least one voxel. Both are in array order (x, y, z = first, second,
    third data dimension), so that width[i] is the size in micrometers of
    one bin along array axis i.

    Args:
        source: The source the density is calculated on
        bin_resolution: Requested binning resolution in micrometers

    Returns:
        Binsize as an int array (for array arithmetic) and the bin width
        in micrometers as a tuple of floats
    """
    resolution = np.asarray(source.resolution)
    binsize = ((np.array(bin_resolution) + 1e-9) // resolution).astype(int)
    if (binsize < 1).any():
        logger.warning(
            "Requested bin resolution smaller than the voxel resolution "
            f"{tuple(resolution)}, using one voxel per bin"
        )
        binsize = np.maximum(binsize, 1)
    bin_width = tuple(float(w) for w in binsize * resolution)
    return binsize, bin_width


def _set_volume_limits(
    ax, offset: np.ndarray, widths: tuple[float, ...], shape: tuple[int, ...]
) -> None:
    """Limit a 3d density plot to the full binned volume.

    The volume starts at the lower clipping corner of the source (zero
    without clipping), so not only the filled bins are shown.

    Args:
        ax: 3d matplotlib axes to limit
        offset: Lower clipping corner in micrometers, per array axis
        widths: Width of one density bin in micrometers, per array axis
        shape: Shape of the density array
    """
    ax.set_xlim(offset[0], offset[0] + shape[0] * widths[0])
    ax.set_ylim(offset[1], offset[1] + shape[1] * widths[1])
    ax.set_zlim(offset[2], offset[2] + shape[2] * widths[2])


@dataclass(frozen=True)
class PositionMeta(PropertyBlock):
    """Metadata of one Position_Analysis record.

    All axis tuples are in array order (x, y, z = first, second, third
    data dimension) and all sizes in micrometers.
    """

    source: str
    dimensionality: int
    bin_resolution: tuple[float, float, float]
    # actual width of one density bin in micrometers per array axis;
    # None for records saved before this field existed
    bin_width: Optional[tuple[float, float, float]] = None
    marginal_axis_2d: Optional[int] = None
    axis_1d: Optional[int] = None
    rot_angle: Optional[float] = None
    rot_axis: Optional[int] = None
    axes_labels: Optional[tuple[str, ...]] = None


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
        source: DataSource | str,
        bin_resolution: tuple[float, float, float],
    ) -> np.ndarray:
        """Calculate a 3d heatmap of all organelles in the source.

        Args:
            source: The source containing the organelle of interest,
                or the name of the respective organelle defined while loading
            bin_resolution: binning resolution in micrometers, in array
                order (x, y, z = first, second, third data dimension)
        """
        res = bin_resolution
        cache_key = f"density3d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}"

        if isinstance(source, str):
            source: DataSource = {s.org_name: s for s in self.project.sources.values()}[
                source
            ]

        binsize, bin_width = _density_bin_sizes(source, bin_resolution)

        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=3,
            bin_resolution=bin_resolution,
            bin_width=bin_width,
            axes_labels=("x", "y", "z"),
        )

        if cache_key not in source.cache:
            n_missing = (binsize - (source.data.shape % binsize)) % binsize

            data = source.data.astype(bool)
            data = da.pad(data, [(0, i) for i in n_missing])

            logger.debug(f"Density3D calculation, binsize {binsize}")
            source.cache[cache_key] = da.coarsen(
                np.mean,
                data,
                {a: b for a, b in zip((0, 1, 2), binsize)},
                trim_excess=False,
            ).compute()

        density = source.cache[cache_key]

        if pos_meta not in [r.meta for r in self.own_records]:
            record = Record(
                PositionProperties(density=density), pos_meta, project=self.project
            )
            self.project.registry.add(record)
        return density

    def density2D(
        self,
        source: DataSource | str,
        bin_resolution: tuple[float, float, float],
        marginal_axis: int = 0,
        rot_angle: float = 0.0,
        rot_axis: int = 2,
    ) -> np.ndarray:
        """Calculate the density of organelles in 2d.

        Args:
            source: The source containing the organelle of interest,
                or the name of the respective organelle defined while loading
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
        if isinstance(source, str):
            source: DataSource = {s.org_name: s for s in self.project.sources.values()}[
                source
            ]
        # rotation keeps the per-axis voxel size, so the bin widths of the
        # 3d density describe the axes of the 2d result as well
        _, bin_width = _density_bin_sizes(source, bin_resolution)
        axes_labels = None
        if rot_angle == 0:
            axes_labels = [0, 1, 2]
            axes_labels.remove(marginal_axis)
            axes_names = ["x", "y", "z"]
            axes_labels = tuple([axes_names[i] for i in axes_labels])

        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=2,
            bin_resolution=bin_resolution,
            bin_width=bin_width,
            marginal_axis_2d=marginal_axis,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            axes_labels=axes_labels,
        )
        density = self.density3D(source, bin_resolution)
        if cache_key not in source.cache:
            rot_axes = [0, 1, 2]
            rot_axes.remove(rot_axis)
            density = rotate(density, rot_angle, rot_axes, reshape=True)
            source.cache[cache_key] = np.mean(density, axis=marginal_axis)

        density_2D = source.cache[cache_key]

        if pos_meta not in [r.meta for r in self.own_records]:
            record = Record(
                PositionProperties(density=density_2D), pos_meta, project=self.project
            )
            self.project.registry.add(record)
        return density_2D

    def density1D(
        self,
        source: DataSource | str,
        bin_resolution: tuple[float, float, float],
        axis: int = 0,
        rot_angle: float = 0.0,
        rot_axis: int = 2,
    ):
        """Calculate the density of organelles along an axis.

        Args:
            source: The source containing the organelle of interest,
                or the name of the respective organelle defined while loading
            bin_resolution: binning resolution in micrometers
            axis: axis along which the density will be calculated
            rot_angle: Rotation angle to rotate the input image.
            rot_axis: Axes around which the 3d volume can be rotated
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be between 0 and 2")
        res = bin_resolution
        cache_key = f"density1d_{res[0]:.6f}-{res[1]:.6f}-{res[2]:.6f}_{axis}_{rot_angle:.2f}_{rot_axis}"

        if isinstance(source, str):
            source: DataSource = {s.org_name: s for s in self.project.sources.values()}[
                source
            ]

        _, bin_width = _density_bin_sizes(source, bin_resolution)
        axes_labels = None
        if rot_angle == 0:
            axes_names = ["x", "y", "z"]
            axes_labels = tuple([axes_names[axis]])
        pos_meta = PositionMeta(
            source=source.org_name,
            dimensionality=1,
            bin_resolution=bin_resolution,
            bin_width=bin_width,
            axis_1d=axis,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            axes_labels=axes_labels,
        )

        # restore also 2d/3d from cache
        marginal_axes = [0, 1, 2]
        marginal_axes.remove(axis)
        density2d = self.density2D(
            source=source,
            bin_resolution=bin_resolution,
            marginal_axis=marginal_axes[1],
            rot_angle=rot_angle,
            rot_axis=rot_axis,
        )
        if cache_key not in source.cache:
            source.cache[cache_key] = np.mean(density2d, axis=marginal_axes[0])

        density_1D = source.cache[cache_key]
        if pos_meta not in [r.meta for r in self.own_records]:
            record = Record(
                PositionProperties(density=density_1D), pos_meta, project=self.project
            )
            self.project.registry.add(record)
        return density_1D

    def _clipping_offset(self, record: Record) -> np.ndarray:
        """Lower clipping corner of the record's source in micrometers.

        The density is calculated on the clipped volume, so bin 0 of the
        density sits at the lower clipping corner. Plot positions are
        offset by this corner to describe the position within the full
        data extent, like the show scene. Zero without clipping, or if the
        record's source is not loaded in the project.

        Note: the offset is read from the source at plot time; plotting a
        record after changing the project clipping reflects the new
        clipping, not the one the record was calculated with.

        Args:
            record: Record containing PositionProperties and PositionMeta

        Returns:
            Lower clipping corner in micrometers, per array axis
        """
        source = {s.org_name: s for s in self.project.sources.values()}.get(
            record.meta.source
        )
        if source is None:
            logger.warning(
                f"Source {record.meta.source} of the record is not loaded in "
                "the project, plotting without clipping offset"
            )
            return np.zeros(3)
        return np.asarray(source.clipping_corners[0], dtype=float)

    def plot_density(self, record: Record, ax=None):
        """Plot density from a Position_Analysis record.

        All plot axes are in micrometers within the full data extent: a
        bin with index i along an array axis is drawn at i * bin_width,
        using the actual bin width recorded in the meta. Positions are
        offset by the lower clipping corner of the record's source, so
        with clipping the lower clipping corner is not at zero — the same
        coordinate frame as the show view. For records saved before
        bin_width existed, the requested bin_resolution is used instead.
        Axis names follow the array order convention (x, y, z = first,
        second, third data dimension), like the clipping and show
        settings.

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

        # width of one density bin in micrometers, per array axis
        widths = record.meta.bin_width or tuple(record.meta.bin_resolution)
        # lower clipping corner of the source, per array axis: bin 0 of
        # the density sits at this corner of the full data extent
        offset = self._clipping_offset(record)

        if len(density.shape) == 3:
            if not (isinstance(ax, plt.Axes) and ax.name == "3d"):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            ax.set_title(f"3D Density Plot - {record.meta.source}")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(f"{labels[0]} [$\\mu$m]")
                ax.set_ylabel(f"{labels[1]} [$\\mu$m]")
                ax.set_zlabel(f"{labels[2]} [$\\mu$m]")

            nonzero_mask = density != 0
            if not nonzero_mask.any():
                _set_volume_limits(ax, offset, widths, density.shape)
                return ax

            coords = np.where(nonzero_mask)
            # array order: x, y, z = first, second, third dimension,
            # positions scaled to micrometers, offset by the lower
            # clipping corner
            x, y, z = (
                offset[0] + coords[0] * widths[0],
                offset[1] + coords[1] * widths[1],
                offset[2] + coords[2] * widths[2],
            )
            densities = density[nonzero_mask]

            ax.scatter(x, y, z, c=densities, cmap="viridis", s=20)

            # show the full binned volume, not only the filled bins,
            # offset by the lower clipping corner
            _set_volume_limits(ax, offset, widths, density.shape)
            return ax

        elif len(density.shape) == 2:
            # the first array axis of the density becomes the horizontal
            # plot axis, the second the vertical one, both in micrometers
            # within the full data extent
            marginal_axis = record.meta.marginal_axis_2d
            widths_2d = [w for i, w in enumerate(widths) if i != marginal_axis]
            offset_2d = [o for i, o in enumerate(offset) if i != marginal_axis]
            extent = (
                offset_2d[0],
                offset_2d[0] + density.shape[0] * widths_2d[0],
                offset_2d[1],
                offset_2d[1] + density.shape[1] * widths_2d[1],
            )
            ax.imshow(density.T, origin="lower", extent=extent)
            ax.set_title(f"2D Density Plot - {record.meta.source}")
            ax.axis("auto")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(f"{labels[0]} [$\\mu$m]")
                ax.set_ylabel(f"{labels[1]} [$\\mu$m]")
            return ax

        elif len(density.shape) == 1:
            bin_width = widths[record.meta.axis_1d]
            ax.plot(
                offset[record.meta.axis_1d] + np.arange(density.shape[0]) * bin_width,
                density,
            )
            ax.set_title(f"1D Density Plot - {record.meta.source}")
            if (labels := record.meta.axes_labels) is not None:
                ax.set_xlabel(f"{labels[0]} [$\\mu$m]")
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
        if not records:
            return plt.subplots()
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
