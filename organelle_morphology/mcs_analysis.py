"""Temporary MCS analysis

Should be merged with new statistics.py
"""

from dask.base import compute
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import logging

import organelle_morphology


logger = logging.getLogger(__name__)


def plot_mcs_hist(df: DataFrame, label: str, measurement: str):
    if measurement not in df.columns:
        raise ValueError(
            f"Measurement {measurement} not available!\nMust be one of {df.columns}"
        )
    if label not in df.index:
        raise ValueError(f"MCS {label} not in df!")

    df_sel = df.loc[label, measurement]
    data_list = []
    group_list = []

    for org, group in df_sel.groupby(lambda i: i.split("_")[0]):
        data_list.extend(group.values)
        group_list.extend([org] * len(group))

    # Create stacked histogram with aligned bins
    plot_df = DataFrame({"value": data_list, "Organelle": group_list})
    sns.histplot(
        data=plot_df,
        x="value",
        hue="Organelle",
        multiple="stack",
        stat="count",
        legend=True,
    )
    plt.xlabel(measurement)


def stats_contacts_a_b(df: DataFrame, mcs_label: str, n_orgs: dict[str, int]):
    logger.info(f"Calculating Stats for {mcs_label}")
    df_contacts = df.loc[mcs_label, "n_contacts"]
    if partners := mcs_label.split(",")[1].split("-"):
        o1, o2 = partners
    else:
        raise ValueError("mcs calculation must be between two organelle types!")

    n_contacts = {}
    n_in_contact = {}
    percent_in_contact = {}
    contacts_per_area = {}
    contacts_per_volume = {}

    for org, group in df_contacts.groupby(lambda i: i.split("_")[0]):
        n_contacts[org] = group.sum()
        n_in_contact[org] = group.loc[group > 0].count()
        percent_in_contact[org] = (n_in_contact[org] / n_orgs[org]) * 100
        contacts_per_area[org] = df.loc[mcs_label, "n_contacts_per_area"].mean()
        contacts_per_volume[org] = df.loc[mcs_label, "n_contacts_per_volume"].mean()

    stats = {
        "n_contacts": n_contacts,
        "n_in_contact": n_in_contact,
        "percent_in_contact": percent_in_contact,
        "contacts_per_area": contacts_per_area,
        "contacts_per_volume": contacts_per_volume,
    }
    return stats


def stats_global(project: "organelle_morphology.Project"):
    """Generate a report about global properties

    Args:
        project: The project

    Returns:
        dict with the following keys:
        n_organelles: number of organelles per organelle type
        areas: total surface area of each organelle type
        volumes: total volume of each organelle type
    """
    # Count
    n_orgs = {}
    for s in project.sources.values():
        n_orgs[s.org_name] = len(project.get_organelles(s.org_name + "*"))

    # Surface area
    areas = {}
    for s in project.sources.values():
        area_delayed = [o.mesh.area for o in project.get_organelles(s.org_name + "*")]
        areas[s.org_name] = compute(area_delayed)[0]

    # Volumes
    vols = {}
    for s in project.sources.values():
        vol_delayed = [o.mesh.volume for o in project.get_organelles(s.org_name + "*")]
        vols[s.org_name] = compute(vol_delayed)[0]

    return {
        "n_organelles": n_orgs,
        "areas": areas,
        "volumes": vols,
    }
