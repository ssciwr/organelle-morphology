import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


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


def stats_contacts_a_b(df: DataFrame, label: str):
    df_sel = df.loc[label, "n_contacts"]
    if partners := label.split(",")[1].split("-"):
        o1, o2 = partners
    else:
        raise ValueError("mcs calculation must be between two organelle types!")

    n_contacts = {}
    n_orgs = {}
    n_in_contact = {}
    percent = {}
    for org, group in df_sel.groupby(lambda i: i.split("_")[0]):
        n_contacts[org] = group.sum()
        n_orgs[org] = group.count()
        n_in_contact[org] = group.loc[group > 0].count()
        percent[org] = n_in_contact[org] / n_orgs[org]

    stats = {
        "n_contacts": n_contacts,
        "n_organelles": n_orgs,
        "n_in_contact": n_in_contact,
        "percent": percent,
    }
    return stats
