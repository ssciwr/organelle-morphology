import pandas as pd
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

from organelle_morphology.analysis import Mcs_Analysis
from organelle_morphology.mcs_analysis import (
    plot_mcs_hist,
    stats_contacts_a_b,
    stats_global,
)
from organelle_morphology.organelle import McsProperties


def write_mcs_df_test_data(df):
    df.to_csv(Path(__file__).parent / "assets" / "mcs_df.csv")


@pytest.fixture
def mcs_df():
    df = pd.read_csv(Path(__file__).parent / "assets" / "mcs_df.csv")
    df = df.set_index(["mcs_label", "organelle"])
    return df


def test_plot_mcs_hist(mcs_df):
    fig = plt.figure()
    plot_mcs_hist(mcs_df, label="0.1-0.3,-", measurement="n_contacts")
    assert len(fig.axes) == 1
    # plt.show()


def test_stats_contacts_a_b(mcs_df):
    n_orgs = {"er": 1000, "mito": 50}
    stats = stats_contacts_a_b(mcs_df, mcs_label="0.0-0.05,mito-er", n_orgs=n_orgs)
    for d in stats.values():
        assert "er" in d
        assert "mito" in d


def test_stats_global(project_with_sources):
    stats = stats_global(project_with_sources)
    for d in stats.values():
        assert "mito" in d
    assert stats["n_organelles"]["mito"] == 19


@pytest.mark.parametrize("rep", range(2))
def test_mcs(project_with_sources, rep):
    if len(ps := list(project_with_sources.path.glob("cache*"))):
        raise RuntimeError(f"Existing caches found!!\n{ps}")
    project_with_sources.search_mcs(10)

    analysis = Mcs_Analysis(project_with_sources, McsProperties)
    props = analysis.get_mcs_properties()
    assert props.shape == (10, 10)

    overview = analysis.get_mcs_overview()
    assert overview.shape == (10, 1)
