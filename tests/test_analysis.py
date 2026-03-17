import pandas as pd
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

from organelle_morphology.analysis import plot_mcs_hist, stats_contacts_a_b


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
    stats_contacts_a_b(mcs_df, label="0.0-0.05,mito-er")
