"""FabCon 2026 Session Analysis CLI.

Commands
--------
preprocess   Load vault sessions and build ML-ready feature matrices.
classify     Train / evaluate classifiers to predict session track/level.
cluster      Discover session groupings with unsupervised algorithms.
reduce       Project high-dimensional features into 2-D / 3-D space.
model-select Grid / randomised search for best model hyperparameters.
"""
import click

from eda.commands.preprocess import preprocess_cmd
from eda.commands.classify import classify_cmd
from eda.commands.cluster import cluster_cmd
from eda.commands.reduce import reduce_cmd
from eda.commands.model_select import model_select_cmd


@click.group()
@click.version_option(version="0.1.0", prog_name="eda")
def cli() -> None:
    """FabCon 2026 session analysis powered by scikit-learn."""


cli.add_command(preprocess_cmd, name="preprocess")
cli.add_command(classify_cmd, name="classify")
cli.add_command(cluster_cmd, name="cluster")
cli.add_command(reduce_cmd, name="reduce")
cli.add_command(model_select_cmd, name="model-select")


if __name__ == "__main__":
    cli()
