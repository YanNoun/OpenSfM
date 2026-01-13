# pyre-strict
import argparse

from opensfm.actions import export_rerun
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "export_rerun"
    help = "Export reconstruction to Rerun format for 3D visualization"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        export_rerun.run_dataset(
            dataset,
            output=args.output,
            reconstruction_index=args.reconstruction_index,
            proj=args.proj,
        )

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Output .rrd file path (default: dataset_path/rerun.rrd)",
        )
        parser.add_argument(
            "--reconstruction-index",
            type=int,
            default=0,
            help="Index of reconstruction to export (default: 0)",
        )
        parser.add_argument(
            "--proj",
            action="store_true",
            help="Use coordinate system from gcp_list.txt",
        )