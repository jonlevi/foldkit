"""
foldkit.cli
Command-line interface for converting AlphaFold3 confidence outputs to NPZ format.
"""

import argparse
import os
from pathlib import Path
import shutil
import tqdm
from .af3_result import AF3Result
from .storage import save_af3_result, load_af3_result


def export_single_result(input_directory: str, output_directory: str, verbose: bool):
    """Export a single AlphaFold3 Result subdirectory to compressed format."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        res = AF3Result.load_result(input_path)
        outfile = os.path.join(output_path, "confidences.npz")
        save_af3_result(res, outfile)
        if verbose:
            print(f"✅ Exported Confidence Data to : {output_path}")
        for p in input_path.iterdir():
            if "json" not in p.name and not p.is_dir():
                shutil.copyfile(p, os.path.join(output_path, p.name))
        if verbose:
            print(f"✅ Copied non Confidence Data to : {output_path}")
    except Exception as e:
        print(f"❌ Failed to export {input_path}: {e}")


def export_multi_result(input_directory: str, output_directory: str, verbose: bool):
    """Export a single AlphaFold3 Result with multiple subdirectories for each seed and sample."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # First, save the top level results to the top level
    export_single_result(input_path, output_path, verbose)

    # Second, save each subdirectory
    subdirectories = [p for p in input_path.iterdir() if p.is_dir()]

    for path in subdirectories:
        try:
            suboutput_dir = Path(os.path.join(output_path, path.name))
            suboutput_dir.mkdir(exist_ok=True)
            export_single_result(path, suboutput_dir, verbose)
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")


def batch_export_multi_result(
    input_directory: str, output_directory: str, verbose: bool
):
    """Export multiple AlphaFold3 Results, each with multiple subdirectories for each seed and sample."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    subdirectories = [p for p in input_path.iterdir() if p.is_dir()]
    for path in tqdm.tqdm(subdirectories):
        suboutput_dir = Path(os.path.join(output_path, path.name))
        suboutput_dir.mkdir(exist_ok=True)
        export_multi_result(path, suboutput_dir, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Export AlphaFold3 result directories into compressed format."
        "Converts confidences into npz format and copies over the rest of the data as is( except the _input_data.json which is redundant)."
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output.",
        default=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_file = subparsers.add_parser(
        "export-single-result",
        help="Export a single AlphaFold3 result directory to compressed format",
    )
    parser_file.add_argument(
        "input_directory", help="Path to AlphaFold3 result directory"
    )
    parser_file.add_argument("output_directory", help="Output directory path")

    parser_dir = subparsers.add_parser(
        "export-multi-result",
        help="Export multiseed/multisample AlphaFold3 results to compressed format.",
    )
    parser_dir.add_argument(
        "input_directory",
        help="Path to parent directory containing subdirectories with AF3 results.",
    )
    parser_dir.add_argument("output_directory", help="Parent output directory path")

    parser_dir = subparsers.add_parser(
        "batch-export-multi-result",
        help="Export multiple AlphaFold3 results to compressed format.",
    )
    parser_dir.add_argument(
        "input_directory",
        help="Path to parent directory containing subdirectories with subdirectories of AF3 results.",
    )
    parser_dir.add_argument("output_directory", help="Parent output directory path")

    args = parser.parse_args()

    command_mappers = {
        "export-single-result": export_single_result,
        "export-multi-result": export_multi_result,
        "batch-export-multi-result": batch_export_multi_result,
    }

    command = command_mappers.get(args.command)

    command(args.input_directory, args.output_directory, args.verbose)
