# PYTHON_ARGCOMPLETE_OK
"""
foldkit.cli
Command-line interface for converting AlphaFold3 confidence outputs to NPZ format.
"""

import argparse
import argcomplete
from argcomplete.completers import DirectoriesCompleter
import csv
import os
from pathlib import Path
import re
import shutil
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .af3_result import AF3Result
from .af3_ensemble import AF3Ensemble
from .storage import save_af3_result


def available_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # sched_getaffinity is Linux-only; fall back on other platforms
        return multiprocessing.cpu_count()


def write_ranking_csv(ranking_scores, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["seed", "sample", "ranking_score"])

        for key, ranking_score in ranking_scores.items():
            match = re.match(r"seed-(\d+)_sample-(\d+)", key)

            if match is None:
                raise ValueError(f"Invalid key format: {key}")

            seed = int(match.group(1))
            sample = int(match.group(2))

            writer.writerow([seed, sample, ranking_score])


def get_cif_paths(top):

    sample_paths = {}

    for _path in top.iterdir():
        path = Path(_path)
        if "cif" not in _path.name:
            continue

        match = re.search(r"_model_(\d+)\.cif$", path.name)

        if match is None:
            raise ValueError(f"Could not extract sample number from: {path}")

        sample = int(match.group(1))
        sample_paths[sample] = str(path)

    return sample_paths


def export_single_result(input_directory: str, output_directory: str, verbose: bool):
    """Export a single AlphaFold3 Result subdirectory to compressed format."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        cif_path = None
        for p in input_path.iterdir():
            if "json" not in p.name and not p.is_dir():
                shutil.copyfile(p, os.path.join(output_path, p.name))
                if "cif" in p.name:
                    cif_path = os.path.join(output_path, p.name)
        res = AF3Result.load_af3_result(input_path)
        outfile = os.path.join(output_path, "confidences.npz")
        save_af3_result(res, outfile, override_cif_path=cif_path)
        if verbose:
            print(f"✅ Exported Data to : {output_path}")

    except Exception as e:
        print(f"❌ Failed to export {input_path}: {e}")


def export_ensemble_result(input_directory: str, output_directory: str, verbose: bool):
    """Export a single AlphaFold3 Ensemble Result with multiple subdirectories for each seed and sample."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    ranking_files = list(input_path.glob("*ranking_scores.csv"))
    if len(ranking_files) == 0:
        raise Exception("Could not find ranking_scores.csv file")
    if len(ranking_files) > 1:
        raise Exception(
            "Found multiple ranking_scores.csv files and don't know which to use."
        )
    ranking_file = ranking_files[0]

    shutil.copyfile(ranking_file, os.path.join(output_path, "ranking_scores.csv"))

    # Save each subdirectory
    subdirectories = [p for p in input_path.iterdir() if p.is_dir()]

    for path in subdirectories:
        try:
            suboutput_dir = Path(os.path.join(output_path, path.name))
            suboutput_dir.mkdir(exist_ok=True)
            export_single_result(path, suboutput_dir, verbose)
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")


def export_webserver_result(input_directory: str, output_directory: str, verbose: bool):
    """Export a single AlphaFold3 Webserver result with multiple files for each sample, usually for one seed."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        for p in input_path.iterdir():
            if "json" not in p.name and not p.is_dir():
                shutil.copyfile(p, os.path.join(output_path, p.name))

        ensemble = AF3Ensemble.load_webserver_result(input_path)
        cif_paths = get_cif_paths(output_path)
        for key, res in ensemble.af3_results.items():
            subdir = Path(os.path.join(output_path, key))
            subdir.mkdir(parents=True, exist_ok=True)
            outfile = os.path.join(subdir, "confidences.npz")
            match = re.match(r"seed-(\d+)_sample-(\d+)", key)
            seed = int(match.group(1))
            sample = int(match.group(2))
            new_cif_path = os.path.join(subdir, "model.cif")
            shutil.move(cif_paths[sample], new_cif_path)
            save_af3_result(res, outfile, override_cif_path=new_cif_path)

        write_ranking_csv(
            ensemble.af3_ranking_scores, os.path.join(output_path, "ranking_scores.csv")
        )

        if verbose:
            print(f"✅ Exported Data to : {output_path}")

    except Exception as e:
        print(f"❌ Failed to export {input_path}: {e}")


def batch_export(
    input_directory: str,
    output_directory: str,
    verbose: bool,
    multithreading: bool = True,
    max_workers: int = None,
):
    """Export multiple AlphaFold3 Ensemble Results, each with multiple subdirectories for each seed and sample."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    subdirectories = [p for p in input_path.iterdir() if p.is_dir()]

    def _export_one(path: Path):
        if verbose:
            tqdm.tqdm.write(
                f"[{path.name}] running on thread {threading.current_thread().name}"
            )
        suboutput_dir = output_path / path.name
        suboutput_dir.mkdir(exist_ok=True)
        export_ensemble_result(path, suboutput_dir, verbose)
        return path.name

    if multithreading:
        if max_workers is None:
            try:
                max_workers = available_cpu_count()
            except Exception:
                max_workers = 4  # safe fallback if detection fails entirely
        print(f"Running in multithreading mode with max_workers={max_workers}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_export_one, p): p for p in subdirectories}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    tqdm.tqdm.write(f"FAILED {futures[future].name}: {e}")
    else:
        print(f"Running with multithreading=False...")
        for path in tqdm.tqdm(subdirectories):
            suboutput_dir = Path(os.path.join(output_path, path.name))
            suboutput_dir.mkdir(exist_ok=True)
            export_ensemble_result(path, suboutput_dir, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Export AlphaFold3 result directories into compressed format."
        "Converts confidences into npz format and copies over the rest of the data as is (except the _input_data.json which is redundant)."
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
    ).completer = DirectoriesCompleter()
    parser_file.add_argument(
        "output_directory", help="Output directory path"
    ).completer = DirectoriesCompleter()

    parser_dir = subparsers.add_parser(
        "export-ensemble-result",
        help="Export multiseed/multisample AlphaFold3 Ensemble of results to compressed format.",
    )
    parser_dir.add_argument(
        "input_directory",
        help="Path to parent directory containing subdirectories with AF3 results.",
    ).completer = DirectoriesCompleter()
    parser_dir.add_argument(
        "output_directory", help="Parent output directory path"
    ).completer = DirectoriesCompleter()

    parser_web = subparsers.add_parser(
        "webserver-export",
        help="Export AlphaFold3 server results to compressed format.",
    )
    parser_web.add_argument(
        "input_directory",
        help="Path to parent directory containing webserver AF3 results.",
    ).completer = DirectoriesCompleter()
    parser_web.add_argument(
        "output_directory", help="Parent output directory path"
    ).completer = DirectoriesCompleter()

    parser_batch = subparsers.add_parser(
        "batch-export",
        help="Export multiple AlphaFold3 results to compressed format.",
    )
    parser_batch.add_argument(
        "input_directory",
        help="Path to parent directory containing subdirectories with subdirectories of AF3 results.",
    ).completer = DirectoriesCompleter()
    parser_batch.add_argument(
        "output_directory", help="Parent output directory path"
    ).completer = DirectoriesCompleter()

    parser_batch.add_argument(
        "--multithreading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use multithreading",
    )
    parser_batch.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of workers for multithreading",
    )

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    command_mappers = {
        "export-single-result": export_single_result,
        "export-ensemble-result": export_ensemble_result,
        "batch-export": batch_export,
        "webserver-export": export_webserver_result,
    }
    command = command_mappers.get(args.command)

    kwargs = {}
    if args.command == "batch-export":
        kwargs["multithreading"] = args.multithreading
        kwargs["max_workers"] = args.max_workers

    command(args.input_directory, args.output_directory, args.verbose, **kwargs)
