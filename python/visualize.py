"""Preview a future CUDA output exported as a VTK dataset; no results fabricated."""

import argparse
from pathlib import Path

import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="VTK dataset, e.g. output/result.vtu")
    parser.add_argument("--scalar", help="Optional field name to color")
    args = parser.parse_args()
    if args.input is None:
        print("No simulation output yet. Supply a VTK dataset when an exporter exists.")
        return
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")
    data = pv.read(args.input)
    plotter = pv.Plotter()
    plotter.add_mesh(data, scalars=args.scalar)
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
