"""View module geometry with matplotlib, or CUDA output with PyVista."""

import argparse
from pathlib import Path

import json
import yaml
from geometry import DATA, ROOT, build_model, export_model, plot_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="VTK dataset, e.g. output/result.vtu")
    parser.add_argument("--scalar", help="Optional field name to color")
    parser.add_argument("--model", type=Path, default=DATA/'module.json')
    parser.add_argument("--backend", help="Matplotlib backend, e.g. TkAgg")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.input is None:
        if args.model.is_file():
            model = json.loads(args.model.read_text())
        elif args.model == DATA/'module.json':
            model = build_model(yaml.safe_load((ROOT/'config.yaml').read_text()),
                                json.loads((DATA/'footprint.json').read_text()))
            export_model(model,DATA)
        else:
            parser.error(f"File not found: {args.model}")
        plot_model(model,args.screenshot,args.backend)
        return
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")
    import pyvista as pv
    data = pv.read(args.input)
    plotter = pv.Plotter(off_screen=bool(args.screenshot))
    plotter.add_mesh(data, scalars=args.scalar)
    plotter.add_axes()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == "__main__":
    main()
