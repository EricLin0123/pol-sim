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
    parser.add_argument("--model", type=Path, help="Open a saved model instead of rebuilding the full assembly")
    parser.add_argument("--backend", help="Matplotlib backend, e.g. TkAgg")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.input is None:
        if args.model and args.model.is_file():
            model = json.loads(args.model.read_text())
        elif args.model is None:
            config = yaml.safe_load((ROOT/'config.yaml').read_text())
            winding = config.get('inductor', {}).get('winding_geometry') or {}
            model = build_model(config, json.loads((DATA/'footprint.json').read_text()),
                                demo_winding=not winding.get('topology'))
            # Viewer-only fallback must not replace canonical simulation inputs.
            export_model(model,ROOT/'output'/'visualization')
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
