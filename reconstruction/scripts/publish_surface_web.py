"""Copy tracked offline viewers into a generated reconstruction output tree."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/rgbd_20261014'))
    args = parser.parse_args()
    web = Path(__file__).resolve().parents[1] / 'web'
    mappings = {
        'surface_all': 'surface_viewer.html',
        'surface_outer': 'surface_viewer.html',
        'surface_contact': 'contact_surface_viewer.html',
        'surface_contact_outer': 'contact_outer_surface_viewer.html',
    }
    for folder, source in mappings.items():
        dest = args.root / folder
        if not (dest / 'manifest.json').exists():
            continue
        html = (web / source).read_text()
        if source == 'surface_viewer.html':
            html = html.replace('href="index.html"', 'href="../surface_experiment/index.html"')
            html = html.replace('src="surface_experiment.png"', 'src="../surface_experiment/surface_experiment.png"')
        (dest / 'viewer.html').write_text(html)
        for name in ('surface-font.woff2', 'OFL-NotoSansSC.txt'):
            shutil.copy2(web / name, dest / name)
        if folder == 'surface_contact_outer':
            overlay = args.root / 'surface_contact' / 'indenter_initial_overlay.jpg'
            if overlay.exists():
                shutil.copy2(overlay, dest / overlay.name)
        print(dest / 'viewer.html')


if __name__ == '__main__':
    main()
