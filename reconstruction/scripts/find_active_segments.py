"""Find the loading segments of a session and write them as a frame list for reconstruction.

A frame is active when the Mark-10 reports indentation or load, or when the tactile array
departs from its rest level (it lags the gauge while the foam recovers). Segments are padded
so each one starts and ends on the undeformed object. Videos and CSVs are left untouched:
frame indices stay the original video_frame_index, so outputs join back to the alignment CSV.
"""
from pathlib import Path
import argparse, csv, json

import numpy as np


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append([i, j]); i = j + 1
        else:
            i += 1
    return out


def find_segments(session, pad_s, merge_s, min_distance_mm, min_load_n, tactile_factor):
    export = json.loads((session / 'multimodal_video_export.json').read_text())
    fps = export['fps']
    rows = list(csv.DictReader((session / 'sensor_force_aligned.csv').open()))
    value = lambda r, k: float(r[k]) if r[k] else np.nan
    distance = np.array([value(r, 'force_distance') for r in rows])
    load = np.array([value(r, 'force_load') for r in rows])
    rms = np.array([value(r, 'rms_delta_v') for r in rows])
    baseline = float(np.nanmedian(rms))
    force_active = (np.nan_to_num(np.abs(distance)) > min_distance_mm) | (np.nan_to_num(load) > min_load_n)
    active = force_active | (rms > tactile_factor * baseline)
    merged = []
    for a, b in runs(active):
        if merged and a - merged[-1][1] <= merge_s * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # Drop isolated tactile blips: a segment must contain gauge activity.
    merged = [[a, b] for a, b in merged if force_active[a:b + 1].any()]
    pad, last = round(pad_s * fps), len(rows) - 1
    segments = []
    for a, b in merged:
        start, end = max(0, a - pad), min(last, b + pad)
        if segments and start <= segments[-1]['end_frame']:
            start = segments[-1]['end_frame'] + 1
        segments.append({'start_frame': start, 'end_frame': end, 'start_s': start / fps, 'end_s': end / fps,
                         'frames': end - start + 1, 'peak_load_n': float(np.nanmax(load[a:b + 1])),
                         'max_indentation_mm': float(np.nanmax(-distance[a:b + 1]))})
    frames = [f for s in segments for f in range(s['start_frame'], s['end_frame'] + 1)]
    return {'version': 1, 'session': session.name, 'fps': fps, 'total_frames': len(rows),
            'kept_frames': len(frames), 'tactile_rms_baseline_v': baseline,
            'criteria': {'pad_s': pad_s, 'merge_s': merge_s, 'min_distance_mm': min_distance_mm,
                         'min_load_n': min_load_n, 'tactile_factor': tactile_factor},
            'segments': segments, 'frames': frames}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, help='default: <session>/active_segments.json')
    p.add_argument('--pad-s', type=float, default=1.0, help='rest frames kept before and after each segment')
    p.add_argument('--merge-s', type=float, default=1.0, help='join segments separated by less than this')
    p.add_argument('--min-distance-mm', type=float, default=0.2)
    p.add_argument('--min-load-n', type=float, default=0.5)
    p.add_argument('--tactile-factor', type=float, default=2.5, help='active when rms_delta_v exceeds factor x median')
    p.add_argument('--print-chunk', metavar='I/N', help='print the I-th of N comma-separated frame chunks (1-based) and exit')
    args = p.parse_args()
    output = args.output or args.session / 'active_segments.json'
    if args.print_chunk:
        i, n = map(int, args.print_chunk.split('/'))
        frames = json.loads(output.read_text())['frames']
        print(','.join(map(str, frames[(i - 1) * len(frames) // n:i * len(frames) // n])))
        return
    result = find_segments(args.session, args.pad_s, args.merge_s, args.min_distance_mm, args.min_load_n, args.tactile_factor)
    output.write_text(json.dumps(result, indent=2))
    for k, s in enumerate(result['segments'], 1):
        print(f"segment {k}: frames {s['start_frame']}-{s['end_frame']} ({s['start_s']:.1f}-{s['end_s']:.1f} s, "
              f"{s['frames']} frames), peak {s['peak_load_n']:.1f} N, {s['max_indentation_mm']:.1f} mm")
    print(f"kept {result['kept_frames']}/{result['total_frames']} frames -> {output}")


if __name__ == '__main__':
    main()
