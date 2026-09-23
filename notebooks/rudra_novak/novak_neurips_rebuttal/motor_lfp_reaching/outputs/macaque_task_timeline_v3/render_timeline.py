"""Native text/line timeline with an unchanged generated illustration insert."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 13.8, 3.5
INK, MUTED = '#171717', '#555555'


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'svg.fonttype': 'none', 'svg.image_inline': True,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'text.color': INK, 'lines.solid_capstyle': 'butt'})
    alignment_path = ROOT.parent / 'prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1/inputs/alignment.json'
    alignment = json.loads(alignment_path.read_text())
    assert alignment['monkey'] == 'T' and alignment['short_delay_ms'] == 700
    assert alignment['recording'] == 'monkeyT_session-y070316009-12_lfp-7'
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    art = plt.imread(ROOT / 'assets/macaque_screen.png')
    ax.imshow(art, extent=(.08, 2.78, 1.12, 2.92), interpolation='lanczos', zorder=1)
    ax.set_aspect('auto')
    text_artists = []

    def label(x, y, text, size=14, bold=False, color=INK, ha='center'):
        text_artists.append(ax.text(x, y, text, fontsize=size,
                                   fontweight='bold' if bold else 'normal',
                                   color=color, ha=ha, va='center', zorder=4))

    def line(x, y, lw=1.35, color=INK):
        ax.plot(x, y, lw=lw, color=color, zorder=2)

    def delay(start, end, title, value):
        y = 1.47
        line([start, end], [y, y], 1.0)
        for x in (start, end):
            line([x, x], [y-.065, y+.065], 1.0)
        label((start+end)/2, 1.16, title, 14)
        label((start+end)/2, .83, value, 15, bold=True)

    # Nominal protocol durations: D1 starts at tone offset; D2 at spatial-cue offset.
    start, tc_on, tc_off, sc_on, sc_off, go, reach = 3.18, 5.28, 5.88, 7.98, 8.145, 10.245, 12.18
    y = 1.97
    ax.annotate('', xy=(13.40, y), xytext=(3.02, y),
                arrowprops={'arrowstyle': '-|>', 'color': INK, 'lw': 1.7,
                            'mutation_scale': 14, 'shrinkA': 0, 'shrinkB': 0}, zorder=2)
    for x in (start, go, reach):
        line([x, x], [y-.15, y+.15], 1.5)
    for left, right in ((tc_on, tc_off), (sc_on, sc_off)):
        ax.add_patch(Rectangle((left, y-.065), right-left, .13,
                               facecolor=INK, edgecolor='none', zorder=3))
        for x in (left, right):
            line([x, x], [y-.12, y+.12], 1.0)

    label(start, 2.77, 'Start', 16, bold=True)
    label((tc_on+tc_off)/2, 2.97, 'Temporal cue (TC)', 16, bold=True)
    label((tc_on+tc_off)/2, 2.60, '200 ms tone', 13, color=MUTED)
    label((sc_on+sc_off)/2, 2.97, 'Spatial cue (SC)', 16, bold=True)
    label((sc_on+sc_off)/2, 2.60, '55 ms target cue', 13, color=MUTED)
    label(go, 2.77, 'GO', 16, bold=True)
    label(reach, 2.77, 'Reach', 16, bold=True)
    delay(start, tc_on, 'Initial hold', '700 ms')
    delay(tc_off, sc_on, 'Delay 1 (D1)', '700 ms')
    delay(sc_off, go, 'Delay 2 (D2)', '700 ms')
    label(1.43, .82, 'Monkey T | Short-delay trials', 11.5, color=MUTED)
    label(13.40, .26, 'Schematic; not to scale', 10, color=MUTED, ha='right')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in text_artists]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        for other in boxes[i+1:]:
            assert not box.overlaps(other), 'Overlapping text labels'
    assert abs((tc_on-start)/(sc_on-tc_off) - 1) < 1e-12
    assert abs((go-sc_off)/(sc_on-tc_off) - 1) < 1e-12
    assert abs((tc_off-tc_on)/(sc_on-tc_off) - 200/700) < 1e-12
    assert abs((sc_off-sc_on)/(sc_on-tc_off) - 55/700) < 1e-12
    fig.savefig(ROOT / 'macaque_reaching_task_horizontal.png', dpi=600)
    fig.savefig(ROOT / 'macaque_reaching_task_horizontal.svg', dpi=600)
    fig.savefig(ROOT / 'preview.png', dpi=180)
    plt.close(fig)
    (ROOT / 'provenance.json').write_text(json.dumps({
        'recording': alignment['recording'], 'trials': 'short-delay',
        'nominal_durations_ms': {'initial_hold': 700, 'temporal_cue': 200,
                                 'delay_1': 700, 'spatial_cue': 55, 'delay_2': 700},
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8152857/',
        'alignment_sha256': hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        'illustration_sha256': hashlib.sha256((ROOT / 'assets/macaque_screen.png').read_bytes()).hexdigest(),
        'timeline': 'Native vector lines and editable text; unchanged raster illustration insert',
        'png_pixels': [8280, 2100], 'png_dpi': 600,
        'validation': 'Text bounding boxes contained and nonoverlapping; nominal interval ratios verified',
        'note': 'Nominal protocol durations, not per-trial measured event intervals. No analysis window displayed.',
    }, indent=2) + '\n')
    print(f'Saved PNG and SVG in {ROOT}')


if __name__ == '__main__':
    main()
