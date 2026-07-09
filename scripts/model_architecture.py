"""
model_architecture.py
----------------------
Coverage matrix: for each real imaging modality in the LOINC/RSNA Radiology
Playbook, what % of its LOINC codes actually carry data for each extraction
attribute (region, focus, laterality, contrast, study type, reason).

This is a capability map — it tells you which extractor "model" has enough
real supporting data to be worth running for a given modality (e.g. MG has
~0% Focus coverage since mammography region is always breast, but ~100%
Laterality since left/right is what actually varies).

Usage:
    python3 scripts/model_architecture.py
    python3 scripts/model_architecture.py --csv /path/to/LoincRsnaRadiologyPlaybook.csv --out output/model_architecture.html
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_all_modalities import load_csv, get_modalities

# Each extractor attribute maps to the LOINC Playbook part type(s) that carry it.
ATTRIBUTES = [
    ('Region',      ['Rad.Anatomic Location.Region Imaged']),
    ('Focus',       ['Rad.Anatomic Location.Imaging Focus']),
    ('Laterality',  ['Rad.Anatomic Location.Laterality', 'Rad.Anatomic Location.Laterality.Presence']),
    ('Contrast',    ['Rad.Timing']),
    ('Study Type',  ['Rad.Modality.Modality Subtype']),
    ('Reason',      ['Rad.Reason for Exam']),
]

# Sequential blue ramp, light→dark (references/palette.md).
RAMP_LOW = '#cde2fb'   # step 100 — near zero
RAMP_HIGH = '#0d366b'  # step 700 — near 100%


def build_coverage_matrix(rows, modalities):
    """
    For each modality, % of its LOINC codes with at least one value present
    for each attribute's underlying part type(s).
    """
    modality_to_loincs = defaultdict(set)
    for row in rows:
        if row['PartTypeName'] == 'Rad.Modality.Modality Type':
            modality_to_loincs[row['PartName']].add(row['LoincNumber'])

    pt_to_loincs = defaultdict(set)
    for row in rows:
        if row['PartTypeName'] and row['PartName']:
            pt_to_loincs[row['PartTypeName']].add(row['LoincNumber'])

    matrix = {}
    for modality in modalities:
        total = len(modality_to_loincs[modality])
        matrix[modality] = {'_total': total}
        for label, part_types in ATTRIBUTES:
            covered = set()
            for pt in part_types:
                covered |= pt_to_loincs.get(pt, set())
            covered &= modality_to_loincs[modality]
            pct = (len(covered) / total * 100) if total else 0.0
            matrix[modality][label] = {'pct': pct, 'count': len(covered), 'total': total}

    return matrix


# ── Color helpers ───────────────────────────────────────────────────────────
def _hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def interpolate_hex(c1, c2, t):
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((
        round(r1 + (r2 - r1) * t),
        round(g1 + (g2 - g1) * t),
        round(b1 + (b2 - b1) * t),
    ))


def relative_luminance(hex_color):
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _hex_to_rgb(hex_color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def cell_ink(fill_hex):
    return '#ffffff' if relative_luminance(fill_hex) < 0.45 else '#0b0b0b'


# ── HTML rendering ──────────────────────────────────────────────────────────
CSS = '''
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
    }
  }
  html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--page); color: var(--text-primary); }
  body { padding: 32px 40px 60px; }
  .wrap { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 6px; }
  .subtitle { font-size: 13px; color: var(--text-secondary); margin-bottom: 24px; max-width: 720px; line-height: 1.5; }

  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }

  .legend { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; font-size: 12px; color: var(--text-secondary); }
  .legend-bar { flex: 0 0 220px; height: 10px; border-radius: 5px; background: linear-gradient(to right, ''' + RAMP_LOW + ''', ''' + RAMP_HIGH + '''); }

  .grid-outer { overflow-x: auto; }
  table.heatmap { border-collapse: separate; border-spacing: 2px; font-size: 12px; }
  table.heatmap th, table.heatmap td { padding: 0; }
  table.heatmap th.corner { background: transparent; }
  table.heatmap th.col-head {
    font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--text-secondary); padding: 0 6px 10px; text-align: center; white-space: nowrap;
  }
  table.heatmap th.row-head {
    font-weight: 500; font-size: 13px; color: var(--text-primary); text-align: right; padding-right: 12px; white-space: nowrap;
  }
  table.heatmap th.row-head .count { display: block; font-size: 11px; color: var(--text-muted); font-weight: 400; margin-top: 1px; }

  .cell {
    width: 92px; height: 52px; border-radius: 6px; display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; cursor: default; outline: none;
    transition: outline 0.08s;
  }
  .cell:hover, .cell:focus-visible { outline: 2px solid var(--text-primary); outline-offset: -2px; }

  #tooltip {
    position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
    font-size: 12px; padding: 7px 10px; border-radius: 6px; line-height: 1.4; max-width: 240px;
    opacity: 0; transform: translateY(4px); transition: opacity 0.1s, transform 0.1s; z-index: 10;
  }
  #tooltip.show { opacity: 1; transform: translateY(0); }
  #tooltip strong { font-variant-numeric: tabular-nums; }

  .toggle-btn {
    font-size: 12px; padding: 7px 14px; border-radius: 7px; border: 1px solid var(--border);
    background: var(--surface-1); color: var(--text-primary); cursor: pointer; margin-bottom: 16px;
  }
  .toggle-btn:hover { background: var(--gridline); }

  table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; display: none; }
  table.data-table.show { display: table; }
  table.data-table th, table.data-table td { padding: 7px 12px; border-bottom: 1px solid var(--gridline); text-align: right; }
  table.data-table th:first-child, table.data-table td:first-child { text-align: left; }
  table.data-table th { color: var(--text-secondary); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  table.heatmap.show-as-table-hidden { display: none; }
'''

JS = '''
  var tooltip = document.getElementById('tooltip');
  function showTip(el, text) {
    tooltip.textContent = text;
    tooltip.classList.add('show');
  }
  function moveTip(evt) {
    tooltip.style.left = (evt.clientX + 14) + 'px';
    tooltip.style.top = (evt.clientY + 14) + 'px';
  }
  function hideTip() {
    tooltip.classList.remove('show');
  }
  document.querySelectorAll('.cell').forEach(function (cell) {
    var text = cell.getAttribute('data-tip');
    cell.addEventListener('pointerenter', function (e) { showTip(cell, text); moveTip(e); });
    cell.addEventListener('pointermove', moveTip);
    cell.addEventListener('pointerleave', hideTip);
    cell.addEventListener('focus', function (e) { showTip(cell, text); });
    cell.addEventListener('blur', hideTip);
  });

  var toggleBtn = document.getElementById('toggle-view');
  var heatmapEl = document.getElementById('heatmap-table');
  var dataTableEl = document.getElementById('data-table');
  var showingTable = false;
  toggleBtn.addEventListener('click', function () {
    showingTable = !showingTable;
    heatmapEl.classList.toggle('show-as-table-hidden', showingTable);
    dataTableEl.classList.toggle('show', showingTable);
    toggleBtn.textContent = showingTable ? 'Show as heatmap' : 'Show as table';
  });
'''


def build_html(matrix, modalities):
    attr_labels = [label for label, _ in ATTRIBUTES]

    col_heads = ''.join(f'<th class="col-head">{label}</th>' for label in attr_labels)

    body_rows = ''
    table_rows = ''
    for modality, count in modalities.items():
        row_cells = ''
        table_cells = ''
        for label in attr_labels:
            cell = matrix[modality][label]
            pct = cell['pct']
            fill = interpolate_hex(RAMP_LOW, RAMP_HIGH, pct / 100)
            ink = cell_ink(fill)
            tip = f"{modality} — {label}: {pct:.1f}% ({cell['count']:,} of {cell['total']:,} codes)"
            row_cells += (
                f'<td><div class="cell" tabindex="0" role="gridcell" '
                f'style="background:{fill};color:{ink}" '
                f'aria-label="{tip}" data-tip="{tip}">{pct:.0f}%</div></td>'
            )
            table_cells += f'<td>{pct:.1f}%</td>'

        body_rows += (
            f'<tr><th class="row-head">{modality}<span class="count">{count:,} codes</span></th>{row_cells}</tr>'
        )
        table_rows += f'<tr><td>{modality}</td><td>{count:,}</td>{table_cells}</tr>'

    table_col_heads = ''.join(f'<th>{label}</th>' for label in attr_labels)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Extraction Coverage by Modality</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Extraction Coverage by Modality</h1>
  <p class="subtitle">
    % of each modality's real LOINC/RSNA Radiology Playbook codes that carry data for each
    extraction attribute — i.e. which extractor model is actually worth running for which modality.
  </p>

  <div class="card">
    <div class="legend">
      <span>0%</span>
      <div class="legend-bar"></div>
      <span>100%</span>
      <span style="margin-left:8px;color:var(--text-muted)">coverage of that modality's codes</span>
    </div>

    <button class="toggle-btn" id="toggle-view">Show as table</button>

    <div class="grid-outer">
      <table class="heatmap" id="heatmap-table">
        <thead>
          <tr><th class="corner"></th>{col_heads}</tr>
        </thead>
        <tbody>
          {body_rows}
        </tbody>
      </table>

      <table class="data-table" id="data-table">
        <thead>
          <tr><th>Modality</th><th>Codes</th>{table_col_heads}</tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>
</div>
<div id="tooltip" role="status"></div>
<script>{JS}</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description='Generate a coverage heatmap: modality x extraction attribute.')
    parser.add_argument('--csv', '-c', help='Path to LOINC/RSNA Playbook CSV (default: data/LoincRsnaRadiologyPlaybook.csv)')
    parser.add_argument('--out', '-o', help='Output HTML path (default: output/model_architecture.html)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    csv_path = Path(args.csv) if args.csv else script_dir / 'data' / 'LoincRsnaRadiologyPlaybook.csv'
    out_path = Path(args.out) if args.out else script_dir / 'output' / 'model_architecture.html'

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    rows = load_csv(str(csv_path))
    modalities = get_modalities(rows)
    print(f"Found {len(modalities)} modalities.")

    matrix = build_coverage_matrix(rows, modalities)
    html = build_html(matrix, modalities)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
