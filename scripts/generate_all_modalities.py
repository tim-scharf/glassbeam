"""
generate_all_modalities.py
--------------------------
Generates a single self-contained tabbed HTML file covering all modalities
in the LOINC/RSNA Radiology Playbook CSV.

Usage:
    python3 generate_all_modalities.py
    python3 generate_all_modalities.py --csv /path/to/LoincRsnaRadiologyPlaybook.csv
    python3 generate_all_modalities.py --out my_output.html

Output:
    LOINC_Radiology_All_Modalities.html  (or --out path)
"""

import csv, sys, os, re, argparse
from collections import defaultdict
from pathlib import Path

# ── Branch config ─────────────────────────────────────────────────────────────
BRANCHES = {
    "modality":       {"label": "Modality",      "color": "#1d9e75", "border": "#0f6e56", "text": "#e1f5ee", "sub": "#9fe1cb", "tag": "teal",   "bar": "teal"},
    "anatomy":        {"label": "Anatomy",        "color": "#7f77dd", "border": "#534ab7", "text": "#eeedfe", "sub": "#afa9ec", "tag": "purple", "bar": "purple"},
    "timing":         {"label": "Protocol",       "color": "#ba7517", "border": "#854f0b", "text": "#faeeda", "sub": "#fac775", "tag": "amber",  "bar": "amber"},
    "pharmaceutical": {"label": "Pharmaceutical", "color": "#d85a30", "border": "#993c1d", "text": "#faece7", "sub": "#f5c4b3", "tag": "coral",  "bar": "coral"},
    "guidance":       {"label": "Guidance",       "color": "#5f5e5a", "border": "#444441", "text": "#f1efe8", "sub": "#b4b2a9", "tag": "gray",   "bar": "gray"},
}

PART_TYPE_META = {
    "Rad.Modality.Modality Subtype":             {"label": "Modality Subtype",  "branch": "modality",       "group": "Modality"},
    "Rad.Anatomic Location.Region Imaged":       {"label": "Region Imaged",     "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Imaging Focus":       {"label": "Imaging Focus",     "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Laterality":          {"label": "Laterality",        "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Laterality.Presence": {"label": "Lat. Presence",     "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Timing":                                {"label": "Timing / Contrast", "branch": "timing",         "group": "Protocol"},
    "Rad.View.Aggregation":                      {"label": "View Aggregation",  "branch": "timing",         "group": "Protocol"},
    "Rad.View.View type":                        {"label": "View Type",         "branch": "timing",         "group": "Protocol"},
    "Rad.View.View Type":                        {"label": "View Type",         "branch": "timing",         "group": "Protocol"},
    "Rad.Reason for Exam":                       {"label": "Reason for Exam",   "branch": "timing",         "group": "Protocol"},
    "Rad.Maneuver.Maneuver type":                {"label": "Maneuver",          "branch": "timing",         "group": "Protocol"},
    "Rad.Maneuver.Maneuver Type":                {"label": "Maneuver",          "branch": "timing",         "group": "Protocol"},
    "Rad.Subject":                               {"label": "Subject",           "branch": "timing",         "group": "Protocol"},
    "Rad.Pharmaceutical.Substance Given":        {"label": "Substance",         "branch": "pharmaceutical", "group": "Pharmaceutical"},
    "Rad.Pharmaceutical.Route":                  {"label": "Route",             "branch": "pharmaceutical", "group": "Pharmaceutical"},
    "Rad.Guidance for.Action":                   {"label": "Action",            "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Object":                   {"label": "Object",            "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Approach":                 {"label": "Approach",          "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Presence":                 {"label": "Presence",          "branch": "guidance",       "group": "Interventional"},
}

GROUP_ORDER = ["Modality", "Anatomy", "Protocol", "Pharmaceutical", "Interventional"]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def get_modalities(rows):
    counts = defaultdict(set)
    for row in rows:
        if row['PartTypeName'] == 'Rad.Modality.Modality Type':
            counts[row['PartName']].add(row['LoincNumber'])
    return {m: len(c) for m, c in sorted(counts.items(), key=lambda x: -len(x[1]))}

def extract_data(rows, modality):
    loinc_set = set()
    for row in rows:
        if row['PartTypeName'] == 'Rad.Modality.Modality Type' and row['PartName'] == modality:
            loinc_set.add(row['LoincNumber'])
    part_data = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row['LoincNumber'] in loinc_set:
            pt, pn = row['PartTypeName'], row['PartName']
            if pt and pn and pt != 'Rad.Modality.Modality Type':
                part_data[pt][pn] += 1
    return len(loinc_set), part_data

# ── SVG tree diagram ──────────────────────────────────────────────────────────
def build_svg_tree(modality, total_codes, part_data):
    # Collect which branches have data
    branch_nodes = {}
    for pt, vals in part_data.items():
        if pt not in PART_TYPE_META:
            continue
        b = PART_TYPE_META[pt]['branch']
        if b not in branch_nodes:
            branch_nodes[b] = []
        branch_nodes[b].append((PART_TYPE_META[pt]['label'], len(vals), pt))

    active_branches = [b for b in ["modality","anatomy","timing","pharmaceutical","guidance"] if b in branch_nodes]
    n = len(active_branches)

    # Layout constants
    ROOT_Y    = 30
    ROOT_W, ROOT_H   = 220, 52
    BRANCH_Y         = 150
    BRANCH_W, BRANCH_H = 180, 52
    CHILD_Y   = 270
    CHILD_W   = 88    # small pills
    CHILD_H   = 32
    CHILD_GAP = 6
    CELL_GAP  = 40    # horizontal gap between branch cells
    MARGIN    = 60
    MIN_TOTAL_W = 1200

    # Each branch's children (part types), same set used to draw pills below.
    branch_children = {
        bkey: sorted(
            [(pt, vals) for pt, vals in part_data.items()
             if pt in PART_TYPE_META and PART_TYPE_META[pt]['branch'] == bkey],
            key=lambda x: PART_TYPE_META[x[0]]['label']
        )
        for bkey in active_branches
    }

    # A branch's cell must be wide enough to hold its own box AND all its child
    # pills side by side — otherwise adjacent branches' pills collide.
    cell_widths = []
    for bkey in active_branches:
        nc = len(branch_children[bkey])
        content_w = nc * CHILD_W + max(nc - 1, 0) * CHILD_GAP
        cell_widths.append(max(BRANCH_W, content_w))

    total_w = sum(cell_widths) + max(n - 1, 0) * CELL_GAP + 2 * MARGIN
    extra_margin = max(0, (MIN_TOTAL_W - total_w) / 2)
    margin = MARGIN + extra_margin
    W = int(total_w + 2 * extra_margin)

    cell_left = []
    x = margin
    for w in cell_widths:
        cell_left.append(x)
        x += w + CELL_GAP

    branch_cx = [int(cell_left[i] + cell_widths[i] / 2) for i in range(n)]
    ROOT_CX   = W // 2

    elems = []
    elems.append('''<defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#b4b2a9" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>''')

    # Root node
    rx = ROOT_CX - ROOT_W // 2
    elems.append(f'''
  <g class="node" onclick="showSection('overview')" style="cursor:pointer">
    <rect x="{rx}" y="{ROOT_Y}" width="{ROOT_W}" height="{ROOT_H}" rx="10"
          fill="#185fa5" stroke="#0c447c" stroke-width="1"/>
    <text x="{ROOT_CX}" y="{ROOT_Y+18}" text-anchor="middle" dominant-baseline="central"
          fill="#e6f1fb" font-size="16" font-weight="600" font-family="-apple-system,sans-serif">{modality} Modality</text>
    <text x="{ROOT_CX}" y="{ROOT_Y+37}" text-anchor="middle" dominant-baseline="central"
          fill="#85b7eb" font-size="12" font-family="-apple-system,sans-serif">{total_codes:,} LOINC codes</text>
  </g>''')

    for i, bkey in enumerate(active_branches):
        bcx = branch_cx[i]
        bx  = bcx - BRANCH_W // 2
        bc  = BRANCHES[bkey]
        items = branch_nodes[bkey]

        subtitle = " · ".join(lbl for lbl, _, _ in items[:2])
        if len(subtitle) > 26:
            subtitle = subtitle[:23] + "…"

        # Root → branch connector
        elems.append(f'<line x1="{ROOT_CX}" y1="{ROOT_Y+ROOT_H}" x2="{bcx}" y2="{BRANCH_Y}" '
                     f'stroke="#b4b2a9" stroke-width="0.8" marker-end="url(#arr)"/>')

        # First section link for this branch
        sec_id = next(
            (pt.replace('.', '-').replace(' ', '-').lower()
             for pt, meta in PART_TYPE_META.items()
             if meta['branch'] == bkey and pt in part_data),
            'overview'
        )

        elems.append(f'''
  <g class="node" onclick="showSection('{sec_id}')" style="cursor:pointer">
    <rect x="{bx}" y="{BRANCH_Y}" width="{BRANCH_W}" height="{BRANCH_H}" rx="9"
          fill="{bc['color']}" stroke="{bc['border']}" stroke-width="0.8"/>
    <text x="{bcx}" y="{BRANCH_Y+18}" text-anchor="middle" dominant-baseline="central"
          fill="{bc['text']}" font-size="14" font-weight="600" font-family="-apple-system,sans-serif">{bc['label']}</text>
    <text x="{bcx}" y="{BRANCH_Y+36}" text-anchor="middle" dominant-baseline="central"
          fill="{bc['sub']}" font-size="11" font-family="-apple-system,sans-serif">{subtitle}</text>
  </g>''')

        # Child pills — each branch's cell was pre-sized to fit these, so no
        # clamping/overlap-avoidance is needed here.
        children = branch_children[bkey]
        nc = len(children)
        if nc == 0:
            continue

        total_cw   = nc * CHILD_W + (nc - 1) * CHILD_GAP
        start_x    = bcx - total_cw // 2

        for j, (pt, vals) in enumerate(children):
            clabel    = PART_TYPE_META[pt]['label']
            short     = clabel if len(clabel) <= 11 else clabel[:10] + "…"
            ccount    = len(vals)
            ccx       = start_x + j * (CHILD_W + CHILD_GAP) + CHILD_W // 2
            cx_left   = start_x + j * (CHILD_W + CHILD_GAP)
            child_sec = pt.replace('.', '-').replace(' ', '-').lower()

            elems.append(f'<line x1="{bcx}" y1="{BRANCH_Y+BRANCH_H}" x2="{ccx}" y2="{CHILD_Y}" '
                         f'stroke="#b4b2a9" stroke-width="0.5" marker-end="url(#arr)"/>')
            elems.append(f'''
  <g class="node" onclick="showSection('{child_sec}')" style="cursor:pointer">
    <rect x="{cx_left}" y="{CHILD_Y}" width="{CHILD_W}" height="{CHILD_H}" rx="5"
          fill="{bc['sub']}" stroke="{bc['color']}" stroke-width="0.5"/>
    <text x="{ccx}" y="{CHILD_Y+11}" text-anchor="middle" dominant-baseline="central"
          fill="{bc['border']}" font-size="10" font-weight="500" font-family="-apple-system,sans-serif">{short}</text>
    <text x="{ccx}" y="{CHILD_Y+23}" text-anchor="middle" dominant-baseline="central"
          fill="{bc['border']}" font-size="9" font-family="-apple-system,sans-serif">{ccount} vals</text>
  </g>''')

    # Legend
    legend_y = CHILD_Y + CHILD_H + 36
    elems.append(f'<line x1="40" y1="{legend_y}" x2="{W-40}" y2="{legend_y}" stroke="#e8e7e1" stroke-width="0.5"/>')
    elems.append(f'<text x="{ROOT_CX}" y="{legend_y+18}" text-anchor="middle" '
                 f'fill="#888780" font-size="12" font-family="-apple-system,sans-serif">Click any node to see full value list</text>')
    lx = 120
    for bkey in active_branches:
        bc = BRANCHES[bkey]
        elems.append(f'<rect x="{lx}" y="{legend_y+34}" width="13" height="13" rx="3" fill="{bc["color"]}"/>')
        elems.append(f'<text x="{lx+18}" y="{legend_y+41}" fill="#5f5e5a" font-size="12" '
                     f'dominant-baseline="central" font-family="-apple-system,sans-serif">{bc["label"]}</text>')
        lx += 220

    svg_h = legend_y + 62
    return f'<svg width="100%" viewBox="0 0 {W} {svg_h}" role="img">{"".join(elems)}</svg>'

# ── Detail section HTML ───────────────────────────────────────────────────────
def bar_html(count, max_count, bar_cls):
    pct = max(1, int((count / max_count) * 100)) if max_count else 0
    return f'<div class="bar-bg"><div class="bar-fill bar-{bar_cls}" style="width:{pct}%"></div></div>'

def section_html(part_type, values_dict, meta):
    bc = BRANCHES[meta['branch']]
    sorted_vals = sorted(values_dict.items(), key=lambda x: -x[1])
    max_count   = sorted_vals[0][1] if sorted_vals else 1
    total       = sum(v for _, v in sorted_vals)
    rows_html   = ""
    for val, cnt in sorted_vals[:60]:
        rows_html += f'''
        <tr>
          <td><span class="tag {bc['tag']}">{val}</span></td>
          <td style="text-align:right;font-variant-numeric:tabular-nums">{cnt:,}</td>
          <td class="bar-cell">{bar_html(cnt, max_count, bc['bar'])}</td>
        </tr>'''
    remainder = len(sorted_vals) - 60
    if remainder > 0:
        rows_html += f'<tr><td colspan="3" style="color:#888780;font-size:12px;padding-top:8px">+ {remainder:,} more values not shown</td></tr>'
    sec_id = part_type.replace('.', '-').replace(' ', '-').lower()
    return f'''
    <div id="sec-{sec_id}" class="section">
      <div class="diagram-card">
        <div style="background:{bc['color']};border-radius:8px 8px 0 0;padding:16px 22px;margin:-28px -28px 22px;">
          <h2 style="color:{bc['text']};margin:0;font-size:16px;font-weight:500">{meta['label']}</h2>
          <p style="color:{bc['sub']};margin:4px 0 0;font-size:12px">{len(sorted_vals):,} distinct values · {total:,} total occurrences</p>
        </div>
        <table>
          <tr><th>Value</th><th style="text-align:right">Count</th><th class="bar-cell">Frequency</th></tr>
          {rows_html}
        </table>
      </div>
    </div>'''

def sidebar_item_html(part_type, values_dict, meta):
    sec_id = part_type.replace('.', '-').replace(' ', '-').lower()
    return (f'<div class="sidebar-item" onclick="showSection(\'{sec_id}\')">'
            f'{meta["label"]} <span class="count">{len(values_dict)}</span></div>')

# ── Per-modality sidebar + main content ───────────────────────────────────────
def build_modality_content(modality, total_codes, part_data):
    groups = defaultdict(list)
    for pt, vals in sorted(part_data.items()):
        if pt in PART_TYPE_META:
            meta = PART_TYPE_META[pt]
            groups[meta['group']].append((pt, vals, meta))

    sidebar_html = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        sidebar_html += f'<div class="sidebar-section"><h2>{group}</h2>'
        for pt, vals, meta in groups[group]:
            sidebar_html += sidebar_item_html(pt, vals, meta)
        sidebar_html += '</div>'

    sections_html = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        for pt, vals, meta in groups[group]:
            sections_html += section_html(pt, vals, meta)

    focus_count   = len(part_data.get("Rad.Anatomic Location.Imaging Focus", {}))
    region_count  = len(part_data.get("Rad.Anatomic Location.Region Imaged", {}))
    subtype_count = len(part_data.get("Rad.Modality.Modality Subtype", {}))
    guidance_count = len(part_data.get("Rad.Guidance for.Action", {}))

    summary_rows = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        for pt, vals, meta in groups[group]:
            bc = BRANCHES[meta['branch']]
            sec_id = pt.replace('.', '-').replace(' ', '-').lower()
            summary_rows += f'''
          <tr style="cursor:pointer" onclick="showSection(\'{sec_id}\')">
            <td><span class="tag {bc['tag']}">{meta['label']}</span></td>
            <td style="color:#888780;font-size:12px">{group}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums">{len(vals):,}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums;color:#888780">{sum(vals.values()):,}</td>
          </tr>'''

    svg_tree = build_svg_tree(modality, total_codes, part_data)

    main_html = f'''
      <div id="sec-overview" class="section active">
        <div class="stats-bar">
          <div class="stat"><div class="val">{total_codes:,}</div><div class="lbl">Total {modality} LOINC codes</div></div>
          <div class="stat"><div class="val">{focus_count or "—"}</div><div class="lbl">Imaging focus values</div></div>
          <div class="stat"><div class="val">{region_count or "—"}</div><div class="lbl">Regions imaged</div></div>
          <div class="stat"><div class="val">{guidance_count or "—"}</div><div class="lbl">Guidance actions</div></div>
          <div class="stat"><div class="val">{subtype_count or "—"}</div><div class="lbl">Modality subtypes</div></div>
        </div>
        <div class="diagram-card">
          <h2>{modality} — Attribute Tree</h2>
          <p class="subtitle">All LOINC Radiology Part types used across {total_codes:,} {modality} procedure codes. Click any node to explore.</p>
          {svg_tree}
        </div>
        <div class="diagram-card">
          <h2>{modality} — Attribute summary</h2>
          <p class="subtitle">Click any row to see the full value breakdown.</p>
          <table>
            <tr><th>Part type</th><th>Group</th><th style="text-align:right">Distinct values</th><th style="text-align:right">Total occurrences</th></tr>
            {summary_rows}
          </table>
        </div>
      </div>
      {sections_html}'''

    return sidebar_html, main_html

# ── Namespace helper (prevents ID collisions across tabs) ─────────────────────
def namespace(content, safe):
    content = re.sub(r'id="sec-([^"]+)"',
                     lambda m: f'id="{safe}-sec-{m.group(1)}"', content)
    content = re.sub(r"showSection\('([^']+)'\)",
                     lambda m: f"showSection('{safe}', '{m.group(1)}')", content)
    return content

# ── Full page HTML ────────────────────────────────────────────────────────────
CSS = '''
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: #f5f4f0; color: #2c2c2a; overflow: hidden; display: flex; flex-direction: column; }

  header { background: #1a1a18; color: #f0efe9; padding: 16px 40px; flex-shrink: 0; display: flex; align-items: center; gap: 20px; }
  header h1 { font-size: 18px; font-weight: 500; letter-spacing: -0.01em; }
  header p  { font-size: 12px; color: #9c9a92; margin-top: 3px; }
  .badge { background: #185fa5; color: #e6f1fb; font-size: 11px; padding: 3px 10px; border-radius: 12px; margin-left: 10px; vertical-align: middle; }

  .tab-bar { background: #fff; border-bottom: 2px solid #e8e7e1; flex-shrink: 0; display: flex; overflow-x: auto; scrollbar-width: none; }
  .tab-bar::-webkit-scrollbar { display: none; }
  .tab-btn { display: flex; flex-direction: column; align-items: flex-start; padding: 12px 24px; cursor: pointer; border: none; background: none; border-bottom: 3px solid transparent; margin-bottom: -2px; white-space: nowrap; transition: background 0.12s; gap: 3px; min-width: 120px; }
  .tab-btn:hover { background: #f5f4f0; }
  .tab-btn.active { border-bottom-color: #185fa5; background: #f0f6fc; }
  .tab-code { font-size: 15px; font-weight: 600; color: #2c2c2a; }
  .tab-btn.active .tab-code { color: #185fa5; }
  .tab-sub { font-size: 11px; color: #888780; }
  .tab-btn.active .tab-sub { color: #378add; }

  .panels-wrapper { flex: 1; overflow: hidden; display: flex; flex-direction: column; min-height: 0; }
  .modality-panel { flex: 1; min-height: 0; }

  .sidebar { width: 300px; min-width: 300px; background: #fff; border-right: 1px solid #d3d1c7; overflow-y: auto; flex-shrink: 0; height: 100%; }
  .sidebar-section { border-bottom: 1px solid #eeece6; }
  .sidebar-section h2 { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: #888780; padding: 13px 18px 5px; }
  .sidebar-item { display: flex; justify-content: space-between; align-items: center; padding: 9px 18px; font-size: 13px; cursor: pointer; border-left: 3px solid transparent; transition: background 0.1s; }
  .sidebar-item:hover, .sidebar-overview:hover { background: #f5f4f0; }
  .sidebar-item.active, .sidebar-overview.active { background: #e6f1fb; border-left-color: #185fa5; color: #0c447c; font-weight: 500; }
  .sidebar-overview { padding: 9px 18px; font-size: 13px; cursor: pointer; border-left: 3px solid transparent; display: block; }
  .count { font-size: 11px; color: #888780; background: #f1efe8; padding: 2px 8px; border-radius: 10px; }
  .sidebar-item.active .count { background: #b5d4f4; color: #0c447c; }

  .main { flex: 1; overflow-y: auto; padding: 32px 40px; min-height: 0; height: 100%; }

  .stats-bar { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
  .stat { background: #fff; border: 1px solid #d3d1c7; border-radius: 10px; padding: 18px 22px; flex: 1; min-width: 120px; }
  .stat .val { font-size: 28px; font-weight: 500; color: #185fa5; }
  .stat .lbl { font-size: 12px; color: #888780; margin-top: 4px; }

  .diagram-card { background: #fff; border: 1px solid #d3d1c7; border-radius: 12px; padding: 28px; max-width: 1600px; margin-bottom: 24px; }
  .diagram-card h2 { font-size: 16px; font-weight: 500; margin-bottom: 5px; }
  .diagram-card .subtitle { font-size: 13px; color: #5f5e5a; margin-bottom: 20px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 9px 14px; background: #f5f4f0; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #5f5e5a; }
  td { padding: 7px 14px; border-bottom: 1px solid #f1efe8; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafaf8; }
  .bar-cell { width: 200px; }
  .bar-bg { background: #e8e7e1; border-radius: 3px; height: 7px; }
  .bar-fill { height: 7px; border-radius: 3px; }
  .bar-teal   { background: #1d9e75; }
  .bar-purple { background: #7f77dd; }
  .bar-amber  { background: #ba7517; }
  .bar-coral  { background: #d85a30; }
  .bar-gray   { background: #888780; }

  .tag { display: inline-block; font-size: 12px; padding: 3px 10px; border-radius: 12px; }
  .tag.teal   { background: #e1f5ee; color: #085041; }
  .tag.purple { background: #eeedfe; color: #3c3489; }
  .tag.amber  { background: #faeeda; color: #633806; }
  .tag.coral  { background: #faece7; color: #712b13; }
  .tag.gray   { background: #f1efe8; color: #444441; }

  .section { display: none; }
  .section.active { display: block; }
  .node { cursor: pointer; }
  .node rect { transition: opacity 0.15s; }
  .node:hover rect { opacity: 0.82; }

  footer { text-align: center; font-size: 11px; color: #888780; padding: 12px; border-top: 1px solid #e8e7e1; background: #fff; flex-shrink: 0; }
'''

JS = '''
  function switchTab(safe) {
    document.querySelectorAll('.modality-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    var panel = document.getElementById('panel-' + safe);
    if (panel) panel.style.display = 'flex';
    var tab = document.getElementById('tab-' + safe);
    if (tab) tab.classList.add('active');
  }
  function showSection(safe, id) {
    var panel = document.getElementById('panel-' + safe);
    if (!panel) return;
    panel.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    panel.querySelectorAll('.sidebar-item, .sidebar-overview').forEach(s => s.classList.remove('active'));
    var el = document.getElementById(safe + '-sec-' + id);
    if (el) el.classList.add('active');
    panel.querySelectorAll('.sidebar-item, .sidebar-overview').forEach(function(item) {
      var oc = item.getAttribute('onclick') || '';
      if (oc.includes("'" + safe + "'") && oc.includes("'" + id + "'")) item.classList.add('active');
    });
    var mainEl = document.getElementById('main-' + safe);
    if (mainEl) mainEl.scrollTop = 0;
  }
'''

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate combined LOINC/RSNA Radiology Playbook HTML.')
    parser.add_argument('--csv', '-c', help='Path to CSV file (default: data/LoincRsnaRadiologyPlaybook.csv)')
    parser.add_argument('--out', '-o', help='Output HTML path (default: output/LOINC_Radiology_All_Modalities.html)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    output_dir = script_dir / 'output'

    # Find CSV file
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = data_dir / 'LoincRsnaRadiologyPlaybook.csv'

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'LOINC_Radiology_All_Modalities.html'

    print(f"Loading {csv_path} ...")
    rows = load_csv(str(csv_path))
    print(f"Loaded {len(rows):,} rows.")

    modalities = get_modalities(rows)
    total_all  = sum(modalities.values())
    print(f"Found {len(modalities)} modalities, {total_all:,} total codes.\n")

    tab_buttons = ""
    panels_html = ""

    for i, (label, code_count) in enumerate(modalities.items()):
        safe    = label.replace('+','_').replace('.','_').replace('{','').replace('}','').replace(' ','_')
        active  = "active" if i == 0 else ""
        display = "flex"   if i == 0 else "none"

        print(f"  [{i+1}/{len(modalities)}] {label} ({code_count:,} codes)...")
        total_codes, part_data = extract_data(rows, label)
        sidebar_content, main_content = build_modality_content(label, total_codes, part_data)

        sidebar_ns = namespace(sidebar_content, safe)
        main_ns    = namespace(main_content,    safe)

        tab_buttons += f'''
      <button class="tab-btn {active}" onclick="switchTab('{safe}')" id="tab-{safe}">
        <span class="tab-code">{label}</span>
        <span class="tab-sub">{code_count:,} codes</span>
      </button>'''

        panels_html += f'''
  <div id="panel-{safe}" class="modality-panel" style="display:{display}">
    <nav class="sidebar">
      <div class="sidebar-section">
        <h2>Overview</h2>
        <div class="sidebar-overview active" onclick="showSection('{safe}', 'overview')">Summary &amp; Stats</div>
      </div>
      {sidebar_ns}
    </nav>
    <main class="main" id="main-{safe}">
      {main_ns}
    </main>
  </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LOINC/RSNA Radiology Playbook — All Modalities</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>LOINC/RSNA Radiology Playbook <span class="badge">All Modalities</span></h1>
    <p>{len(modalities)} modalities &nbsp;·&nbsp; {total_all:,} total procedure codes &nbsp;·&nbsp; Select a tab to explore</p>
  </div>
</header>
<div class="tab-bar">{tab_buttons}</div>
<div class="panels-wrapper">{panels_html}</div>
<footer>LOINC/RSNA Radiology Playbook &nbsp;·&nbsp; Generated from {csv_path.name} &nbsp;·&nbsp; {total_all:,} total procedure codes</footer>
<script>{JS}</script>
</body>
</html>'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    size_kb = out_path.stat().st_size / 1024
    print(f"Size: {size_kb:.0f} KB")
    print("\nOpen in any browser — fully self-contained, no internet required.")

if __name__ == '__main__':
    main()
