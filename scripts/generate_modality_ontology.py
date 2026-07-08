"""
generate_modality_ontology.py
-----------------------------
Generates a self-contained interactive HTML ontology report
for any modality in the LOINC/RSNA Radiology Playbook CSV.

Usage:
    python3 scripts/generate_modality_ontology.py --modality CT
    python3 scripts/generate_modality_ontology.py --modality MR --csv data/custom.csv
    python3 scripts/generate_modality_ontology.py --list          # list all available modalities
    python3 scripts/generate_modality_ontology.py --all           # generate all modalities
    python3 scripts/generate_modality_ontology.py --all --out /path/to/dir

Output:
    output/<Modality>_Ontology.html  (default output directory, or custom --out path)
    CSV is loaded from data/LoincRsnaRadiologyPlaybook.csv by default
"""

import csv, sys, os, argparse
from collections import defaultdict
from pathlib import Path

# ── Branch config ─────────────────────────────────────────────────────────────
BRANCHES = {
    "modality":       {"label": "Modality",       "color": "#1d9e75", "border": "#0f6e56", "text": "#e1f5ee", "sub": "#9fe1cb", "tag": "teal",   "bar": "teal"},
    "anatomy":        {"label": "Anatomy",         "color": "#7f77dd", "border": "#534ab7", "text": "#eeedfe", "sub": "#afa9ec", "tag": "purple", "bar": "purple"},
    "timing":         {"label": "Protocol",        "color": "#ba7517", "border": "#854f0b", "text": "#faeeda", "sub": "#fac775", "tag": "amber",  "bar": "amber"},
    "pharmaceutical": {"label": "Pharmaceutical",  "color": "#d85a30", "border": "#993c1d", "text": "#faece7", "sub": "#f5c4b3", "tag": "coral",  "bar": "coral"},
    "guidance":       {"label": "Guidance",        "color": "#5f5e5a", "border": "#444441", "text": "#f1efe8", "sub": "#b4b2a9", "tag": "gray",   "bar": "gray"},
}

PART_TYPE_META = {
    "Rad.Modality.Modality Subtype":             {"label": "Modality Subtype",         "branch": "modality",       "group": "Modality"},
    "Rad.Anatomic Location.Region Imaged":       {"label": "Region Imaged",            "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Imaging Focus":       {"label": "Imaging Focus",            "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Laterality":          {"label": "Laterality",               "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Anatomic Location.Laterality.Presence": {"label": "Laterality Presence",      "branch": "anatomy",        "group": "Anatomy"},
    "Rad.Timing":                                {"label": "Timing / Contrast",        "branch": "timing",         "group": "Protocol"},
    "Rad.View.Aggregation":                      {"label": "View Aggregation",         "branch": "timing",         "group": "Protocol"},
    "Rad.View.View type":                        {"label": "View Type",                "branch": "timing",         "group": "Protocol"},
    "Rad.View.View Type":                        {"label": "View Type",                "branch": "timing",         "group": "Protocol"},
    "Rad.Reason for Exam":                       {"label": "Reason for Exam",          "branch": "timing",         "group": "Protocol"},
    "Rad.Maneuver.Maneuver type":                {"label": "Maneuver",                 "branch": "timing",         "group": "Protocol"},
    "Rad.Maneuver.Maneuver Type":                {"label": "Maneuver",                 "branch": "timing",         "group": "Protocol"},
    "Rad.Subject":                               {"label": "Subject",                  "branch": "timing",         "group": "Protocol"},
    "Rad.Pharmaceutical.Substance Given":        {"label": "Pharmaceutical Substance", "branch": "pharmaceutical", "group": "Pharmaceutical"},
    "Rad.Pharmaceutical.Route":                  {"label": "Pharmaceutical Route",     "branch": "pharmaceutical", "group": "Pharmaceutical"},
    "Rad.Guidance for.Action":                   {"label": "Guidance Action",          "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Object":                   {"label": "Guidance Object",          "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Approach":                 {"label": "Guidance Approach",        "branch": "guidance",       "group": "Interventional"},
    "Rad.Guidance for.Presence":                 {"label": "Guidance Presence",        "branch": "guidance",       "group": "Interventional"},
}

GROUP_ORDER = ["Modality", "Anatomy", "Protocol", "Pharmaceutical", "Interventional"]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def get_available_modalities(rows):
    counts = defaultdict(set)
    for row in rows:
        if row['PartTypeName'] == 'Rad.Modality.Modality Type':
            counts[row['PartName']].add(row['LoincNumber'])
    return {m: len(c) for m, c in sorted(counts.items(), key=lambda x: -len(x[1]))}

def extract_modality_data(rows, modality):
    loinc_set = set()
    for row in rows:
        if row['PartTypeName'] == 'Rad.Modality.Modality Type' and row['PartName'] == modality:
            loinc_set.add(row['LoincNumber'])
    if not loinc_set:
        return None, None
    part_data = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row['LoincNumber'] in loinc_set:
            pt, pn = row['PartTypeName'], row['PartName']
            if pt and pn and pt != 'Rad.Modality.Modality Type':
                part_data[pt][pn] += 1
    return len(loinc_set), part_data

# ── SVG tree diagram ──────────────────────────────────────────────────────────

def build_svg_tree(modality, total_codes, part_data):
    """
    Dynamically build the attribute tree SVG for any modality.
    Layout: root at top, one branch node per branch group, child nodes below.
    """

    # Collect branch summaries
    branch_nodes = {}   # branch_key -> list of (label, count)
    for pt, vals in part_data.items():
        if pt not in PART_TYPE_META:
            continue
        meta = PART_TYPE_META[pt]
        b = meta['branch']
        if b not in branch_nodes:
            branch_nodes[b] = []
        branch_nodes[b].append((meta['label'], len(vals), sum(vals.values())))

    # Only show branches that have data
    active_branches = [b for b in ["modality","anatomy","timing","pharmaceutical","guidance"] if b in branch_nodes]
    n = len(active_branches)

    # SVG layout constants
    W = 840
    ROOT_CX = W // 2
    ROOT_Y = 30
    ROOT_W, ROOT_H = 200, 50
    BRANCH_Y = 150
    BRANCH_W, BRANCH_H = 150, 50
    CHILD_Y = 260
    CHILD_H = 40

    # Space branches evenly
    margin = 40
    usable = W - 2 * margin
    spacing = usable / max(n - 1, 1) if n > 1 else 0
    branch_cx = [int(margin + i * spacing) for i in range(n)]

    lines = []
    nodes = []

    # Root node
    rx = ROOT_CX - ROOT_W // 2
    lines.append(f'''
    <g class="node" onclick="showSection('overview')" style="cursor:pointer">
      <rect x="{rx}" y="{ROOT_Y}" width="{ROOT_W}" height="{ROOT_H}" rx="10"
            fill="#185fa5" stroke="#0c447c" stroke-width="1"/>
      <text x="{ROOT_CX}" y="{ROOT_Y+18}" text-anchor="middle" dominant-baseline="central"
            fill="#e6f1fb" font-size="15" font-weight="500" font-family="-apple-system,sans-serif">{modality} Modality</text>
      <text x="{ROOT_CX}" y="{ROOT_Y+36}" text-anchor="middle" dominant-baseline="central"
            fill="#85b7eb" font-size="11" font-family="-apple-system,sans-serif">{total_codes:,} LOINC codes</text>
    </g>''')

    max_child_y = CHILD_Y

    for i, bkey in enumerate(active_branches):
        bcx = branch_cx[i]
        bx = bcx - BRANCH_W // 2
        bc = BRANCHES[bkey]
        items = branch_nodes[bkey]
        total_vals = sum(cnt for _, cnt, _ in items)

        # Subtitle: list labels
        subtitle_parts = [lbl for lbl, _, _ in items[:3]]
        subtitle = " · ".join(subtitle_parts)
        if len(subtitle) > 28:
            subtitle = subtitle[:25] + "…"

        # Line from root to branch
        lines.append(f'<line x1="{ROOT_CX}" y1="{ROOT_Y+ROOT_H}" x2="{bcx}" y2="{BRANCH_Y}" '
                     f'stroke="#b4b2a9" stroke-width="0.8" marker-end="url(#arr)"/>')

        # Branch node - find first part type for this branch for section link
        sec_id = None
        for pt, meta in PART_TYPE_META.items():
            if meta['branch'] == bkey and pt in part_data:
                sec_id = pt.replace('.', '-').replace(' ', '-').lower()
                break

        onclick = f"showSection('{sec_id}')" if sec_id else "showSection('overview')"
        lines.append(f'''
    <g class="node" onclick="{onclick}" style="cursor:pointer">
      <rect x="{bx}" y="{BRANCH_Y}" width="{BRANCH_W}" height="{BRANCH_H}" rx="8"
            fill="{bc['color']}" stroke="{bc['border']}" stroke-width="0.8"/>
      <text x="{bcx}" y="{BRANCH_Y+17}" text-anchor="middle" dominant-baseline="central"
            fill="{bc['text']}" font-size="13" font-weight="500" font-family="-apple-system,sans-serif">{bc['label']}</text>
      <text x="{bcx}" y="{BRANCH_Y+34}" text-anchor="middle" dominant-baseline="central"
            fill="{bc['sub']}" font-size="10" font-family="-apple-system,sans-serif">{subtitle}</text>
    </g>''')

        # Child nodes - one per part type in this branch
        children = [(pt, vals) for pt, vals in part_data.items()
                    if pt in PART_TYPE_META and PART_TYPE_META[pt]['branch'] == bkey]
        children.sort(key=lambda x: PART_TYPE_META[x[0]]['label'])

        nc = len(children)
        if nc == 0:
            continue

        child_w = min(110, max(70, BRANCH_W // nc + 10))
        total_cw = nc * child_w + (nc - 1) * 8
        child_start = bcx - total_cw // 2

        for j, (pt, vals) in enumerate(children):
            clabel = PART_TYPE_META[pt]['label']
            # Shorten label
            short = clabel.replace('Imaging ', '').replace('Pharmaceutical ', '').replace('Guidance ', '').replace('Laterality ', 'Lat. ').replace(' Imaged','').replace(' Aggregation','')
            if len(short) > 14:
                short = short[:13] + '…'
            ccount = len(vals)
            ccx = child_start + j * (child_w + 8) + child_w // 2
            cx_left = child_start + j * (child_w + 8)
            child_sec = pt.replace('.', '-').replace(' ', '-').lower()

            # Line from branch to child
            lines.append(f'<line x1="{bcx}" y1="{BRANCH_Y+BRANCH_H}" x2="{ccx}" y2="{CHILD_Y}" '
                         f'stroke="#b4b2a9" stroke-width="0.6" marker-end="url(#arr)"/>')

            lines.append(f'''
    <g class="node" onclick="showSection('{child_sec}')" style="cursor:pointer">
      <rect x="{cx_left}" y="{CHILD_Y}" width="{child_w}" height="{CHILD_H}" rx="6"
            fill="{bc['sub']}" stroke="{bc['color']}" stroke-width="0.5"/>
      <text x="{ccx}" y="{CHILD_Y+14}" text-anchor="middle" dominant-baseline="central"
            fill="{bc['border']}" font-size="11" font-weight="500" font-family="-apple-system,sans-serif">{short}</text>
      <text x="{ccx}" y="{CHILD_Y+28}" text-anchor="middle" dominant-baseline="central"
            fill="{bc['border']}" font-size="10" font-family="-apple-system,sans-serif">{ccount} values</text>
    </g>''')

            max_child_y = max(max_child_y, CHILD_Y + CHILD_H)

    # Legend
    legend_y = max_child_y + 40
    lines.append(f'<line x1="20" y1="{legend_y}" x2="{W-20}" y2="{legend_y}" stroke="#e8e7e1" stroke-width="0.5"/>')
    lines.append(f'<text x="{ROOT_CX}" y="{legend_y+20}" text-anchor="middle" '
                 f'fill="#888780" font-size="11" font-family="-apple-system,sans-serif">Click any node to see full value list</text>')

    lx = 30
    for bkey in active_branches:
        bc = BRANCHES[bkey]
        lines.append(f'<rect x="{lx}" y="{legend_y+36}" width="12" height="12" rx="2" fill="{bc["color"]}"/>')
        lines.append(f'<text x="{lx+16}" y="{legend_y+42}" fill="#5f5e5a" font-size="11" '
                     f'dominant-baseline="central" font-family="-apple-system,sans-serif">{bc["label"]}</text>')
        lx += 130

    svg_h = legend_y + 70

    svg = f'''<svg width="100%" viewBox="0 0 {W} {svg_h}" role="img">
  <title>{modality} Modality Ontology Tree</title>
  <desc>Hierarchical attribute tree for {modality} in the LOINC/RSNA Radiology Playbook</desc>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#b4b2a9" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  {''.join(lines)}
</svg>'''
    return svg

# ── HTML sections ─────────────────────────────────────────────────────────────

def bar_html(count, max_count, bar_cls):
    pct = max(1, int((count / max_count) * 100)) if max_count else 0
    return f'<div class="bar-bg"><div class="bar-fill bar-{bar_cls}" style="width:{pct}%"></div></div>'

def section_html(part_type, values_dict, meta):
    branch = meta['branch']
    bc = BRANCHES[branch]
    sorted_vals = sorted(values_dict.items(), key=lambda x: -x[1])
    max_count = sorted_vals[0][1] if sorted_vals else 1
    total = sum(v for _, v in sorted_vals)
    TOP_N = 60

    rows_html = ""
    for val, cnt in sorted_vals[:TOP_N]:
        rows_html += f"""
        <tr>
          <td><span class="tag {bc['tag']}">{val}</span></td>
          <td style="text-align:right;font-variant-numeric:tabular-nums">{cnt:,}</td>
          <td class="bar-cell">{bar_html(cnt, max_count, bc['bar'])}</td>
        </tr>"""

    remainder = len(sorted_vals) - TOP_N
    if remainder > 0:
        rows_html += f'<tr><td colspan="3" style="color:#888780;font-size:12px;padding-top:8px">+ {remainder:,} more values not shown</td></tr>'

    sec_id = part_type.replace('.', '-').replace(' ', '-').lower()
    return f"""
    <div id="sec-{sec_id}" class="section">
      <div class="diagram-card">
        <div class="card-header" style="background:{bc['color']};border-radius:8px 8px 0 0;padding:16px 20px;margin:-24px -24px 20px;">
          <h2 style="color:{bc['text']};margin:0;font-size:16px;font-weight:500">{meta['label']}</h2>
          <p style="color:{bc['sub']};margin:4px 0 0;font-size:12px">{len(sorted_vals):,} distinct values · {total:,} total occurrences</p>
        </div>
        <table>
          <tr><th>Value</th><th style="text-align:right">Count</th><th class="bar-cell">Frequency</th></tr>
          {rows_html}
        </table>
      </div>
    </div>"""

def sidebar_item_html(part_type, values_dict, meta):
    sec_id = part_type.replace('.', '-').replace(' ', '-').lower()
    return (f'<div class="sidebar-item" onclick="showSection(\'{sec_id}\')">'
            f'{meta["label"]} <span class="count">{len(values_dict)}</span></div>')

# ── Full HTML ─────────────────────────────────────────────────────────────────

def generate_html(modality, total_codes, part_data):
    groups = defaultdict(list)
    for pt, vals in sorted(part_data.items()):
        if pt in PART_TYPE_META:
            meta = PART_TYPE_META[pt]
            groups[meta['group']].append((pt, vals, meta))

    # Sidebar
    sidebar_html = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        sidebar_html += f'<div class="sidebar-section"><h2>{group}</h2>'
        for pt, vals, meta in groups[group]:
            sidebar_html += sidebar_item_html(pt, vals, meta)
        sidebar_html += '</div>'

    # Detail sections
    sections_html = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        for pt, vals, meta in groups[group]:
            sections_html += section_html(pt, vals, meta)

    # Stats
    focus_count = len(part_data.get("Rad.Anatomic Location.Imaging Focus", {}))
    subtype_count = len(part_data.get("Rad.Modality.Modality Subtype", {}))
    guidance_count = len(part_data.get("Rad.Guidance for.Action", {}))
    region_count = len(part_data.get("Rad.Anatomic Location.Region Imaged", {}))

    # Summary table rows
    summary_rows = ""
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        for pt, vals, meta in groups[group]:
            bc = BRANCHES[meta['branch']]
            sec_id = pt.replace('.', '-').replace(' ', '-').lower()
            total_occ = sum(vals.values())
            summary_rows += f"""
          <tr style="cursor:pointer" onclick="showSection('{sec_id}')">
            <td><span class="tag {bc['tag']}">{meta['label']}</span></td>
            <td style="color:#888780;font-size:12px">{group}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums">{len(vals):,}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums;color:#888780">{total_occ:,}</td>
          </tr>"""

    svg_tree = build_svg_tree(modality, total_codes, part_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{modality} Modality Ontology — LOINC/RSNA Radiology Playbook</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f4f0; color: #2c2c2a; min-height: 100vh; }}
  header {{ background: #1a1a18; color: #f0efe9; padding: 20px 32px; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }}
  header h1 {{ font-size: 18px; font-weight: 500; }}
  header p  {{ font-size: 12px; color: #9c9a92; margin-top:3px; }}
  .badge {{ background: #185fa5; color: #e6f1fb; font-size: 11px; padding: 2px 10px; border-radius: 20px; margin-left:8px; vertical-align:middle; }}
  .mod-badge {{ background: #3c3489; color: #eeedfe; font-size: 15px; font-weight: 600; padding: 5px 16px; border-radius: 6px; white-space:nowrap; }}
  .layout {{ display: flex; height: calc(100vh - 72px); overflow: hidden; }}
  .sidebar {{ width: 270px; min-width: 270px; background: #fff; border-right: 1px solid #d3d1c7; overflow-y: auto; }}
  .sidebar-section {{ border-bottom: 1px solid #e8e7e1; }}
  .sidebar-section h2 {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #888780; padding: 12px 16px 4px; }}
  .sidebar-item {{ display:flex; justify-content:space-between; align-items:center; padding:8px 16px; font-size:13px; cursor:pointer; border-left:3px solid transparent; transition:background 0.1s; }}
  .sidebar-item:hover, .sidebar-overview:hover {{ background: #f1efe8; }}
  .sidebar-item.active, .sidebar-overview.active {{ background:#e6f1fb; border-left-color:#185fa5; color:#0c447c; font-weight:500; }}
  .sidebar-overview {{ padding:8px 16px; font-size:13px; cursor:pointer; border-left:3px solid transparent; display:block; }}
  .count {{ font-size:11px; color:#888780; background:#f1efe8; padding:1px 7px; border-radius:10px; }}
  .sidebar-item.active .count {{ background:#b5d4f4; color:#0c447c; }}
  .main {{ flex:1; overflow-y:auto; padding:24px 28px; }}
  .stats-bar {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px; }}
  .stat {{ background:#fff; border:1px solid #d3d1c7; border-radius:8px; padding:14px 18px; flex:1; min-width:100px; }}
  .stat .val {{ font-size:26px; font-weight:500; color:#185fa5; }}
  .stat .lbl {{ font-size:11px; color:#888780; margin-top:3px; }}
  .diagram-card {{ background:#fff; border:1px solid #d3d1c7; border-radius:12px; padding:24px; max-width:880px; margin-bottom:22px; }}
  .diagram-card h2 {{ font-size:15px; font-weight:500; margin-bottom:4px; }}
  .diagram-card .subtitle {{ font-size:12px; color:#5f5e5a; margin-bottom:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; padding:8px 12px; background:#f1efe8; font-weight:500; font-size:10px; text-transform:uppercase; letter-spacing:0.06em; color:#5f5e5a; }}
  td {{ padding:6px 12px; border-bottom:1px solid #f1efe8; vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  .bar-cell {{ width:160px; }}
  .bar-bg {{ background:#e8e7e1; border-radius:3px; height:7px; }}
  .bar-fill {{ height:7px; border-radius:3px; }}
  .bar-teal   {{ background:#1d9e75; }}
  .bar-purple {{ background:#7f77dd; }}
  .bar-amber  {{ background:#ba7517; }}
  .bar-coral  {{ background:#d85a30; }}
  .bar-gray   {{ background:#888780; }}
  .tag {{ display:inline-block; font-size:12px; padding:2px 9px; border-radius:10px; }}
  .tag.teal   {{ background:#e1f5ee; color:#085041; }}
  .tag.purple {{ background:#eeedfe; color:#3c3489; }}
  .tag.amber  {{ background:#faeeda; color:#633806; }}
  .tag.coral  {{ background:#faece7; color:#712b13; }}
  .tag.gray   {{ background:#f1efe8; color:#444441; }}
  .node {{ cursor:pointer; }}
  .node rect {{ transition:opacity 0.15s; }}
  .node:hover rect {{ opacity:0.82; }}
  .section {{ display:none; }}
  .section.active {{ display:block; }}
  footer {{ text-align:center; font-size:11px; color:#888780; padding:14px; border-top:1px solid #e8e7e1; background:#fff; }}
</style>
</head>
<body>
<header>
  <span class="mod-badge">{modality}</span>
  <div>
    <h1>Modality Ontology <span class="badge">LOINC/RSNA Radiology Playbook</span></h1>
    <p>All attribute branches extracted from {total_codes:,} {modality} procedure codes</p>
  </div>
</header>
<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-section">
      <h2>Overview</h2>
      <div class="sidebar-overview active" onclick="showSection('overview')">Summary &amp; Stats</div>
    </div>
    {sidebar_html}
  </nav>
  <main class="main">
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
        <p class="subtitle">Click any row to see full value breakdown.</p>
        <table>
          <tr><th>Part type</th><th>Group</th><th style="text-align:right">Distinct values</th><th style="text-align:right">Total occurrences</th></tr>
          {summary_rows}
        </table>
      </div>
    </div>
    {sections_html}
  </main>
</div>
<footer>LOINC/RSNA Radiology Playbook &middot; {modality} Modality &middot; Generated from LoincRsnaRadiologyPlaybook.csv &middot; {total_codes:,} procedure codes</footer>
<script>
  function showSection(id) {{
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.sidebar-item, .sidebar-overview').forEach(s => s.classList.remove('active'));
    var el = document.getElementById('sec-' + id);
    if (el) el.classList.add('active');
    document.querySelectorAll('.sidebar-item, .sidebar-overview').forEach(function(item) {{
      if ((item.getAttribute('onclick') || '').includes("'" + id + "'")) item.classList.add('active');
    }});
    document.querySelector('.main').scrollTop = 0;
  }}
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate HTML ontology report for a LOINC/RSNA radiology modality.')
    parser.add_argument('--modality', '-m', type=str, help='Modality code (e.g. CT, MR, US, XR)')
    parser.add_argument('--csv', '-c', type=str, help='Path to CSV file (default: data/LoincRsnaRadiologyPlaybook.csv)')
    parser.add_argument('--list', '-l', action='store_true', help='List all available modalities')
    parser.add_argument('--all', '-a', action='store_true', help='Generate reports for all modalities')
    parser.add_argument('--out', '-o', type=str, help='Output directory (default: output/)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    output_dir = Path(args.out) if args.out else script_dir / 'output'

    # Find CSV file
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = data_dir / 'LoincRsnaRadiologyPlaybook.csv'

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}"); sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    rows = load_csv(str(csv_path))
    print(f"Loaded {len(rows):,} rows.")
    modalities = get_available_modalities(rows)

    if args.list:
        print(f"\nAvailable modalities:\n  {'Code':<25} {'LOINC codes':>12}")
        print(f"  {'-'*25} {'-'*12}")
        for m, cnt in modalities.items():
            print(f"  {m:<25} {cnt:>12,}")
        sys.exit(0)

    targets = list(modalities.keys()) if args.all else ([args.modality] if args.modality else [])
    if not targets:
        print("ERROR: specify --modality, --all, or --list"); sys.exit(1)

    for modality in targets:
        if modality not in modalities:
            print(f"SKIP: '{modality}' not found."); continue
        print(f"\nProcessing: {modality} ({modalities[modality]:,} codes)")
        total_codes, part_data = extract_modality_data(rows, modality)
        html = generate_html(modality, total_codes, part_data)
        safe = modality.replace('+','_').replace('.','_').replace('{','').replace('}','').replace(' ','_')
        out_path = output_dir / f"{safe}_Ontology.html"
        out_path.write_text(html, encoding='utf-8')
        print(f"  → {out_path}")

    print("\nDone!")

if __name__ == '__main__':
    main()
