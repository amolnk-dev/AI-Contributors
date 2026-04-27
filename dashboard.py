#!/usr/bin/env python3
"""Live swarm dashboard — auto-discovers VMs across all regions.

Usage:
    python dashboard.py
    python dashboard.py --port 8050 --refresh 90
"""

import argparse
import json
import os
import random
import subprocess
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
DASHBOARD_FILE = TASK_DIR / "dashboard.html"
PROJECT = "ai-nm26osl-1823"
REFRESH_SEC = 90
BASELINE = 90.10
START_TIME = time.time()

# Cache
FLEET_DATA = {"vms": [], "experiments": [], "last_update": "", "total_vms": 0,
              "total_experiments": 0, "total_kept": 0, "best_metric": 0, "best_vm": "",
              "per_region": {}, "per_focus": {}, "orchestrator": ""}


def discover_vms():
    """Auto-discover all ainm-astar-* VMs across all regions."""
    try:
        r = subprocess.run(
            ["gcloud", "compute", "instances", "list", f"--project={PROJECT}",
             "--filter=name~ainm-astar AND status:RUNNING",
             "--format=csv[no-heading](name,zone)"],
            capture_output=True, text=True, timeout=30
        )
        vms = []
        for line in r.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                vms.append((parts[0], parts[1]))
        return vms
    except Exception:
        return []


def sample_vm(vm_name, zone):
    """Get results from one VM."""
    try:
        r = subprocess.run(
            ["gcloud", "compute", "ssh", vm_name, f"--zone={zone}", f"--project={PROJECT}",
             "--ssh-flag=-o StrictHostKeyChecking=no",
             "--command=cat /tmp/astar/results.tsv 2>/dev/null; echo ===LOG===; tail -2 /tmp/astar/swarm.log 2>/dev/null"],
            capture_output=True, text=True, timeout=12
        )
        if r.returncode != 0:
            return None

        parts = r.stdout.split("===LOG===")
        tsv = parts[0].strip() if parts else ""
        log = parts[1].strip() if len(parts) > 1 else ""

        experiments = []
        for line in tsv.split("\n")[1:]:  # skip header
            cols = line.split("\t")
            if len(cols) >= 6:
                try:
                    experiments.append({
                        "val_metric": float(cols[1]),
                        "status": cols[4],
                        "description": cols[5][:60],
                    })
                except (ValueError, IndexError):
                    pass

        kept = [e for e in experiments if e["status"] == "keep"]
        best = max((e["val_metric"] for e in kept), default=0)

        return {
            "runs": len(experiments),
            "kept": len(kept),
            "best": best,
            "last_log": log[:100],
            "experiments": experiments,
        }
    except Exception:
        return None


def update_data():
    """Discover VMs, sample subset, update cached data."""
    all_vms = discover_vms()
    total_vms = len(all_vms)

    # Sample ~30 VMs for results (don't SSH to all 500+)
    sample_size = min(30, total_vms)
    sampled = random.sample(all_vms, sample_size) if all_vms else []

    # Group by region
    per_region = {}
    for _, zone in all_vms:
        region = zone.rsplit("-", 1)[0]
        per_region[region] = per_region.get(region, 0) + 1

    total_experiments = 0
    total_kept = 0
    best_metric = 0
    best_vm = ""
    all_exps = []
    vm_results = []

    for vm_name, zone in sampled:
        data = sample_vm(vm_name, zone)
        if data:
            total_experiments += data["runs"]
            total_kept += data["kept"]
            if data["best"] > best_metric:
                best_metric = data["best"]
                best_vm = vm_name
            vm_results.append({"name": vm_name, "zone": zone, **data})
            for e in data["experiments"]:
                e["vm"] = vm_name
                all_exps.append(e)

    # Extrapolate from sample
    if sample_size > 0 and total_vms > 0:
        scale = total_vms / sample_size
        total_experiments = int(total_experiments * scale)
        total_kept = int(total_kept * scale)

    # Read orchestrator log
    orch_log = ""
    orch_file = TASK_DIR / "orchestrator_live.log"
    if orch_file.exists():
        lines = orch_file.read_text().strip().split("\n")
        orch_log = "\n".join(lines[-5:])

    FLEET_DATA.update({
        "total_vms": total_vms,
        "total_experiments": total_experiments,
        "total_kept": total_kept,
        "best_metric": best_metric,
        "best_vm": best_vm,
        "per_region": per_region,
        "vms": vm_results,
        "experiments": sorted(all_exps, key=lambda x: -x["val_metric"])[:50],
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "orchestrator": orch_log,
    })


def generate_html():
    d = FLEET_DATA
    now = d["last_update"]
    uptime = (time.time() - START_TIME) / 3600
    exp_per_hr = d["total_experiments"] / max(uptime, 0.01)

    # Region cards
    region_cards = ""
    for region in sorted(d["per_region"].keys()):
        count = d["per_region"][region]
        region_cards += f'<div class="region-card"><span class="region-name">{region}</span><span class="region-count">{count} VMs</span></div>'

    # VM table (sampled)
    vm_rows = ""
    for v in sorted(d["vms"], key=lambda x: -x["best"])[:20]:
        best_str = f'{v["best"]:.4f}' if v["best"] > 0 else "—"
        region = v["zone"].rsplit("-", 1)[0]
        sc = "#3fb950" if v["best"] > BASELINE else "#c9d1d9"
        vm_rows += f'<tr><td>{v["name"]}</td><td>{region}</td><td>{v["runs"]}</td><td>{v["kept"]}</td><td style="color:{sc};font-weight:bold">{best_str}</td><td class="desc">{v["last_log"][:50]}</td></tr>'

    # Top experiments
    exp_rows = ""
    for i, e in enumerate(d["experiments"][:30]):
        delta = e["val_metric"] - BASELINE
        dc = "#3fb950" if delta > 0 else "#f44336"
        sc = "#3fb950" if e["status"] == "keep" else "#8b949e"
        exp_rows += f'<tr><td>{i+1}</td><td>{e.get("vm","?")}</td><td style="font-weight:bold">{e["val_metric"]:.4f}</td><td style="color:{dc}">{delta:+.4f}</td><td style="color:{sc}">{e["status"]}</td><td class="desc">{e["description"]}</td></tr>'

    # Orchestrator log
    orch_html = d["orchestrator"].replace("\n", "<br>") if d["orchestrator"] else "Not running"

    # Improvements feed
    improvements_feed = ""
    kept_exps = [e for e in d["experiments"] if e["status"] == "keep" and e["val_metric"] > 0]
    kept_exps.sort(key=lambda x: -x["val_metric"])
    for e in kept_exps[:20]:
        delta = e["val_metric"] - BASELINE
        cls = "above" if delta > 0 else "below"
        dc = "#3fb950" if delta > 0 else "#f0883e"
        improvements_feed += f'''<div class="feed-item {cls}">
            <span class="feed-vm">{e.get("vm","?")}</span>
            <span class="feed-desc">{e["description"]}</span>
            <span class="feed-metric" style="color:{dc}">{e["val_metric"]:.4f}</span>
            <span class="feed-delta" style="color:{dc}">{delta:+.4f}</span>
        </div>'''
    if not improvements_feed:
        improvements_feed = '<div class="feed-item">No improvements found yet — VMs running baselines...</div>'

    # Chart data
    kept_data = json.dumps([{"v": e["val_metric"], "d": e.get("description", "")[:30]}
                            for e in d["experiments"] if e["status"] == "keep"][:40])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Astar Swarm — {d["total_vms"]} VMs</title>
    <meta http-equiv="refresh" content="{REFRESH_SEC}">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0d1117; color:#c9d1d9; }}
        .header {{ background:linear-gradient(135deg,#161b22,#0d1117); padding:20px 30px; border-bottom:1px solid #30363d; }}
        .header h1 {{ color:#58a6ff; font-size:1.4em; }}
        .header .sub {{ color:#8b949e; font-size:0.8em; margin-top:3px; }}
        .stats {{ display:flex; gap:0; border-bottom:1px solid #30363d; }}
        .stat {{ flex:1; padding:16px; text-align:center; border-right:1px solid #30363d; }}
        .stat:last-child {{ border-right:none; }}
        .stat .value {{ font-size:2em; font-weight:700; color:#58a6ff; }}
        .stat.best .value {{ color:#3fb950; }}
        .stat .label {{ color:#8b949e; font-size:0.7em; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }}
        .content {{ padding:15px 25px; }}
        .section {{ margin-bottom:20px; }}
        .section h2 {{ color:#c9d1d9; font-size:1em; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #21262d; }}
        .regions {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:15px; }}
        .region-card {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:8px 14px; display:flex; gap:8px; align-items:center; }}
        .region-name {{ color:#8b949e; font-size:0.8em; }}
        .region-count {{ color:#58a6ff; font-weight:600; font-size:0.9em; }}
        table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:6px; overflow:hidden; font-size:0.85em; }}
        th {{ background:#21262d; color:#8b949e; text-align:left; padding:7px 10px; font-size:0.75em; text-transform:uppercase; }}
        td {{ padding:6px 10px; border-top:1px solid #21262d; }}
        td.desc {{ color:#8b949e; font-size:0.8em; max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
        tr:hover {{ background:#1c2128; }}
        canvas {{ background:#161b22; border-radius:6px; border:1px solid #30363d; width:100%; }}
        .orch {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; font-size:0.75em; color:#8b949e; font-family:monospace; line-height:1.5; }}
        .feed {{ display:flex; flex-direction:column; gap:6px; }}
        .feed-item {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; }}
        .feed-item.above {{ border-left:3px solid #3fb950; }}
        .feed-item.below {{ border-left:3px solid #f0883e; }}
        .feed-desc {{ color:#c9d1d9; font-size:0.85em; flex:1; }}
        .feed-metric {{ font-weight:700; font-size:0.95em; min-width:80px; text-align:right; }}
        .feed-delta {{ font-size:0.8em; min-width:70px; text-align:right; margin-left:10px; }}
        .feed-vm {{ color:#8b949e; font-size:0.75em; min-width:100px; }}
        .footer {{ color:#484f58; font-size:0.7em; padding:10px 25px; border-top:1px solid #21262d; }}
        .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:15px; }}
        @media(max-width:900px) {{ .stats {{ flex-wrap:wrap; }} .stat {{ min-width:33%; }} .two-col {{ grid-template-columns:1fr; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Astar Island — Autoresearch Swarm</h1>
        <div class="sub">Karpathy Protocol + Gemini 3.1 Flash | {now} | Refreshes every {REFRESH_SEC}s</div>
    </div>

    <div class="stats">
        <div class="stat"><div class="value">{d["total_vms"]}</div><div class="label">VMs Running</div></div>
        <div class="stat"><div class="value">{d["total_experiments"]:,}</div><div class="label">Experiments</div></div>
        <div class="stat"><div class="value">{d["total_kept"]}</div><div class="label">Improvements</div></div>
        <div class="stat best"><div class="value">{d["best_metric"]:.4f}</div><div class="label">Best WAVG</div></div>
        <div class="stat"><div class="value">{BASELINE:.2f}</div><div class="label">Baseline</div></div>
        <div class="stat"><div class="value">{exp_per_hr:,.0f}</div><div class="label">Exp/Hour</div></div>
        <div class="stat"><div class="value">{len(d["per_region"])}</div><div class="label">Regions</div></div>
    </div>

    <div class="content">
        <div class="section">
            <h2>Regions</h2>
            <div class="regions">{region_cards}</div>
        </div>

        <div class="section">
            <h2>Orchestrator</h2>
            <div class="orch">{orch_html}</div>
        </div>

        <div class="two-col">
            <div class="section">
                <h2>Top VMs (sampled {len(d["vms"])}/{d["total_vms"]})</h2>
                <table>
                    <tr><th>VM</th><th>Region</th><th>Runs</th><th>Kept</th><th>Best</th><th>Last</th></tr>
                    {vm_rows}
                </table>
            </div>
            <div class="section">
                <h2>Kept Improvements</h2>
                <canvas id="chart" height="250"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>Improvement Feed (kept experiments)</h2>
            <div class="feed">{improvements_feed}</div>
        </div>

        <div class="section">
            <h2>Top 30 Experiments (across fleet)</h2>
            <table>
                <tr><th>#</th><th>VM</th><th>WAVG</th><th>vs Baseline</th><th>Status</th><th>Description</th></tr>
                {exp_rows}
            </table>
        </div>
    </div>

    <script>
    const kept = {kept_data};
    const c = document.getElementById('chart');
    if (c && kept.length > 0) {{
        const ctx = c.getContext('2d');
        const W = c.width = c.offsetWidth * 2; const H = c.height = 500;
        const pad = {{l:55,r:15,t:15,b:50}};
        const maxV = Math.max(...kept.map(d=>d.v)) + 0.2;
        const minV = Math.min(...kept.map(d=>d.v), {BASELINE}) - 0.3;
        const barW = Math.max(6, (W-pad.l-pad.r)/kept.length - 3);

        ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
        for(let i=0;i<=4;i++) {{
            const y=pad.t+(H-pad.t-pad.b)*i/4;
            ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
            ctx.fillStyle='#484f58';ctx.font='18px monospace';
            ctx.fillText((maxV-(maxV-minV)*i/4).toFixed(2),2,y+5);
        }}
        const baseY=pad.t+(H-pad.t-pad.b)*(1-(({BASELINE}-minV)/(maxV-minV)));
        ctx.strokeStyle='#f0883e';ctx.setLineDash([6,4]);
        ctx.beginPath();ctx.moveTo(pad.l,baseY);ctx.lineTo(W-pad.r,baseY);ctx.stroke();
        ctx.setLineDash([]);ctx.fillStyle='#f0883e';ctx.font='16px sans-serif';
        ctx.fillText('baseline {BASELINE}',W-pad.r-160,baseY-6);

        kept.forEach((d,i) => {{
            const x=pad.l+i*(barW+3);
            const h=(H-pad.t-pad.b)*((d.v-minV)/(maxV-minV));
            ctx.fillStyle=d.v>{BASELINE}?'#3fb950':'#f0883e';
            ctx.fillRect(x,H-pad.b-h,barW,h);
        }});
    }}
    </script>

    <div class="footer">
        Dashboard auto-discovers VMs via <code>gcloud compute instances list</code> |
        Orchestrator: <code>python orchestrator_live.py</code> |
        Collect: <code>bash scripts/gcp/collect-astar.sh</code>
    </div>
</body>
</html>"""
    return html


def refresh():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing ({FLEET_DATA['total_vms']} VMs)...", flush=True)
    try:
        update_data()
        DASHBOARD_FILE.write_text(generate_html())
        print(f"  {FLEET_DATA['total_vms']} VMs | {FLEET_DATA['total_experiments']:,} exp | best={FLEET_DATA['best_metric']:.4f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)


def loop():
    while True:
        refresh()
        time.sleep(REFRESH_SEC)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/dashboard.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_FILE.read_bytes() if DASHBOARD_FILE.exists() else b"<h1>Loading...</h1>")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *a): pass


def main():
    global REFRESH_SEC
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--refresh", type=int, default=90)
    args = p.parse_args()
    REFRESH_SEC = args.refresh

    refresh()
    threading.Thread(target=loop, daemon=True).start()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nDashboard: http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
