"""torch.profiler traces of a few warm iterations of one workload.

``profile_workload`` writes two files to the output folder:
  <name>.trace.json   Chrome trace; open in https://ui.perfetto.dev or
                      chrome://tracing;
  <name>.kernels.txt  operators and kernels sorted by self GPU time.
It also returns a summary for the benchmark's JSON. Stages that run inside
``label(stage)`` show up in the trace as ranges named ``<name>/<stage>``,
e.g. full/1d and full/2d.

With ``stacks=True`` it also records Python call stacks and writes, for GPU
and for CPU time:
  <name>.stacks_gpu.txt / .stacks_cpu.txt   folded stacks (one line per
                      stack, "frame;frame;op microseconds"), for
                      https://www.speedscope.app or flamegraph.pl;
  <name>.flame_gpu.svg / .flame_cpu.svg     flamegraphs; open in a browser,
                      hover a frame for its time, click to zoom.
GPU time is attributed to the operator that launched each kernel. Under
torch.compile the stacks end at the compiled graph; run with --no-compile to
see time per module.

Call it after compilation and warm-up, so that the trace holds only the
steady state.
"""
import contextlib
import html
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function, schedule

_active = None  # name of the workload being profiled


def label(stage):
    """A named trace range while profiling; a no-op otherwise."""
    if _active is None:
        return contextlib.nullcontext()
    return record_function(f"{_active}/{stage}")


def _self_device_us(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def summarize_events(events, iters, top=15):
    """Per-iteration GPU kernel time, CPU operator time and the top kernels.

    Kernel totals count only GPU events, since the CPU operators that launch
    them report the same time again. ``cpu_ms_per_iter`` includes profiler
    overhead; compared with ``device_ms_per_iter`` it shows whether a
    workload is launch-bound.
    """
    # annotation ranges (record_function, ProfilerStep, compiled graphs) also
    # appear on the GPU timeline and overlap the kernels inside them
    kernels = [(e, _self_device_us(e)) for e in events
               if e.device_type == DeviceType.CUDA
               and not getattr(e, "is_user_annotation", False)]
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    total_dev = sum(t for _, t in kernels)
    kernels.sort(key=lambda kt: -kt[1])
    return {
        "iterations": iters,
        "device_ms_per_iter": total_dev / 1e3 / iters,
        "cpu_ms_per_iter": sum(e.self_cpu_time_total for e in cpu) / 1e3 / iters,
        "kernel_launches_per_iter": sum(e.count for e, _ in kernels) / iters,
        "top_kernels": [
            {
                "name": e.key,
                "device_ms_per_iter": t / 1e3 / iters,
                "calls_per_iter": e.count / iters,
                "share": t / total_dev if total_dev else None,
            }
            for e, t in kernels[:top]
        ],
    }


def _frames(evt):
    """Python stack (outermost first) of the nearest CPU ancestor that has
    one, then the op names from that ancestor down to ``evt``. Launch
    events such as cudaLaunchKernel, which carry the kernels' GPU time,
    have no stack of their own."""
    names, e = [], evt
    while e is not None:
        names.append(e.name)
        stack = getattr(e, "stack", None)
        if stack:
            return list(reversed(stack)) + names[::-1]
        e = getattr(e, "cpu_parent", None)
    return names[::-1]


_RUN_SPECIFIC = (
    # object addresses: "<built-in method to of Tensor object at 0x7f…>"
    (re.compile(r" at 0x[0-9a-fA-F]+"), ""),
    # compiled-graph hashes differ whenever shapes do (e.g. batch sizes)
    (re.compile(r"## Call CompiledFxGraph \w+ ##"), "## Call CompiledFxGraph ##"),
    # inductor's generated code: /tmp/torchinductor_<user>/<xx>/<hash>.py(N)
    (re.compile(r"^.*torchinductor_[^/]*/.*\.py\(\d+\): "), "<inductor code>: "),
)


def normalize_frame(frame):
    """A frame without run-specific parts, so that the same code lines up
    across runs in differential flamegraphs."""
    for pattern, repl in _RUN_SPECIFIC:
        frame = pattern.sub(repl, frame)
    return frame.replace(";", ":")


def fold_stacks(events, metric):
    """Folded stacks {"outer;...;inner;op": total} of the CPU events;
    ``metric(evt)`` gives an event's self time. GPU-side events are left
    out, since their time is already that of the CPU events that launched
    them."""
    folded = defaultdict(float)
    for evt in events:
        if evt.device_type != DeviceType.CPU:
            continue
        value = metric(evt)
        if value > 0:
            frames = [normalize_frame(f) for f in _frames(evt)]
            folded[";".join(frames)] += value
    return dict(folded)


def write_folded(folded, path):
    path.write_text("".join(f"{s} {round(v)}\n" for s, v in
                            sorted(folded.items()) if round(v) > 0))


def fmt_us(us):
    return f"{us / 1e3:,.2f} ms" if abs(us) >= 1e3 else f"{us:,.1f} µs"


def stack_tree(*folded):
    """Merged call tree of one or more folded-stack dicts; every node holds
    one total per input, ``node["v"][i]``."""
    def node(name):
        return {"name": name, "v": [0.0] * len(folded), "children": {}}

    root = node("all")
    for side, stacks in enumerate(folded):
        for stack, value in stacks.items():
            n = root
            n["v"][side] += value
            for frame in stack.split(";"):
                if frame not in n["children"]:
                    n["children"][frame] = node(frame)
                n = n["children"][frame]
                n["v"][side] += value
    return root


def _render(root, title, header, width_of, fill_of, tip_of, width=1600,
            row=17):
    """SVG flamegraph of a ``stack_tree``: root at the bottom, frame width
    ``width_of(node)``, hover for ``tip_of(node)``, click a frame to zoom
    into it (click the root to reset). Tips start with "<name> — "."""
    total = width_of(root) or 1.0
    rects, depth_max = [], 0

    def layout(n, x, depth):
        nonlocal depth_max
        depth_max = max(depth_max, depth)
        rects.append((x, width_of(n), depth, n))
        for child in sorted(n["children"].values(), key=lambda c: c["name"]):
            layout(child, x, depth + 1)
            x += width_of(child)

    layout(root, 0.0, 0)
    pad_top, pad_bottom = 40 + 14 * len(header), 10
    height = pad_top + (depth_max + 1) * row + pad_bottom
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="monospace" font-size="11">',
        "<style>rect{stroke:#fff;stroke-width:.5}g:hover rect{stroke:#000}"
        "text{pointer-events:none}</style>",
        f'<text x="10" y="20" font-size="14">{html.escape(title)}</text>',
    ]
    for i, line in enumerate(header):
        out.append(f'<text x="10" y="{34 + 14 * i}">{html.escape(line)}</text>')
    out.append(f'<text id="info" x="10" y="{34 + 14 * len(header)}" '
               f'fill="#555"> </text>')
    for x, w, depth, n in rects:
        if w <= 0:
            continue
        y = height - pad_bottom - (depth + 1) * row
        out.append(
            f'<g class="f" data-x="{x / total:.8f}" data-w="{w / total:.8f}" '
            f'data-d="{depth}"><title>{html.escape(tip_of(n))}</title>'
            f'<rect x="0" y="{y}" width="0" height="{row - 1}" '
            f'fill="{fill_of(n)}"/>'
            f'<text x="0" y="{y + row - 5}">{html.escape(n["name"])}</text></g>'
        )
    out.append(f"""<script><![CDATA[
var W = {width}, gs = document.querySelectorAll("g.f");
function draw(x0, w0, d0) {{
  gs.forEach(function (g) {{
    var x = (+g.dataset.x - x0) / w0 * W, w = +g.dataset.w / w0 * W;
    var r = g.querySelector("rect"), t = g.querySelector("text");
    var visible = w >= 0.5 && x + w > 0 && x < W && +g.dataset.d >= d0;
    g.style.display = visible ? "" : "none";
    r.setAttribute("x", x); r.setAttribute("width", Math.max(w - 0.5, 0));
    t.setAttribute("x", x + 3);
    var name = g.querySelector("title").textContent.split(" — ")[0];
    var n = Math.floor((w - 6) / 6.6);
    t.textContent = n < 3 ? "" : name.length > n
      ? name.slice(0, n - 2) + ".." : name;
  }});
}}
gs.forEach(function (g) {{
  g.addEventListener("click", function () {{
    draw(+g.dataset.x, +g.dataset.w, 0);
  }});
  g.addEventListener("mouseover", function () {{
    document.getElementById("info").textContent =
      g.querySelector("title").textContent;
  }});
}});
draw(0, 1, 0);
]]></script></svg>""")
    return "\n".join(out)


def flamegraph_svg(folded, title):
    """Flamegraph of folded stacks (values in µs) as a standalone SVG."""
    root = stack_tree(folded)
    total = root["v"][0] or 1.0

    def fill(n):
        return f"hsl({10 + zlib.crc32(n['name'].encode()) % 40},85%,60%)"

    def tip(n):
        v = n["v"][0]
        return f"{n['name']} — {fmt_us(v)} ({100 * v / total:.1f}%)"

    return _render(root, f"{title} — total {fmt_us(root['v'][0])}", [],
                   lambda n: n["v"][0], fill, tip)


def diff_flamegraph_svg(a, b, title, label_a="A", label_b="B",
                        width_from="b"):
    """Differential flamegraph of folded stacks ``a`` → ``b`` (same units).

    Frame widths follow ``b`` (or ``a``, which shows frames that ``b`` no
    longer has); colour shows the change: red slower in b, blue faster,
    stronger for larger changes relative to the largest one, grey unchanged.
    """
    root = stack_tree(a, b)
    side = 1 if width_from == "b" else 0
    deltas = []

    def collect(n):
        for c in n["children"].values():
            deltas.append(abs(c["v"][1] - c["v"][0]))
            collect(c)

    collect(root)
    scale = max(deltas, default=0.0) or 1.0

    def fill(n):
        d = (n["v"][1] - n["v"][0]) / scale
        if abs(d) < 0.01:
            return "hsl(0,0%,82%)"
        hue = 0 if d > 0 else 215
        return f"hsl({hue},{30 + 60 * abs(d):.0f}%,{78 - 23 * abs(d):.0f}%)"

    def tip(n):
        va, vb = n["v"]
        rel = "" if va == 0 else f" ({100 * (vb - va) / va:+.1f}%)"
        tag = " [only in " + (label_b if va == 0 else label_a) + "]" \
            if (va == 0) != (vb == 0) else ""
        return (f"{n['name']} — {label_a} {fmt_us(va)}, {label_b} "
                f"{fmt_us(vb)}, Δ {'+' if vb >= va else '-'}"
                f"{fmt_us(abs(vb - va))}{rel}{tag}")

    va, vb = root["v"]
    header = [
        f"{label_a}: {fmt_us(va)}   {label_b}: {fmt_us(vb)}   "
        f"Δ {vb - va:+,.1f} µs"
        + (f" ({100 * (vb - va) / va:+.1f}%)" if va else ""),
        f"widths: {label_b if side else label_a}; red = slower in "
        f"{label_b}, blue = faster, grey = unchanged",
    ]
    return _render(root, title, header, lambda n: n["v"][side], fill, tip)


def write_flamegraphs(events, out_dir, name):
    """Folded stacks and SVG flamegraphs of GPU and CPU self time."""
    paths = {}
    for kind, metric in (("gpu", _self_device_us),
                         ("cpu", lambda e: e.self_cpu_time_total)):
        folded = fold_stacks(events, metric)
        stacks = out_dir / f"{name}.stacks_{kind}.txt"
        svg = out_dir / f"{name}.flame_{kind}.svg"
        write_folded(folded, stacks)
        svg.write_text(flamegraph_svg(
            folded, f"{name}: {kind.upper()} self time by Python stack"))
        paths[f"stacks_{kind}"] = str(stacks)
        paths[f"flamegraph_{kind}"] = str(svg)
    return paths


def meta_path(stacks_path):
    """``<name>.meta.json`` next to ``<name>.stacks_<kind>.txt``."""
    p = Path(stacks_path)
    return p.with_name(p.name.split(".stacks_")[0] + ".meta.json")


def profile_workload(name, step, out_dir, iters=3, stacks=False,
                     predictions_per_iter=None):
    """Profiles ``iters`` calls of ``step()``, after one untimed call that
    starts the profiler up. Returns the summary plus the file paths;
    ``stacks`` also records Python stacks and writes flamegraphs.
    ``<name>.meta.json`` records the iterations and predictions per
    iteration, so spliceai2-flamediff can compare per prediction."""
    global _active
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{name}.trace.json"
    table = out_dir / f"{name}.kernels.txt"
    (out_dir / f"{name}.meta.json").write_text(json.dumps(
        {"iterations": iters, "predictions_per_iter": predictions_per_iter}
    ))
    result = {}

    def ready(prof):
        prof.export_chrome_trace(str(trace))
        averages = prof.key_averages()
        sort = ("self_device_time_total"
                if any(hasattr(e, "self_device_time_total") for e in averages)
                else "self_cuda_time_total")
        table.write_text(averages.table(sort_by=sort, row_limit=50))
        result.update(summarize_events(averages, iters))
        if stacks:
            result.update(write_flamegraphs(prof.events(), out_dir, name))

    torch.cuda.synchronize()
    _active = name
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=iters),
            on_trace_ready=ready, record_shapes=True, profile_memory=True,
            with_stack=stacks,
        ) as prof:
            for _ in range(1 + iters):
                step()
                torch.cuda.synchronize()
                prof.step()
    finally:
        _active = None
    return {"trace": str(trace), "kernels": str(table), **result}
