"""
visualization.py — all figures for the app (Plotly, interactive) and the report
(Matplotlib, static PNG). One shared neon-on-charcoal palette.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram

BG = "#0d0f14"
PANEL = "#151923"
GRID = "rgba(255,255,255,0.06)"
TEXT = "#e6e8ef"
MUTED = "#8a90a2"
VIOLET, CYAN, CORAL, AMBER, LIME, PINK = "#a78bfa", "#22d3ee", "#fb7185", "#fbbf24", "#a3e635", "#f472b6"
PALETTE = [VIOLET, CYAN, CORAL, AMBER, LIME, PINK]
MIX_COLOR = "#94a3b8"


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _layout(fig: go.Figure, height: int, title: str | None = None, **kw) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=(62 if fig.layout.annotations else 40) if title else (28 if fig.layout.annotations else 10), b=10), showlegend=kw.pop("showlegend", False),
        title=dict(text=title, x=0.01, font=dict(size=14, color=TEXT)) if title else None, **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def envelope(x: np.ndarray, sr: int, n_points: int = 1200):
    """Min/max envelope per bucket -> cheap but faithful waveform drawing for long audio."""
    n = len(x)
    b = max(1, n // n_points)
    m = (n // b) * b
    xr = x[:m].reshape(-1, b)
    t = (np.arange(xr.shape[0]) * b + b / 2) / sr
    return t, xr.min(axis=1), xr.max(axis=1)


def _wave_traces(x, sr, c, name):
    t, lo, hi = envelope(x, sr)
    return [
        go.Scatter(x=t, y=hi, mode="lines", line=dict(color=c, width=1), hoverinfo="skip", name=name),
        go.Scatter(x=t, y=lo, mode="lines", line=dict(color=c, width=1), fill="tonexty",
                   fillcolor=_rgba(c, 0.35), name=name, hovertemplate="%{x:.2f}s<extra>" + name + "</extra>"),
    ]


def waveform(x: np.ndarray, sr: int, c: str = VIOLET, height: int = 110, name: str = "") -> go.Figure:
    fig = go.Figure(_wave_traces(x, sr, c, name))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return _layout(fig, height)


def waveform_stack(signals, names, sr, colors=None, title=None, height_per=95) -> go.Figure:
    n = len(signals)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=names)
    for i, x in enumerate(signals):
        c = colors[i] if colors and i < len(colors) else color(i)
        for tr in _wave_traces(x, sr, c, names[i]):
            fig.add_trace(tr, row=i + 1, col=1)
        fig.update_yaxes(showticklabels=False, row=i + 1, col=1)
    fig.update_annotations(font=dict(size=12, color=MUTED), x=0, xanchor="left")
    fig.update_xaxes(title_text="time (s)", row=n, col=1)
    return _layout(fig, 50 + height_per * n, title)


def reveal_grid(S, X, Y, src_names, sr, mix_names=None, est_names=None) -> go.Figure:
    """Three columns: originals | mixtures | recovered (aligned) — the money shot."""
    n = max(len(S), len(X), len(Y))
    titles = []
    mix_names = mix_names or [f"Mic {i + 1}" for i in range(len(X))]
    est_names = est_names or [f"Recovered → {s}" for s in src_names]
    for i in range(n):
        titles += [src_names[i] if i < len(S) else "", mix_names[i] if i < len(X) else "",
                   est_names[i] if i < len(Y) else ""]
    fig = make_subplots(rows=n, cols=3, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.03,
                        subplot_titles=titles, column_titles=None)
    for i in range(n):
        if i < len(S):
            for tr in _wave_traces(S[i], sr, color(i), src_names[i]):
                fig.add_trace(tr, row=i + 1, col=1)
        if i < len(X):
            for tr in _wave_traces(X[i], sr, MIX_COLOR, mix_names[i]):
                fig.add_trace(tr, row=i + 1, col=2)
        if i < len(Y):
            for tr in _wave_traces(Y[i], sr, color(i), est_names[i]):
                fig.add_trace(tr, row=i + 1, col=3)
    fig.update_yaxes(showticklabels=False)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return _layout(fig, 70 + 120 * n)


def overlay(s: np.ndarray, y: np.ndarray, sr: int, start_s: float, dur_s: float = 0.05,
            c: str = VIOLET, name="source") -> go.Figure:
    """Sample-level overlay of the original and the aligned estimate on a short window."""
    i0, i1 = int(start_s * sr), int((start_s + dur_s) * sr)
    t = np.arange(i0, i1) / sr * 1000
    fig = go.Figure([
        go.Scatter(x=t, y=s[i0:i1], name="original", line=dict(color="rgba(255,255,255,0.85)", width=3)),
        go.Scatter(x=t, y=y[i0:i1], name="recovered (sign/scale fixed)", line=dict(color=c, width=1.6, dash="dot")),
    ])
    fig.update_xaxes(title_text="time (ms)")
    return _layout(fig, 260, f"Zoom: {name}", showlegend=True,
                   legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def heatmap(M, row_labels, col_labels, title=None, colorscale=None, zmid=None, fmt=".2f", height=None) -> go.Figure:
    M = np.asarray(M, float)
    cs = colorscale or [[0, "#120c24"], [0.5, "#6d28d9"], [1, "#22d3ee"]]
    fig = go.Figure(go.Heatmap(
        z=M, x=col_labels, y=row_labels, colorscale=cs, zmid=zmid,
        text=[[format(v, fmt) for v in r] for r in M], texttemplate="%{text}",
        textfont=dict(size=14, color="white"), xgap=3, ygap=3, showscale=False,
        hovertemplate="%{y} ← %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height or 90 + 55 * len(row_labels), title)


def spectrogram_fig(x: np.ndarray, sr: int, title: str | None = None, height: int = 220) -> go.Figure:
    f, t, Sxx = spectrogram(x, fs=sr, nperseg=512, noverlap=384)
    db = 10 * np.log10(Sxx + 1e-12)
    vmax = np.percentile(db, 99.5)
    step = max(1, len(t) // 400)
    fig = go.Figure(go.Heatmap(x=t[::step], y=f, z=db[:, ::step], zmin=vmax - 70, zmax=vmax,
                               colorscale="Magma", showscale=False,
                               hovertemplate="%{x:.2f}s · %{y:.0f} Hz<extra></extra>"))
    fig.update_yaxes(title_text="Hz")
    return _layout(fig, height, title)


def scatter_2d(P: np.ndarray, title: str, c: str = VIOLET, n: int = 3000, seed: int = 0) -> go.Figure:
    """Joint distribution of the first two rows — shows rotation / independence visually."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(P.shape[1], size=min(n, P.shape[1]), replace=False)
    a, b = P[0, idx], P[1, idx]
    lim = float(np.percentile(np.abs(np.r_[a, b]), 99.5)) * 1.1
    fig = go.Figure(go.Scattergl(x=a, y=b, mode="markers", marker=dict(size=3, color=c, opacity=0.45),
                                 hoverinfo="skip"))
    fig.update_xaxes(range=[-lim, lim], showticklabels=False)
    fig.update_yaxes(range=[-lim, lim], showticklabels=False, scaleanchor="x", scaleratio=1)
    return _layout(fig, 300, title)


def histogram_vs_gauss(x: np.ndarray, title: str, c: str = VIOLET) -> go.Figure:
    x = (x - x.mean()) / (x.std() + 1e-12)
    edges = np.linspace(-5, 5, 121)
    h, _ = np.histogram(x, bins=edges, density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    fig = go.Figure([
        go.Bar(x=mid, y=h, marker_color=_rgba(c, 0.7), name="signal", marker_line_width=0),
        go.Scatter(x=mid, y=np.exp(-mid ** 2 / 2) / np.sqrt(2 * np.pi), line=dict(color="white", width=2, dash="dash"),
                   name="Gaussian"),
    ])
    fig.update_layout(bargap=0)
    return _layout(fig, 240, title, showlegend=True, legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))


def bars(categories, series: dict, title=None, yaxis="SI-SDR (dB)", height=320) -> go.Figure:
    cols = [CYAN, VIOLET, CORAL, AMBER, LIME]
    fig = go.Figure([go.Bar(x=categories, y=v, name=k, marker_color=cols[i % len(cols)],
                            text=[f"{x:.1f}" for x in v], textposition="outside")
                     for i, (k, v) in enumerate(series.items())])
    fig.update_yaxes(title_text=yaxis)
    return _layout(fig, height, title, showlegend=True, barmode="group",
                   legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def lines(x, series: dict, title=None, xaxis="", yaxis="", height=320, log_y=False) -> go.Figure:
    cols = [CYAN, VIOLET, CORAL, AMBER, LIME, PINK]
    fig = go.Figure([go.Scatter(x=x, y=v, name=k, mode="lines+markers", line=dict(color=cols[i % 6], width=2.5),
                                marker=dict(size=7)) for i, (k, v) in enumerate(series.items())])
    fig.update_xaxes(title_text=xaxis)
    fig.update_yaxes(title_text=yaxis, type="log" if log_y else None)
    return _layout(fig, height, title, showlegend=True, legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def gauge(value_db: float, title: str = "Separation quality (mean SI-SDR)") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value_db, number=dict(suffix=" dB", font=dict(size=36, color=TEXT)),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(axis=dict(range=[-5, 50], tickcolor=MUTED), bar=dict(color=CYAN, thickness=0.28),
                   bgcolor=PANEL, borderwidth=0,
                   steps=[dict(range=[-5, 3], color="rgba(251,113,133,0.25)"),
                          dict(range=[3, 10], color="rgba(251,191,36,0.20)"),
                          dict(range=[10, 20], color="rgba(167,139,250,0.22)"),
                          dict(range=[20, 50], color="rgba(34,211,238,0.18)")])))
    fig = _layout(fig, 250)
    fig.update_layout(margin=dict(t=55, b=5, l=30, r=30))
    return fig


# ---------------------------------------------------------------------------
# Static Matplotlib figures for the report / notebook
# ---------------------------------------------------------------------------

def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#2a3040", "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#232838",
        "axes.grid": True, "font.size": 10, "axes.titleweight": "bold",
    })
    return plt


def mpl_reveal(S, X, Y, names, sr, path):
    plt = mpl_style()
    n = max(len(S), len(X))
    fig, axes = plt.subplots(n, 3, figsize=(13, 1.8 * n + 0.8), sharex=True)
    axes = np.atleast_2d(axes)
    for c, title in enumerate(["Original sources", "Observed mixtures", "Recovered by ICA (matched)"]):
        axes[0, c].set_title(title)
    for i in range(n):
        for c, (data, col) in enumerate([(S, color(i)), (X, MIX_COLOR), (Y, color(i))]):
            ax = axes[i, c]
            if i < len(data):
                t, lo, hi = envelope(data[i], sr)
                ax.fill_between(t, lo, hi, color=col, lw=0)
                lab = names[i] if c != 1 else f"Mic {i + 1}"
                ax.text(0.01, 0.85, lab, transform=ax.transAxes, color=TEXT, fontsize=8)
            else:
                ax.axis("off")
            ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mpl_bar(categories, series: dict, title, ylabel, path):
    plt = mpl_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    k = len(series)
    w = 0.8 / k
    cols = [CYAN, VIOLET, CORAL, AMBER, LIME]
    for i, (name, vals) in enumerate(series.items()):
        xs = np.arange(len(categories)) + (i - (k - 1) / 2) * w
        ax.bar(xs, vals, w, label=name, color=cols[i % 5])
        for x_, v in zip(xs, vals):
            ax.text(x_, v + (0.6 if v >= 0 else -2), f"{v:.1f}", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(range(len(categories)), categories, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mpl_lines(x, series: dict, title, xlabel, ylabel, path, xticklabels=None):
    plt = mpl_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [CYAN, VIOLET, CORAL, AMBER, LIME, PINK]
    for i, (name, v) in enumerate(series.items()):
        ax.plot(x, v, "-o", label=name, color=cols[i % 6], lw=2)
    if xticklabels is not None:
        ax.set_xticks(x, xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mpl_scatter_panels(panels, path, n=2500):
    """panels: list of (title, 2xT array, color)."""
    plt = mpl_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4))
    rng = np.random.default_rng(0)
    for ax, (title, P, c) in zip(axes, panels):
        idx = rng.choice(P.shape[1], size=min(n, P.shape[1]), replace=False)
        ax.scatter(P[0, idx], P[1, idx], s=2, alpha=0.4, color=c)
        lim = np.percentile(np.abs(P[:2, idx]), 99.5) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mpl_heatmap(M, rows, cols, title, path, fmt="{:.2f}"):
    plt = mpl_style()
    M = np.asarray(M)
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 0.7 * len(rows) + 1.4))
    ax.imshow(M, cmap="cool", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", color="black", fontsize=10, weight="bold")
    ax.set_xticks(range(len(cols)), cols, rotation=20)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mpl_spectrograms(signals, titles, sr, path):
    plt = mpl_style()
    n = len(signals)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3), sharey=True)
    for ax, x, t in zip(np.atleast_1d(axes), signals, titles):
        f, tt, Sxx = spectrogram(x, fs=sr, nperseg=512, noverlap=384)
        db = 10 * np.log10(Sxx + 1e-12)
        vmax = np.percentile(db, 99.5)
        ax.pcolormesh(tt, f, db, vmin=vmax - 70, vmax=vmax, cmap="magma", shading="auto")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("s")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mixing_sankey(A, src_names, mic_names, height=330) -> go.Figure:
    """Flow diagram: how much of each source reaches each microphone (link width = |a_ij|)."""
    A = np.abs(np.asarray(A, float))
    m, n = A.shape
    labels = list(src_names) + list(mic_names)
    src, tgt, val, col = [], [], [], []
    for i in range(m):
        for j in range(n):
            src.append(j)
            tgt.append(n + i)
            val.append(float(A[i, j]))
            col.append(_rgba(color(j), 0.45))
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, pad=18, thickness=16, line=dict(width=0),
                  color=[color(j) for j in range(n)] + [MIX_COLOR] * m),
        link=dict(source=src, target=tgt, value=val, color=col,
                  hovertemplate="%{source.label} → %{target.label}: gain %{value:.2f}<extra></extra>")))
    return _layout(fig, height)


def rotation_search(Z2: np.ndarray, theta_now: float, angles: np.ndarray, score: np.ndarray) -> go.Figure:
    """Non-Gaussianity of the rotated whitened data as a function of rotation angle."""
    fig = go.Figure([
        go.Scatter(x=np.degrees(angles), y=score, mode="lines", line=dict(color=CYAN, width=3),
                   fill="tozeroy", fillcolor=_rgba(CYAN, 0.12), name="non-Gaussianity"),
        go.Scatter(x=[np.degrees(theta_now)], y=[np.interp(theta_now, angles, score)], mode="markers",
                   marker=dict(size=14, color=CORAL, line=dict(color="white", width=2)), name="your angle"),
    ])
    fig.update_xaxes(title_text="rotation angle θ (degrees)")
    fig.update_yaxes(title_text="Σ negentropy")
    return _layout(fig, 300, "ICA = find the rotation that maximises non-Gaussianity")
