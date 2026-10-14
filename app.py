"""
UNMIX — The AI Audio Separation Lab.  "Discover the signals hidden in sound."

    streamlit run app.py

All signal processing lives in the modules next to this file:
    whitening.py (PCA whitening) · ica.py (FastICA) · evaluation.py (matching + metrics)
    audio_utils.py (I/O, mixing) · visualization.py (figures) · fingerprint.py (optional mini-Shazam)
This file is only the interface.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import audio_utils as au
import evaluation as ev
import visualization as viz
from ica import excess_kurtosis, fastica, negentropy, run_ica, run_pca
from whitening import center, covariance, pca_whiten, whiteness_error

ROOT = Path(__file__).resolve().parent
SR = au.DEFAULT_SR

st.set_page_config(page_title="UNMIX — Audio Separation Lab", page_icon="🎛️", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# Style
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --violet:#a78bfa; --cyan:#22d3ee; --coral:#fb7185; --amber:#fbbf24; --lime:#a3e635;
        --bg:#0d0f14; --panel:#151923; --line:rgba(255,255,255,.08); --muted:#8a90a2; }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; }
.stApp { background:
   radial-gradient(1200px 600px at 10% -10%, rgba(167,139,250,.12), transparent 60%),
   radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.08), transparent 60%),
   radial-gradient(800px 600px at 50% 120%, rgba(251,113,133,.07), transparent 60%), var(--bg); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.01em; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stSidebar"] { background: rgba(13,15,20,.92); border-right: 1px solid var(--line); }
.block-container { padding-top: 3.6rem; max-width: 1400px; }

.glass { background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 18px;
         padding: 18px 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
         box-shadow: 0 10px 40px rgba(0,0,0,.25); margin-bottom: 14px; }
.glass h4 { margin: 0 0 6px 0; }
.kicker { font-family:'JetBrains Mono',monospace; font-size:.75rem; letter-spacing:.18em; text-transform:uppercase;
          color: var(--cyan); }
.section-title { font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; margin:.1rem 0 .2rem 0;
   background: linear-gradient(90deg, #fff 0%, #c4b5fd 45%, #67e8f9 100%); -webkit-background-clip:text;
   background-clip:text; color: transparent; }
.section-sub { color: var(--muted); margin-bottom: 1.2rem; font-size: 1.02rem; }

.hero { position:relative; overflow:hidden; border-radius: 26px; padding: 46px 44px 38px 44px; margin-bottom: 18px;
        border: 1px solid rgba(167,139,250,.25);
        background: linear-gradient(135deg, rgba(167,139,250,.16), rgba(34,211,238,.07) 55%, rgba(251,113,133,.10)); }
.hero h1 { font-size: 4.2rem; margin: 0; line-height: 1; font-weight: 700;
   background: linear-gradient(90deg,#c4b5fd,#22d3ee 50%,#fb7185); -webkit-background-clip:text; background-clip:text;
   color: transparent; letter-spacing: .06em; }
.hero .tag { font-size: 1.35rem; color: #e6e8ef; margin-top: 10px; font-family:'Space Grotesk'; }
.hero .desc { color: var(--muted); max-width: min(720px, calc(100% - 260px)); margin-top: 10px; }
@media (max-width: 900px) { .eq { display:none } .hero .desc { max-width: 100% } .hero h1 { font-size: 3rem } }
.eq { position:absolute; right: 40px; bottom: 28px; display:flex; gap:5px; align-items:flex-end; height: 120px; opacity:.9 }
.eq span { width: 7px; border-radius: 4px; background: linear-gradient(180deg, var(--cyan), var(--violet));
           animation: bounce 1.2s ease-in-out infinite; box-shadow: 0 0 12px rgba(34,211,238,.45); }
.eq span:nth-child(3n) { background: linear-gradient(180deg, var(--coral), var(--violet)); }
@keyframes bounce { 0%,100% { height: 12% } 50% { height: 100% } }

.pill { display:inline-flex; align-items:center; gap:8px; padding: 6px 14px; border-radius: 999px; font-weight:600;
        font-size:.85rem; border:1px solid var(--line); background: rgba(255,255,255,.04); }
.dot { width:9px; height:9px; border-radius:50%; }
.dot.pulse { animation: pulse 1.1s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 currentColor } 70% { box-shadow:0 0 0 9px transparent } 100% { box-shadow:0 0 0 0 transparent } }

.pipeline { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin: 6px 0 4px 0; }
.step { padding: 8px 14px; border-radius: 12px; font-size: .85rem; border:1px solid var(--line);
        background: rgba(255,255,255,.03); color: var(--muted); font-family:'JetBrains Mono'; }
.step.on { color:#0d0f14; font-weight:600; border-color: transparent;
           background: linear-gradient(90deg, var(--violet), var(--cyan)); box-shadow: 0 0 18px rgba(167,139,250,.35); }
.arrow { color: var(--muted); }

.metric-card { border-radius: 16px; padding: 16px 18px; border: 1px solid var(--line); background: rgba(255,255,255,.03); }
.metric-card .v { font-family:'Space Grotesk'; font-size: 2rem; font-weight:700; }
.metric-card .l { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; }
.src-chip { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:middle }
.note { border-left: 3px solid var(--amber); padding: 10px 14px; background: rgba(251,191,36,.06);
        border-radius: 0 12px 12px 0; color: #f3e8c8; margin: 8px 0 14px 0; font-size: .93rem; }
.formula { font-family:'JetBrains Mono'; background: rgba(0,0,0,.35); border:1px solid var(--line); border-radius: 12px;
           padding: 12px 16px; color: #c4b5fd; margin: 8px 0; font-size: .92rem; white-space: pre-wrap; }

.stButton>button, .stDownloadButton>button { border-radius: 12px !important; font-weight: 600 !important; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
   background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; border: 0 !important; color: white !important;
   box-shadow: 0 6px 24px rgba(139,92,246,.35) !important; transition: transform .15s ease, box-shadow .15s ease; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px);
   box-shadow: 0 10px 30px rgba(34,211,238,.35) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 14px !important;
                             background: rgba(255,255,255,.02); }
audio { width: 100%; height: 38px; filter: invert(0.9) hue-rotate(180deg) saturate(1.4); border-radius: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS_STYLE = {
    "Ready": ("#22d3ee", False), "Mixing": ("#fbbf24", True), "Separating": ("#a78bfa", True),
    "Completed": ("#a3e635", False), "Error": ("#fb7185", False),
}

# =============================================================================
# State
# =============================================================================
ss = st.session_state
for k, v in {"status": "Ready", "sources": [], "S": None, "names": None, "A": None, "X": None, "snr": None,
             "result": None, "eval": None, "pca_eval": None, "settings": {}, "real": None, "run_seconds": None}.items():
    ss.setdefault(k, v)


def status_html(s: str) -> str:
    c, pulse = STATUS_STYLE.get(s, ("#8a90a2", False))
    return (f'<span class="pill"><span class="dot {"pulse" if pulse else ""}" style="background:{c};color:{c}"></span>'
            f'<span style="color:{c}">{s.upper()}</span></span>')


def set_status(s: str):
    ss.status = s
    if "status_slot" in globals():
        status_slot.markdown(status_html(s), unsafe_allow_html=True)


def plot(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False, "displayModeBar": False})


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def invalidate(level: str):
    """Sources changed -> drop the mix & results; mix changed -> drop the results."""
    if level == "sources":
        ss.S = ss.A = ss.X = None
    ss.result = ss.eval = ss.pca_eval = None
    ss.status = "Ready"


@st.cache_data(show_spinner=False)
def bundled_audio(path: str) -> np.ndarray:
    return au.load_audio(path, SR)


@st.cache_resource(show_spinner=False)
def fingerprint_db():
    import fingerprint as fp
    return fp.FingerprintDB.from_library()


BUNDLED = au.list_sources()
DEFAULT_PICK = [n for n in ("Speech Female", "Music Neon Drift", "Ambient Rain") if n in BUNDLED]
PRESET_A = np.array([[1.00, 0.60, 0.45], [0.55, 1.00, 0.50], [0.40, 0.65, 1.00]])


def load_selection(picks: list[str], uploads=None, duration=8.0):
    srcs = []
    for p in picks:
        srcs.append({"name": p, "signal": bundled_audio(str(BUNDLED[p])), "origin": "sample"})
    for f in uploads or []:
        srcs.append({"name": Path(f.name).stem.replace("_", " ").title(),
                     "signal": au.load_audio(f.getvalue(), SR, f.name), "origin": "upload"})
    ss.sources = srcs
    ss.duration = duration
    invalidate("sources")


def build_mix(A: np.ndarray, snr):
    set_status("Mixing")
    sig = [s["signal"] for s in ss.sources]
    S = au.align_lengths(sig, "trim")[:, : int(ss.get("duration", 8.0) * SR)]
    S = au.standardize(S)
    ss.S, ss.names, ss.A, ss.snr = S, [s["name"] for s in ss.sources], np.asarray(A, float), snr
    ss.X = au.add_noise(au.mix(S, ss.A), snr, seed=0)
    invalidate("mix")
    set_status("Ready")


def run_separation(contrast="logcosh", algorithm="symmetric", max_iter=400, tol=1e-6, seed=0, progress=None):
    set_status("Separating")
    n = ss.S.shape[0]
    t0 = time.perf_counter()
    r = run_ica(ss.X, n_components=n, contrast=contrast, algorithm=algorithm, max_iter=max_iter, tol=tol, seed=seed)
    ss.run_seconds = time.perf_counter() - t0
    ss.result = r
    A_sq = ss.A if ss.A.shape[0] == ss.A.shape[1] else None
    ss.eval = ev.evaluate(ss.S, r.sources, X=ss.X, W_total=r.unmixing if A_sq is not None else None, A=A_sq,
                          names=ss.names)
    ss.pca_eval = ev.evaluate(ss.S, run_pca(ss.X, n), X=ss.X, names=ss.names)
    ss.settings = dict(contrast=contrast, algorithm=algorithm, max_iter=max_iter, tol=tol, seed=seed)
    set_status("Completed")


def quick_demo():
    load_selection(DEFAULT_PICK, duration=8.0)
    build_mix(PRESET_A, None)
    run_separation()


# =============================================================================
# Sidebar
# =============================================================================
def _demo_clicked():
    # Callbacks run before the script, so the page radio can still be switched here.
    ss.page = "✨  D · Separation Reveal"
    ss.pending_demo = True


PAGES = ["🏠  Lab Overview", "🎙️  A · Sound Studio", "🌀  B · Mixing Chamber", "⚙️  C · ICA Engine",
         "✨  D · Separation Reveal", "📊  E · Evaluation", "🔬  F · Science of Sound",
         "📱  Real-World Mode", "🎵  Mini-Shazam (extension)"]
# Deep links for presentations: ?page=reveal&demo=1 opens the Reveal with the demo already run.
PAGE_SLUGS = dict(zip(["home", "studio", "mixing", "engine", "reveal", "eval", "science", "real", "shazam"], PAGES))
if not ss.get("_query_handled"):
    ss._query_handled = True
    qp = st.query_params
    if qp.get("page") in PAGE_SLUGS:
        ss.page = PAGE_SLUGS[qp["page"]]
    if qp.get("demo") == "1":
        ss.pending_demo = True

with st.sidebar:
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.9rem;font-weight:700;letter-spacing:.08em;'
                'background:linear-gradient(90deg,#c4b5fd,#22d3ee);-webkit-background-clip:text;color:transparent">'
                'UNMIX</div><div style="color:#8a90a2;margin-top:-4px;margin-bottom:14px">'
                'Discover the signals hidden in sound.</div>', unsafe_allow_html=True)
    status_slot = st.empty()
    status_slot.markdown(status_html(ss.status), unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    n_src = len(ss.sources)
    st.caption(f"Sources loaded: **{n_src}**  ·  Mix: **{'yes' if ss.X is not None else '—'}**  ·  "
               f"ICA: **{'done' if ss.result is not None else '—'}**")
    st.button("⚡ One-click demo", type="primary", width="stretch", on_click=_demo_clicked,
              help="Loads speech + music + rain, mixes them with the preset matrix, runs ICA and opens the Reveal.")

if ss.pop("pending_demo", False):
    with st.spinner("Loading, mixing and unmixing…"):
        quick_demo()
    st.rerun()


def header(kicker: str, title: str, sub: str):
    st.markdown(f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div>'
                f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def need(cond: bool, msg: str) -> bool:
    if not cond:
        st.info(msg, icon="🧭")
    return cond


def metric_card(col, label, value, color="#e6e8ef"):
    col.markdown(f'<div class="metric-card"><div class="l">{label}</div>'
                 f'<div class="v" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)


# =============================================================================
# Pages
# =============================================================================

def page_home():
    bars = "".join(f'<span style="animation-delay:{(i * 0.11) % 1.2:.2f}s;animation-duration:{0.9 + (i % 5) * 0.15:.2f}s"></span>'
                   for i in range(22))
    st.markdown(f"""
    <div class="hero">
      <div class="kicker">The AI Audio Separation Lab</div>
      <h1>UNMIX</h1>
      <div class="tag">Discover the signals hidden in sound.</div>
      <div class="desc">One recording. Multiple hidden voices. One algorithm to separate them. UNMIX mixes clean
      speech, music and ambience with a matrix you control, then recovers every source with
      <b>PCA whitening</b> and <b>FastICA written from scratch</b> — and proves it with real metrics.</div>
      <div class="eq">{bars}</div>
    </div>""", unsafe_allow_html=True)

    done = [bool(ss.sources), ss.X is not None, ss.X is not None, ss.result is not None, ss.result is not None,
            ss.eval is not None, ss.eval is not None]
    steps = ["load", "mix  X = AS", "center", "whiten  Z = KX", "FastICA  Y = WZ", "match (Hungarian)", "evaluate"]
    html = '<div class="pipeline">' + '<span class="arrow">→</span>'.join(
        f'<span class="step {"on" if d else ""}">{s}</span>' for s, d in zip(steps, done)) + "</div>"
    st.markdown('<div class="glass"><h4>Pipeline</h4>' + html + "</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="glass"><div class="kicker">step 1</div><h4>🎙️ Sound Studio</h4>'
                    'Pick speech, music and ambient sources — or upload your own WAVs.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass"><div class="kicker">step 2</div><h4>🌀 Mixing Chamber</h4>'
                    'Design the mixing matrix A: random, preset or hand-edited, with optional sensor noise.</div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass"><div class="kicker">step 3</div><h4>⚙️ ICA Engine → ✨ Reveal</h4>'
                    'Center, whiten, rotate. Listen to what comes out and score it against the truth.</div>',
                    unsafe_allow_html=True)

    if ss.eval is not None:
        s = ss.eval["summary"]
        c = st.columns(4)
        metric_card(c[0], "mean SI-SDR (ICA)", f"{s['mean_si_sdr']:.1f} dB", "#22d3ee")
        metric_card(c[1], "improvement over best mic", f"{s['mean_si_sdr_improvement']:+.1f} dB", "#a78bfa")
        metric_card(c[2], "mean |correlation|", f"{s['mean_abs_corr']:.4f}", "#a3e635")
        metric_card(c[3], "PCA-only SI-SDR", f"{ss.pca_eval['summary']['mean_si_sdr']:.1f} dB", "#fb7185")
    else:
        st.markdown('<div class="note">New here? Hit <b>⚡ One-click demo</b> in the sidebar, then open '
                    '<b>✨ Separation Reveal</b>. Or walk through the stages A → F yourself.</div>',
                    unsafe_allow_html=True)

    res_path = ROOT / "outputs" / "results.json"
    if res_path.exists():
        R = json.loads(res_path.read_text(encoding="utf-8"))
        e3 = R.get("e3_snr", [])
        if e3:
            with st.expander("📈 Pre-computed benchmark (from `python experiments.py`)", expanded=False):
                plot(viz.lines([str(o["snr_db"]) for o in e3],
                               {"ICA": [o["ica"] for o in e3], "PCA only": [o["pca"] for o in e3],
                                "best single mic": [o["input"] for o in e3]},
                               "Separation quality vs sensor noise", "input SNR (dB)", "mean SI-SDR (dB)"))
                st.caption("Full write-up with all eight experiments: REPORT.md")


def page_studio():
    header("A · Sound Studio", "Choose your voices", "Clean, independent sources go in. Pick at least two — "
           "the classic demo is speech + music + ambience.")
    left, right = st.columns([1.15, 1])
    with left:
        picks = st.multiselect("Bundled samples", list(BUNDLED), default=[s["name"] for s in ss.sources
                                                                           if s["origin"] == "sample"] or DEFAULT_PICK)
        uploads = st.file_uploader("…or upload clean WAV sources", type=["wav"], accept_multiple_files=True,
                                   help="Any sample rate / mono or stereo — UNMIX resamples to 16 kHz mono.")
    with right:
        duration = st.slider("Clip length (seconds)", 2.0, 10.0, float(ss.get("duration", 8.0)), 0.5)
        st.markdown('<div class="glass" style="font-size:.9rem">Every file is converted to <b>mono</b>, '
                    '<b>resampled to 16 kHz</b> (polyphase, anti-aliased), <b>trimmed to a common length</b> and '
                    '<b>standardised</b> (zero mean, unit variance — ICA cannot recover scale anyway).<br><br>'
                    'The bundled files are deliberately stored at 16, 22.05, 44.1 and 48 kHz, mono and stereo, '
                    'so this preprocessing is exercised for real.</div>', unsafe_allow_html=True)
        if st.button("Load sources", type="primary", width="stretch"):
            try:
                if len(picks) + len(uploads or []) < 2:
                    st.error("Pick at least two sources — separating one signal from itself is not a thing.")
                else:
                    load_selection(picks, uploads, duration)
                    st.success(f"Loaded {len(ss.sources)} sources.")
            except au.AudioError as e:
                st.error(str(e))

    if not need(bool(ss.sources), "Load some sources to see them here."):
        return
    st.markdown("#### Your sources")
    for i, s in enumerate(ss.sources):
        x = s["signal"][: int(ss.get("duration", 8.0) * SR)]
        k = excess_kurtosis(x[None])[0]
        with st.container():
            c1, c2 = st.columns([1, 2.4])
            with c1:
                st.markdown(f'<span class="src-chip" style="background:{viz.color(i)}"></span>'
                            f'<b>{s["name"]}</b> <span style="color:#8a90a2">· {s["origin"]}</span>',
                            unsafe_allow_html=True)
                st.caption(f"{len(x) / SR:.1f} s · excess kurtosis {k:.2f} "
                           f"({'super-Gaussian ✓' if k > 0.5 else 'near-Gaussian ⚠' if abs(k) < 0.5 else 'sub-Gaussian'})")
                st.audio(au.wav_bytes(x, SR), format="audio/wav")
            with c2:
                plot(viz.waveform(x, SR, viz.color(i), 105, s["name"]), key=f"src_wave_{i}")


def page_mixing():
    header("B · Mixing Chamber", "Blend them together", "Each microphone hears a weighted sum of every source: "
           "X = A·S. You choose A.")
    if not need(len(ss.sources) >= 2, "Load at least two sources in the Sound Studio first."):
        return
    n = len(ss.sources)
    names = [s["name"] for s in ss.sources]
    c1, c2, c3 = st.columns([1, 1, 1])
    m = c1.number_input("Microphones (observations)", min_value=n, max_value=n + 4, value=n,
                        help="ICA needs at least as many mixtures as sources. Extra mics are reduced by PCA.")
    mode = c2.radio("Mixing matrix", ["Preset room", "Random", "Hand-edit"], horizontal=True)
    snr_choice = c3.select_slider("Sensor noise (input SNR)", ["clean", 40, 30, 20, 15, 10, 5, 0], value="clean")
    snr = None if snr_choice == "clean" else float(snr_choice)

    if mode == "Random":
        seed = st.number_input("Random seed", 0, 10_000, 42)
        A = au.random_mixing_matrix(int(m), n, seed=int(seed))
    elif mode == "Preset room" and n == 3 and m == 3:
        A = PRESET_A.copy()
    else:
        A = au.random_mixing_matrix(int(m), n, seed=7)
    if mode == "Hand-edit":
        st.caption("Edit the gains — row = microphone, column = source. Negative values are allowed "
                   "(think phase-inverted wiring).")
        base = ss.A if ss.A is not None and ss.A.shape == (m, n) else A
        df = st.data_editor(pd.DataFrame(base, index=[f"Mic {i + 1}" for i in range(m)], columns=names),
                            key=f"editor_{m}_{n}")
        A = df.to_numpy(dtype=float)
    cond = np.linalg.cond(A)
    rank = np.linalg.matrix_rank(A)
    if rank < n:
        st.error(f"This matrix has rank {rank} < {n} sources — it is not invertible, so no algorithm can "
                 "separate the sources. Change some gains.")
    elif cond > 100:
        st.warning(f"cond(A) = {cond:.0f}: nearly singular. Without noise ICA still works (equivariance), but any "
                   "sensor noise will be massively amplified by the unmixing matrix.")

    if st.button("🌀 Mix the sources", type="primary", disabled=rank < n):
        build_mix(A, snr)
        st.toast("Mixed! Head to the ICA Engine.", icon="🌀")

    left, right = st.columns([1, 1])
    with left:
        plot(viz.heatmap(A, [f"Mic {i + 1}" for i in range(int(m))], names, "Mixing matrix A",
                         colorscale=[[0, "#fb7185"], [0.5, "#151923"], [1, "#22d3ee"]] if A.min() < 0 else None,
                         zmid=0 if A.min() < 0 else None), key="mixheat")
        st.caption(f"cond(A) = {cond:.2f} · rank {rank}")
    with right:
        plot(viz.mixing_sankey(A, names, [f"Mic {i + 1}" for i in range(int(m))]), key="sankey")
        st.caption("How the sources flow into each microphone (width = |gain|).")
    eqs = "\n".join(f"Mic {i + 1}(t) = " + " + ".join(f"{A[i, j]:.2f}·{names[j]}(t)" for j in range(n))
                    + (" + noise(t)" if snr else "") for i in range(int(m)))
    st.markdown(f'<div class="formula">{eqs}</div>', unsafe_allow_html=True)

    if ss.X is not None:
        st.markdown("#### Originals vs. what the microphones hear")
        a, b = st.columns(2)
        with a:
            plot(viz.waveform_stack(list(ss.S), ss.names, SR, title="Original sources"), key="mix_orig")
        with b:
            plot(viz.waveform_stack(list(ss.X), [f"Mic {i + 1}" for i in range(len(ss.X))], SR,
                                    colors=[viz.MIX_COLOR] * len(ss.X), title="Observed mixtures"), key="mix_obs")
        cols = st.columns(len(ss.X))
        for i, c in enumerate(cols):
            c.caption(f"🎧 Mic {i + 1}")
            c.audio(au.wav_bytes(ss.X[i], SR), format="audio/wav")


def page_engine():
    header("C · ICA Engine", "Center → whiten → rotate", "Every step below is computed by hand-written NumPy "
           "code (whitening.py, ica.py) — no scikit-learn.")
    if not need(ss.X is not None, "Create a mixture in the Mixing Chamber first."):
        return
    c = st.columns(5)
    contrast = c[0].selectbox("Contrast G", ["logcosh", "exp", "cube"],
                              help="logcosh ≈ general purpose; exp for very spiky sources; cube = kurtosis")
    algorithm = c[1].selectbox("Strategy", ["symmetric", "deflation"])
    max_iter = c[2].number_input("Max iterations", 10, 5000, 400, 10)
    tol = c[3].select_slider("Tolerance", [1e-3, 1e-4, 1e-5, 1e-6, 1e-8], value=1e-6, format_func=lambda v: f"{v:.0e}")
    seed = c[4].number_input("Init seed", 0, 9999, 0)

    if st.button("⚙️ Run UNMIX", type="primary"):
        with st.status("Running the separation pipeline…", expanded=True) as box:
            set_status("Separating")
            st.write("① Centering the observations (subtract per-mic mean)…")
            st.write("② Covariance matrix + eigendecomposition (np.linalg.eigh)…")
            st.write("③ PCA whitening  Z = D^-½ Eᵀ X …")
            st.write(f"④ FastICA fixed-point iterations ({contrast}, {algorithm})…")
            try:
                run_separation(contrast, algorithm, int(max_iter), float(tol), int(seed))
                r = ss.result
                st.write(f"⑤ Hungarian matching + metrics… done in {ss.run_seconds:.2f} s")
                box.update(label=f"Separated in {r.n_iter} iterations "
                                 f"({'converged' if r.converged else 'NOT converged — try more iterations'})",
                           state="complete" if r.converged else "error")
            except ValueError as e:
                set_status("Error")
                box.update(label="Separation failed", state="error")
                st.error(str(e))
                return

    r = ss.result
    if r is None:
        Xc, _ = center(ss.X)
        C = covariance(Xc)
        st.markdown("#### Preview: the observations are strongly correlated")
        plot(viz.heatmap(C, [f"Mic {i + 1}" for i in range(len(C))], [f"Mic {i + 1}" for i in range(len(C))],
                         "Cov(X) before whitening"), key="covx")
        return

    wr = r.whitening
    k = st.columns(4)
    metric_card(k[0], "iterations", f"{r.n_iter}", "#22d3ee")
    metric_card(k[1], "converged", "yes ✓" if r.converged else "no ✗", "#a3e635" if r.converged else "#fb7185")
    metric_card(k[2], "max |Cov(Z) − I|", f"{whiteness_error(wr.Z):.1e}", "#a78bfa")
    metric_card(k[3], "runtime", f"{ss.run_seconds * 1000:.0f} ms", "#fbbf24")
    st.write("")

    t1, t2, t3, t4 = st.tabs(["① Whitening", "② FastICA convergence", "③ Matrices", "④ Geometry"])
    with t1:
        a, b, c3 = st.columns(3)
        Xc, _ = center(ss.X)
        mics = [f"Mic {i + 1}" for i in range(len(ss.X))]
        with a:
            plot(viz.heatmap(covariance(Xc), mics, mics, "Cov(X) — correlated"), key="cx")
        with b:
            comps = [f"PC {i + 1}" for i in range(wr.n_components)]
            plot(viz.heatmap(np.cov(wr.Z), comps, comps, "Cov(Z) — identity ✓"), key="cz")
        with c3:
            ev_ratio = wr.explained_variance_ratio
            plot(viz.bars([f"λ{i + 1}" for i in range(len(ev_ratio))], {"variance share": list(ev_ratio * 100)},
                          "Eigenvalues of Cov(X)", "% of variance", 300), key="eig")
        st.markdown('<div class="formula">C = Xc Xcᵀ/(T−1) = E D Eᵀ      K = D^-½ Eᵀ      Z = K Xc   ⇒   Cov(Z) = I</div>',
                    unsafe_allow_html=True)
        st.caption("After whitening, the remaining unknown is just a rotation — that is all FastICA has to find.")
    with t2:
        h = np.maximum(np.array(r.history), 1e-16)
        plot(viz.lines(list(range(1, len(h) + 1)), {"max |1 − |⟨wₙₑw, wₒₗd⟩||": list(h)},
                       "Convergence of the fixed-point iteration", "iteration", "change (log)", log_y=True),
             key="conv")
        st.markdown('<div class="formula">w ← E[z·g(wᵀz)] − E[g′(wᵀz)]·w        then  W ← (WWᵀ)^-½ W  (symmetric decorrelation)</div>',
                    unsafe_allow_html=True)
    with t3:
        a, b = st.columns(2)
        ic = [f"IC {i + 1}" for i in range(r.unmixing.shape[0])]
        with a:
            plot(viz.heatmap(r.unmixing, ic, [f"Mic {i + 1}" for i in range(r.unmixing.shape[1])],
                             "Unmixing matrix W·K", colorscale=[[0, "#fb7185"], [0.5, "#151923"], [1, "#22d3ee"]],
                             zmid=0), key="unmix")
        with b:
            if ss.A.shape[0] == ss.A.shape[1]:
                P = r.unmixing @ ss.A
                plot(viz.heatmap(P, ic, ss.names, "Global system (W·K)·A → scaled permutation?",
                                 colorscale=[[0, "#fb7185"], [0.5, "#151923"], [1, "#22d3ee"]], zmid=0), key="glob")
                st.caption("One dominant entry per row/column = perfect separation. Its position is the "
                           "permutation, its sign the polarity flip.")
            else:
                P = r.unmixing @ ss.A
                plot(viz.heatmap(P, ic, ss.names, "(W·K)·A"), key="glob2")
    with t4:
        if ss.S.shape[0] >= 2:
            Xc, _ = center(ss.X)
            e = ss.eval
            cols = st.columns(4)
            panels = [("Sources", ss.S[:2], viz.VIOLET), ("Mixtures", Xc[:2], viz.MIX_COLOR),
                      ("Whitened", wr.Z[:2], viz.AMBER), ("ICA output", r.sources[[e["perm"][0], e["perm"][1]]], viz.CYAN)]
            for col, (t, P, cl) in zip(cols, panels):
                with col:
                    plot(viz.scatter_2d(P, t, cl), key=f"geo_{t}")
            st.caption(f"Joint distribution of {ss.names[0]} vs {ss.names[1]}. Mixing shears the '+' shape; whitening "
                       "makes it round in covariance but leaves it rotated; ICA finds the rotation that makes the "
                       "arms line up with the axes again.")


def page_reveal():
    header("D · Separation Reveal", "Hear the hidden voices", "Original → mixed → recovered. Every component has "
           "its own player and download.")
    if not need(ss.result is not None, "Run the ICA Engine (or the ⚡ One-click demo) first."):
        return
    r, e = ss.result, ss.eval
    aligned = e["aligned"]
    plot(viz.reveal_grid(ss.S, ss.X, aligned, ss.names, SR), key="reveal")
    st.markdown('<div class="note">⚠️ <b>ICA has two built-in ambiguities.</b> It cannot know which component is '
                '“source 1” (<b>permutation</b>), nor the true loudness or polarity (<b>scale & sign</b>) — '
                'A·S = (A·D·P)(P⁻¹D⁻¹·S) for any permutation P and diagonal scaling D. The raw components below are '
                'in the order ICA produced them; the right column above was matched to the originals with the '
                'Hungarian algorithm and rescaled by least squares (sign flips included).</div>',
                unsafe_allow_html=True)

    inv = {comp: i for i, comp in enumerate(e["perm"])}
    st.markdown("#### Recovered components (raw ICA order)")
    cols = st.columns(min(3, r.sources.shape[0]))
    for k in range(r.sources.shape[0]):
        with cols[k % len(cols)]:
            i = inv.get(k)
            row = e["rows"][i] if i is not None else None
            colr = viz.color(i) if i is not None else viz.MIX_COLOR
            label = f"→ {row['source']}" if row else "unmatched"
            st.markdown(f'<div class="glass"><div class="kicker">Component {k + 1}</div>'
                        f'<h4 style="color:{colr}">{label}</h4>'
                        + (f'<span style="color:#8a90a2">|r| = {row["abs_corr"]:.4f} · SI-SDR {row["si_sdr"]:.1f} dB · '
                           f'sign {"+" if row["sign"] > 0 else "− (flipped)"}</span>' if row else "")
                        + "</div>", unsafe_allow_html=True)
            plot(viz.waveform(r.sources[k], SR, colr, 90), key=f"rc_{k}")
            data = au.wav_bytes(r.sources[k], SR)
            st.audio(data, format="audio/wav")
            st.download_button("⬇ Download WAV", data, f"unmix_component_{k + 1}"
                               + (f"_{slug(row['source'])}" if row else "") + ".wav", "audio/wav",
                               key=f"dl_{k}", width="stretch")

    st.markdown("#### Before / after, up close")
    a, b = st.columns([1, 2])
    with a:
        pick = st.selectbox("Source", ss.names)
        i = ss.names.index(pick)
        start = st.slider("Zoom position (s)", 0.0, max(0.0, ss.S.shape[1] / SR - 0.06), 1.0, 0.01)
        st.caption("Original")
        st.audio(au.wav_bytes(ss.S[i], SR), format="audio/wav")
        st.caption("Recovered (sign & scale aligned)")
        st.audio(au.wav_bytes(aligned[i], SR), format="audio/wav")
    with b:
        plot(viz.overlay(ss.S[i], aligned[i], SR, start, 0.05, viz.color(i), pick), key="ovl")
    with st.expander("🌈 Spectrograms — frequency content before and after", expanded=True):
        c = st.columns(3)
        mic = int(np.argmax(np.abs(ss.A[:, i])))
        with c[0]:
            plot(viz.spectrogram_fig(ss.S[i], SR, f"{pick} — original"), key="sp0")
        with c[1]:
            plot(viz.spectrogram_fig(ss.X[mic], SR, f"Mic {mic + 1} — mixture"), key="sp1")
        with c[2]:
            plot(viz.spectrogram_fig(aligned[i], SR, f"{pick} — recovered"), key="sp2")

    st.download_button("📦 Download everything (WAVs + report)", bundle_zip(), "unmix_results.zip",
                       "application/zip", type="primary")


def build_report_md() -> str:
    e, p, r = ss.eval, ss.pca_eval, ss.result
    L = [f"# UNMIX run report — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"Sources: {', '.join(ss.names)}  ", f"Samples: {ss.S.shape[1]} @ {SR} Hz ({ss.S.shape[1] / SR:.1f} s)  ",
         f"Sensor noise: {'none' if ss.snr is None else f'{ss.snr:.0f} dB SNR'}  ",
         f"Algorithm: FastICA, contrast={ss.settings.get('contrast')}, {ss.settings.get('algorithm')}, "
         f"tol={ss.settings.get('tol')}, seed={ss.settings.get('seed')}  ",
         f"Iterations: {r.n_iter} (converged={r.converged})", "", "## Mixing matrix A", "", "```",
         *["  ".join(f"{v:+.3f}" for v in row) for row in ss.A], "```", "", "## Estimated unmixing matrix W·K", "",
         "```", *["  ".join(f"{v:+.3f}" for v in row) for row in r.unmixing], "```", "", "## Metrics", "",
         "| source | matched IC | sign | abs corr | SI-SDR dB | SIR dB | best-mic SI-SDR | PCA SI-SDR |",
         "|---|---|---|---|---|---|---|---|"]
    for a, b in zip(e["rows"], p["rows"]):
        L.append(f"| {a['source']} | {a['matched_component']} | {a['sign']:+d} | {a['abs_corr']:.4f} | {a['si_sdr']:.2f} | "
                 f"{a['sir']:.2f} | {a['input_si_sdr']:.2f} | {b['si_sdr']:.2f} |")
    s = e["summary"]
    L += ["", f"Mean SI-SDR {s['mean_si_sdr']:.2f} dB (improvement {s['mean_si_sdr_improvement']:+.2f} dB over the best "
              f"single mic); PCA-only {p['summary']['mean_si_sdr']:.2f} dB."
          + (f" Amari index {s['amari_index']:.4f}." if "amari_index" in s else ""), "",
          "Metrics are invariant to the permutation/sign/scale ambiguity of ICA (Hungarian matching on |corr|, "
          "least-squares gain)."]
    return "\n".join(L)


def bundle_zip() -> bytes:
    buf = io.BytesIO()
    e, r = ss.eval, ss.result
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, n in enumerate(ss.names):
            z.writestr(f"originals/{i + 1}_{slug(n)}.wav", au.wav_bytes(ss.S[i], SR))
            z.writestr(f"recovered_aligned/{slug(n)}.wav", au.wav_bytes(e["aligned"][i], SR))
        for k in range(r.sources.shape[0]):
            z.writestr(f"ica_components_raw/component_{k + 1}.wav", au.wav_bytes(r.sources[k], SR))
        for i in range(ss.X.shape[0]):
            z.writestr(f"mixtures/mic_{i + 1}.wav", au.wav_bytes(ss.X[i], SR))
        z.writestr("report.md", build_report_md())
        z.writestr("metrics.json", json.dumps({"rows": e["rows"], "summary": e["summary"],
                                               "pca": ss.pca_eval["summary"], "A": ss.A.tolist(),
                                               "unmixing": r.unmixing.tolist(), "settings": ss.settings}, indent=2))
    return buf.getvalue()


def page_eval():
    header("E · Evaluation Dashboard", "How good is the separation?", "Scored against the clean originals, "
           "after resolving ICA's permutation, sign and scale ambiguity.")
    if not need(ss.eval is not None, "Run a separation first."):
        return
    e, p = ss.eval, ss.pca_eval
    s = e["summary"]
    a, b = st.columns([1, 1.6])
    with a:
        plot(viz.gauge(s["mean_si_sdr"]), key="gauge")
        q, why = ev.quality_label(s["mean_si_sdr"])
        st.markdown(f'<div style="text-align:center;margin-top:-12px"><b style="font-size:1.2rem">{q}</b><br>'
                    f'<span style="color:#8a90a2">{why}</span></div>', unsafe_allow_html=True)
    with b:
        k = st.columns(3)
        metric_card(k[0], "mean |r|", f"{s['mean_abs_corr']:.4f}", "#a3e635")
        metric_card(k[1], "Δ vs best mic", f"{s['mean_si_sdr_improvement']:+.1f} dB", "#a78bfa")
        metric_card(k[2], "Amari index", f"{s['amari_index']:.4f}" if "amari_index" in s else "n/a", "#22d3ee")
        st.write("")
        plot(viz.bars(ss.names, {"best single mic": [r_["input_si_sdr"] for r_ in e["rows"]],
                                 "PCA only": [r_["si_sdr"] for r_ in p["rows"]],
                                 "ICA": [r_["si_sdr"] for r_ in e["rows"]]}, "SI-SDR per source", height=300),
             key="bars")

    df = pd.DataFrame(e["rows"]).rename(columns={
        "source": "Source", "matched_component": "Matched IC", "sign": "Sign", "abs_corr": "|r|",
        "si_sdr": "SI-SDR (dB)", "sir": "SIR (dB)", "sar": "SAR (dB)", "input_si_sdr": "Best-mic SI-SDR",
        "si_sdr_improvement": "Improvement (dB)"})
    df["Quality"] = [ev.quality_label(v)[0] for v in df["SI-SDR (dB)"]]
    df["SAR (dB)"] = [f"{v:.1f}" if v is not None and v == v else "∞ (no noise)" for v in df["SAR (dB)"]]
    st.dataframe(df.round(3), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        C = np.abs(np.array(e["corr_matrix"]))
        plot(viz.heatmap(C, ss.names, [f"IC {i + 1}" for i in range(C.shape[1])],
                         "|correlation|: sources × ICA components"), key="corr")
        st.caption("Hungarian assignment picks one bright cell per row & column (maximum total |r|).")
    with c2:
        C2 = np.abs(np.array(p["corr_matrix"]))
        plot(viz.heatmap(C2, ss.names, [f"PC {i + 1}" for i in range(C2.shape[1])],
                         "|correlation|: sources × PCA components"), key="corrp")
        st.caption("PCA components are uncorrelated with each other — but each is still a blend of sources.")

    with st.expander("ℹ️ What the metrics mean — and their limits"):
        st.markdown("""
* **|r|** — absolute Pearson correlation with the matched component. Intuitive, invariant to sign/scale, but it
  saturates: 0.99 still means ≈ 17 dB of distortion.
* **SI-SDR** — scale-invariant signal-to-distortion ratio: project the estimate onto the source
  (ŷ = αs + e, α = ⟨ŷ,s⟩/‖s‖²) and compare energies, 10·log10(‖αs‖²/‖e‖²). Each +10 dB = 10× less error energy.
* **SIR / SAR** — the error split into leakage from the *other* sources vs. everything else (noise, artefacts).
* **Best-mic SI-SDR** — the baseline “do nothing, just listen to the closest microphone”.
* **Amari index** — 0 when (W·K)·A is exactly a scaled permutation; uses the true A, so only for square synthetic mixes.

**Limitations:** all of these need the clean originals. They say nothing about perceptual quality in detail, and
in the real-world mode there is no ground truth at all. With sensor noise, ICA's output contains noise too —
separation is not denoising.""")

    st.markdown("### 🧪 Live experiments on your sources")
    tA, tB = st.tabs(["Noise sweep (ICA vs PCA)", "Mixing conditions"])
    with tA:
        if st.button("Run noise sweep", key="btn_sweep"):
            snrs = ["clean", 40, 30, 20, 10, 5, 0]
            out = {"ICA": [], "PCA only": [], "best single mic": []}
            bar = st.progress(0.0)
            Xclean = au.mix(ss.S, ss.A)
            for j, sv in enumerate(snrs):
                X = au.add_noise(Xclean, None if sv == "clean" else float(sv), seed=1)
                rr = run_ica(X, n_components=ss.S.shape[0], seed=0)
                ee = ev.evaluate(ss.S, rr.sources, X=X)
                out["ICA"].append(ee["summary"]["mean_si_sdr"])
                out["PCA only"].append(ev.evaluate(ss.S, run_pca(X, ss.S.shape[0]))["summary"]["mean_si_sdr"])
                out["best single mic"].append(ee["summary"]["mean_input_si_sdr"])
                bar.progress((j + 1) / len(snrs))
            ss.sweep_results = ([str(v) for v in snrs], out)
        if ss.get("sweep_results"):
            x, out = ss.sweep_results
            plot(viz.lines(x, out, "Mean SI-SDR vs input SNR", "input SNR (dB)", "SI-SDR (dB)"), key="sweepplot")
            st.caption("ICA output quality follows input SNR: the unmixing matrix passes sensor noise through.")
    with tB:
        if st.button("Compare 12 random mixing matrices @ 30 dB SNR", key="btn_conds"):
            rng = np.random.default_rng(0)
            rows = []
            n = ss.S.shape[0]
            for t in range(12):
                A = rng.uniform(-1, 1, (n, n))
                X = au.add_noise(au.mix(ss.S, A), 30.0, seed=t)
                rr = run_ica(X, n_components=n, seed=t)
                rows.append({"cond(A)": np.linalg.cond(A),
                             "ICA SI-SDR": ev.evaluate(ss.S, rr.sources)["summary"]["mean_si_sdr"],
                             "PCA SI-SDR": ev.evaluate(ss.S, run_pca(X, n))["summary"]["mean_si_sdr"]})
            ss.cond_results = pd.DataFrame(rows).sort_values("cond(A)")
        if ss.get("cond_results") is not None:
            d = ss.cond_results
            plot(viz.lines(list(np.round(np.log10(d["cond(A)"]), 2)),
                           {"ICA": list(d["ICA SI-SDR"]), "PCA only": list(d["PCA SI-SDR"])},
                           "Ill-conditioned mixing amplifies noise", "log10 cond(A)", "mean SI-SDR (dB)"),
                 key="condplot")

    c1, c2, c3 = st.columns(3)
    c1.download_button("📝 Report (Markdown)", build_report_md(), "unmix_report.md", "text/markdown", width="stretch")
    c2.download_button("🧾 Metrics (JSON)", json.dumps({"rows": e["rows"], "summary": e["summary"],
                                                       "pca_summary": p["summary"], "A": ss.A.tolist(),
                                                       "settings": ss.settings}, indent=2),
                       "unmix_metrics.json", "application/json", width="stretch")
    c3.download_button("📦 Full bundle (ZIP)", bundle_zip(), "unmix_results.zip", "application/zip", width="stretch")


@st.cache_data(show_spinner=False)
def science_data():
    S = au.standardize(au.load_sources([au.SOURCES_DIR / "speech_female.wav", au.SOURCES_DIR / "music_neon_drift.wav"],
                                       SR, duration_s=4))
    A = np.array([[1.0, 0.7], [0.45, 1.0]])
    wr = pca_whiten(A @ S)
    angles = np.linspace(0, np.pi / 2, 91)
    score = []
    for th in angles:
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        score.append(float(negentropy(R @ wr.Z[:, ::4]).sum()))
    return S, wr.Z, angles, np.array(score)


@st.cache_data(show_spinner=False)
def clt_data():
    names = ["speech_female", "music_neon_drift", "ambient_rain", "speech_male", "ambient_birds"]
    return au.standardize(au.load_sources([au.SOURCES_DIR / f"{n}.wav" for n in names], SR, duration_s=8))


def page_science():
    header("F · The Science Behind the Sound", "Why does this work at all?", "Four ideas — each one you can poke.")
    t1, t2, t3, t4 = st.tabs(["① ICA vs PCA", "② Whitening", "③ Non-Gaussianity & the CLT", "④ Limits"])
    S, Z, angles, score = science_data()
    with t1:
        a, b = st.columns([1, 1])
        with a:
            st.markdown("""
<div class="glass"><h4>Uncorrelated ≠ independent</h4>
<b>PCA</b> uses only <b>second-order statistics</b> (the covariance). It finds orthogonal directions of maximal
variance, which makes the outputs <i>uncorrelated</i>: E[y₁y₂] = 0.<br><br>
<b>ICA</b> asks for much more: <i>statistical independence</i>, p(y₁,y₂) = p(y₁)p(y₂). That constrains
<b>all higher-order moments</b> too, e.g. E[y₁²y₂²] = E[y₁²]E[y₂²]. To get there ICA uses higher-order
information — kurtosis, negentropy — that the covariance cannot see.<br><br>
After whitening <i>every</i> rotation of the data is still uncorrelated, so PCA has no way to choose
between them. That is why PCA alone leaves the sources mixed (see the PCA bars in the Evaluation dashboard).</div>
""", unsafe_allow_html=True)
        with b:
            th = st.slider("Rotate the whitened data (degrees)", 0, 90, 20, key="rot1")
            thr = np.radians(th)
            R = np.array([[np.cos(thr), -np.sin(thr)], [np.sin(thr), np.cos(thr)]])
            Y = R @ Z
            cov = np.cov(Y)[0, 1]
            m4 = np.mean(Y[0] ** 2 * Y[1] ** 2) - np.mean(Y[0] ** 2) * np.mean(Y[1] ** 2)
            plot(viz.scatter_2d(Y, f"θ = {th}°   ·   E[y₁y₂] = {cov:+.3f}   ·   E[y₁²y₂²]−E[y₁²]E[y₂²] = {m4:+.2f}",
                                viz.CYAN), key="rotsc")
            st.caption("The covariance stays ≈ 0 at every angle (PCA is blind to rotation); the 4th-order "
                       "cross-moment only vanishes at the independent angle.")
    with t2:
        a, b = st.columns([1, 1.2])
        with a:
            st.markdown("""
<div class="glass"><h4>Whitening = PCA + rescaling</h4>
Centre the data, then with the eigendecomposition Cov(X) = E D Eᵀ apply
<div class="formula">K = D^-½ Eᵀ ,   Z = K Xc   ⇒   Cov(Z) = I</div>
<b>Why it simplifies ICA:</b> if the sources have unit variance, Cov(Z) = (KA)(KA)ᵀ = I, so KA is
<b>orthogonal</b>. The unknown n×n mixing matrix (n² numbers) becomes an unknown <b>rotation</b>
(n(n−1)/2 numbers — just 1 angle for two sources, 3 for three). FastICA's update + symmetric decorrelation
only ever searches over rotations.</div>""", unsafe_allow_html=True)
        with b:
            A = np.array([[1.0, 0.7], [0.45, 1.0]])
            X = A @ S
            Xc, _ = center(X)
            c1, c2 = st.columns(2)
            with c1:
                plot(viz.scatter_2d(Xc, "Mixed (sheared, correlated)", viz.MIX_COLOR), key="wh1")
            with c2:
                plot(viz.scatter_2d(Z, "Whitened (round covariance, still rotated)", viz.AMBER), key="wh2")
    with t3:
        a, b = st.columns([1, 1.2])
        with a:
            st.markdown("""
<div class="glass"><h4>Mixtures drift toward Gaussian</h4>
The <b>Central Limit Theorem</b>: a sum of independent random variables tends to be more Gaussian than the
terms themselves (under mild conditions). So a mixture y = wᵀx is generally <i>more</i> Gaussian than any single
non-Gaussian source — and it is <i>least</i> Gaussian exactly when w picks out one source.<br><br>
ICA therefore <b>maximises non-Gaussianity</b>, measured by an approximation of negentropy
<div class="formula">J(y) ∝ ( E[G(y)] − E[G(ν)] )²,   G(u) = log cosh u,   ν ~ N(0,1)</div>
Speech is very spiky (super-Gaussian, high kurtosis), which is why it separates so well.</div>""",
                        unsafe_allow_html=True)
        with b:
            Sall = clt_data()
            k = st.slider("Number of independent sources summed", 1, 5, 1, key="clt")
            y = Sall[:k].sum(axis=0)
            plot(viz.histogram_vs_gauss(y, f"Sum of {k} source(s): excess kurtosis = {excess_kurtosis(y[None])[0]:.2f}",
                                        viz.color(k - 1)), key="clthist")
        st.markdown("##### Try it: find the rotation by hand")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            th = st.slider("Rotation angle θ", 0.0, 90.0, 10.0, 0.5, key="rot2")
            plot(viz.rotation_search(Z, np.radians(th), angles, score), key="rs")
        with c2:
            thr = np.radians(th)
            R = np.array([[np.cos(thr), -np.sin(thr)], [np.sin(thr), np.cos(thr)]])
            best = np.degrees(angles[np.argmax(score)])
            plot(viz.scatter_2d(R @ Z, f"rotated whitened data (peak at ≈ {best:.1f}°)", viz.CYAN), key="rs2")
            st.caption("At the peak the arms of the distribution line up with the axes: each output is one source. "
                       "FastICA finds this peak with a few Newton steps instead of a grid search.")
    with t4:
        st.markdown("""
<div class="glass"><h4>When ICA works — and when it can't</h4>
The standard model and its identifiability result assume:
<ul>
<li><b>Instantaneous linear mixing</b> X = AS. Real rooms add propagation delays and reverberation (<i>convolutive</i>
mixing); then time-domain ICA degrades and frequency-domain ICA or beamforming is needed.</li>
<li><b>At least as many observations as sources</b> (m ≥ n), and A of full column rank.</li>
<li><b>Statistically independent sources</b> — two people reading the same script in sync are not.</li>
<li><b>Non-Gaussian sources — at most one Gaussian.</b> Any rotation of two Gaussians is still Gaussian, so ICA has no
signal to lock on to (Experiment E6 in REPORT.md shows exactly this failure).</li>
<li><b>Unavoidable ambiguities:</b> order (permutation), scale and sign of the outputs.</li>
<li><b>Noise</b> is not removed — it is unmixed along with the sources, and amplified when A is ill-conditioned.</li>
</ul></div>""", unsafe_allow_html=True)


def page_real():
    header("Real-World Mode", "Your own recordings", "Record the same scene on 2+ devices (phones, laptop) at the "
           "same time, upload the files, and let ICA have a go. No ground truth → no SI-SDR, just ears and statistics.")
    st.markdown('<div class="note">Real rooms are <b>convolutive</b>: each mic hears delayed, reverberant copies, '
                'and devices have different clocks. Instantaneous ICA will usually give <i>partial</i> separation '
                'here. Best results: devices close together, sources far apart and on opposite sides, little '
                'reverberation, recordings started together. UNMIX aligns start offsets by cross-correlation.</div>',
                unsafe_allow_html=True)
    ups = st.file_uploader("Upload 2–6 simultaneous recordings (WAV)", type=["wav"], accept_multiple_files=True,
                           key="realup")
    c1, c2, c3 = st.columns(3)
    n_comp = c1.number_input("Sources to extract", 2, 6, 2)
    align = c2.toggle("Align device offsets (x-corr)", value=True)
    contrast = c3.selectbox("Contrast", ["logcosh", "exp", "cube"], key="realc")
    if st.button("⚙️ Separate recordings", type="primary", disabled=not ups or len(ups) < 2):
        try:
            set_status("Separating")
            sigs = [au.load_audio(f.getvalue(), SR, f.name) for f in ups]
            X = au.align_lengths(sigs, "trim")
            lags = [0] * len(sigs)
            if align:
                X, lags = au.align_by_xcorr(X, SR)
            if n_comp > X.shape[0]:
                raise ValueError("Need at least as many recordings as sources to extract.")
            r = run_ica(X, n_components=int(n_comp), contrast=contrast)
            ss.real = {"X": X, "r": r, "lags": lags, "names": [f.name for f in ups]}
            set_status("Completed")
        except (au.AudioError, ValueError) as e:
            set_status("Error")
            st.error(str(e))
    if ss.real:
        X, r = ss.real["X"], ss.real["r"]
        st.caption("Detected start offsets: " + ", ".join(f"{n}: {l / SR * 1000:+.1f} ms"
                                                          for n, l in zip(ss.real["names"], ss.real["lags"])))
        a, b = st.columns(2)
        with a:
            plot(viz.waveform_stack(list(X), ss.real["names"], SR, colors=[viz.MIX_COLOR] * len(X),
                                    title="Recordings (aligned)"), key="rw1")
        with b:
            plot(viz.waveform_stack(list(r.sources), [f"Component {i + 1}" for i in range(len(r.sources))], SR,
                                    title="ICA components"), key="rw2")
        kin, kout = excess_kurtosis(X), excess_kurtosis(r.sources)
        st.markdown(f"Independence proxies — mean excess kurtosis: recordings **{kin.mean():.2f}** → components "
                    f"**{kout.mean():.2f}**; mean negentropy {negentropy(X).mean():.4f} → {negentropy(r.sources).mean():.4f}. "
                    "Higher non-Gaussianity after ICA suggests the components are closer to individual sources.")
        cols = st.columns(len(r.sources))
        for i, c in enumerate(cols):
            data = au.wav_bytes(r.sources[i], SR)
            c.audio(data, format="audio/wav")
            c.download_button(f"⬇ Component {i + 1}", data, f"realworld_component_{i + 1}.wav", "audio/wav",
                              key=f"rwdl{i}", width="stretch")


def page_shazam():
    header("Optional extension", "Mini-Shazam 🎵", "Landmark fingerprinting over a 6-song library — not ICA, but a "
           "nice demonstration of why separation matters: identify a song hidden under speech and rain.")
    import fingerprint as fp
    db = fingerprint_db()
    with st.expander("📚 The library (6 procedurally composed tracks)"):
        cols = st.columns(3)
        for i, p in enumerate(sorted(au.LIBRARY_DIR.glob("*.wav"))):
            with cols[i % 3]:
                st.caption(db.titles[i])
                st.audio(p.read_bytes(), format="audio/wav")
    opts = []
    if ss.X is not None:
        opts += [f"Mixture · Mic {i + 1}" for i in range(len(ss.X))]
    if ss.result is not None:
        opts += [f"ICA component {k + 1}" for k in range(len(ss.result.sources))]
    opts += ["Upload a clip"]
    q = st.radio("Query", opts, horizontal=True)
    x = None
    if q.startswith("Mixture"):
        x = ss.X[int(q.split()[-1]) - 1]
    elif q.startswith("ICA"):
        x = ss.result.sources[int(q.split()[-1]) - 1]
    else:
        f = st.file_uploader("WAV clip (a few seconds is enough)", type=["wav"], key="shz")
        if f:
            try:
                x = au.load_audio(f.getvalue(), SR, f.name)
            except au.AudioError as e:
                st.error(str(e))
    if x is not None:
        st.audio(au.wav_bytes(x, SR), format="audio/wav")
        if st.button("🔎 Identify", type="primary"):
            rk = db.identify(x, SR)
            conf = fp.confidence(rk)
            best = rk[0]
            verdict = best["title"] if best["score"] >= 8 and conf > 0.4 else "No confident match"
            k = st.columns(3)
            metric_card(k[0], "match", verdict, "#22d3ee" if verdict != "No confident match" else "#fb7185")
            metric_card(k[1], "aligned hash votes", f"{best['score']}", "#a78bfa")
            metric_card(k[2], "confidence", f"{conf:.0%}", "#a3e635")
            plot(viz.bars([r_["title"] for r_ in rk], {"votes": [r_["score"] for r_ in rk]},
                          "Votes for the most consistent time offset", "votes", 300), key="shzbar")
            st.caption("Tip: run the ⚡ demo and compare a mixture with the ICA component that holds the music.")


{PAGES[0]: page_home, PAGES[1]: page_studio, PAGES[2]: page_mixing, PAGES[3]: page_engine, PAGES[4]: page_reveal,
 PAGES[5]: page_eval, PAGES[6]: page_science, PAGES[7]: page_real, PAGES[8]: page_shazam}[page]()

st.markdown('<div style="text-align:center;color:#555c6e;margin-top:40px;font-size:.8rem">UNMIX · FastICA & PCA '
            'whitening implemented from scratch with NumPy · no scikit-learn</div>', unsafe_allow_html=True)
