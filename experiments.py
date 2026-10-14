"""
experiments.py — reproducible ICA experiments. Every number in REPORT.md comes from here.

    python experiments.py            # full suite (~1 min)
    python experiments.py --quick    # fewer random trials

Writes
    outputs/results.json               all metrics
    outputs/figures/*.png              plots used in REPORT.md / README
    outputs/separated/<experiment>/    originals, mixtures, ICA + PCA outputs as WAV
    REPORT.md                          auto-generated write-up with the actual numbers
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import audio_utils as au
import evaluation as ev
import visualization as viz
from ica import excess_kurtosis, run_ica, run_pca

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SEP = OUT / "separated"
SR = 16_000
DURATION = 8.0

SOURCE_FILES = {
    "Speech (female)": "speech_female.wav",
    "Music": "music_neon_drift.wav",
    "Rain": "ambient_rain.wav",
    "Speech (male)": "speech_male.wav",
    "Birds": "ambient_birds.wav",
}
# Hand-written "custom" mixing matrix for the headline experiment: three mics in a
# room, each closest to one source but hearing all of them.
A_CUSTOM = np.array([[1.00, 0.60, 0.45],
                     [0.55, 1.00, 0.50],
                     [0.40, 0.65, 1.00]])


def load(names):
    S = au.load_sources([au.SOURCES_DIR / SOURCE_FILES[n] for n in names], SR, DURATION)
    return au.standardize(S)


def separate_and_score(S, A, names, snr_db=None, seed=0, contrast="logcosh", algorithm="symmetric",
                       n_components=None):
    X = au.add_noise(au.mix(S, A), snr_db, seed=seed)
    k = n_components or S.shape[0]
    t0 = time.perf_counter()
    r = run_ica(X, n_components=k, contrast=contrast, algorithm=algorithm, seed=seed)
    dt = time.perf_counter() - t0
    e_ica = ev.evaluate(S, r.sources, X=X, W_total=r.unmixing, A=A if A.shape[0] == A.shape[1] else None,
                        names=names)
    e_pca = ev.evaluate(S, run_pca(X, k), X=X, names=names)
    return X, r, e_ica, e_pca, dt


def r3(x):
    return float(np.round(x, 3))


# ---------------------------------------------------------------------------

def exp_baseline(results):
    """E1 — speech + music + rain, custom 3x3 matrix, no noise."""
    names = ["Speech (female)", "Music", "Rain"]
    S = load(names)
    X, r, e, p, dt = separate_and_score(S, A_CUSTOM, names)
    folder = SEP / "e1_baseline"
    for i, n in enumerate(names):
        slug = n.lower().replace(" ", "_").replace("(", "").replace(")", "")
        au.save_wav(folder / f"original_{i + 1}_{slug}.wav", S[i], SR)
        au.save_wav(folder / f"mixture_mic{i + 1}.wav", X[i], SR)
        au.save_wav(folder / f"ica_component_{i + 1}.wav", r.sources[i], SR)
        au.save_wav(folder / f"pca_component_{i + 1}.wav", run_pca(X)[i], SR)
        au.save_wav(folder / f"recovered_{slug}.wav", e["aligned"][i], SR)
    viz.mpl_reveal(S, X, e["aligned"], names, SR, FIG / "e1_reveal.png")
    viz.mpl_spectrograms([S[0], X[0], e["aligned"][0]],
                         ["Speech — original", "Mic 1 — mixture", "Speech — recovered by ICA"], SR,
                         FIG / "e1_spectrograms.png")
    Xc = X - X.mean(1, keepdims=True)
    viz.mpl_scatter_panels([
        ("Sources: speech vs music", S[:2], viz.VIOLET),
        ("Mixtures: mic 1 vs mic 2", Xc[:2], viz.MIX_COLOR),
        ("After PCA whitening", r.whitening.Z[:2], viz.AMBER),
        ("After ICA rotation", r.sources[[e["perm"][0], e["perm"][1]]], viz.CYAN),
    ], FIG / "e1_scatter.png")
    viz.mpl_heatmap(np.abs(np.array(e["corr_matrix"])), names, [f"IC {i + 1}" for i in range(3)],
                    "|correlation| source vs ICA component", FIG / "e1_corr.png")
    viz.mpl_bar(names, {"best single mic (input)": [x["input_si_sdr"] for x in e["rows"]],
                        "PCA only": [x["si_sdr"] for x in p["rows"]],
                        "ICA (ours)": [x["si_sdr"] for x in e["rows"]]},
                "E1 — SI-SDR per source (higher is better)", "SI-SDR (dB)", FIG / "e1_pca_vs_ica.png")
    results["e1_baseline"] = {
        "sources": names, "A": A_CUSTOM.tolist(), "cond_A": r3(np.linalg.cond(A_CUSTOM)),
        "ica": {"rows": e["rows"], "summary": e["summary"]}, "pca": {"rows": p["rows"], "summary": p["summary"]},
        "n_iter": r.n_iter, "converged": r.converged, "seconds": r3(dt),
        "unmixing_times_A": (r.unmixing @ A_CUSTOM).round(4).tolist(),
        "whitened_cov_max_err": float(np.max(np.abs(np.cov(r.whitening.Z) - np.eye(3)))),
        "eigvals": r.whitening.eigvals.tolist(),
    }


def exp_matrices(results, n_trials):
    """E2 — how the mixing matrix affects separation: random, ill-conditioned, overdetermined."""
    names = ["Speech (female)", "Music", "Rain"]
    S = load(names)
    rows = []
    rng = np.random.default_rng(2024)
    for t in range(n_trials):
        A = rng.uniform(-1, 1, (3, 3))
        _, _, e0, _, _ = separate_and_score(S, A, names, seed=t)              # noiseless
        _, r, e, p, _ = separate_and_score(S, A, names, snr_db=30.0, seed=t)  # 30 dB sensor noise
        rows.append({"trial": t, "cond": r3(np.linalg.cond(A)),
                     "ica_si_sdr_clean": r3(e0["summary"]["mean_si_sdr"]),
                     "ica_si_sdr": r3(e["summary"]["mean_si_sdr"]),
                     "pca_si_sdr": r3(p["summary"]["mean_si_sdr"]), "amari": r3(e["summary"]["amari_index"])})
    special = {}
    A_ill = np.array([[1.0, 0.8, 0.6], [1.0, 0.8, 0.6005], [0.3, 0.5, 1.0]])
    for label, A, snr in [("custom (well-conditioned)", A_CUSTOM, None),
                          ("ill-conditioned, clean", A_ill, None),
                          ("ill-conditioned, 30 dB SNR", A_ill, 30.0)]:
        _, r, e, p, _ = separate_and_score(S, A, names, snr_db=snr)
        special[label] = {"cond": r3(np.linalg.cond(A)), "ica_si_sdr": r3(e["summary"]["mean_si_sdr"]),
                          "pca_si_sdr": r3(p["summary"]["mean_si_sdr"]), "amari": r3(e["summary"]["amari_index"])}
    # Overdetermined: 3 sources, 5 microphones, 20 dB noise -> PCA reduction to 3 dims.
    A5 = au.random_mixing_matrix(5, 3, seed=5)
    for m, A in [(3, A5[:3]), (5, A5)]:
        _, r, e, p, _ = separate_and_score(S, A, names, snr_db=20.0, n_components=3)
        special[f"{m} mics, 20 dB SNR"] = {"cond": r3(np.linalg.cond(A)), "ica_si_sdr": r3(e["summary"]["mean_si_sdr"]),
                                           "pca_si_sdr": r3(p["summary"]["mean_si_sdr"]), "amari": None}
    conds = np.array([x["cond"] for x in rows])
    order = np.argsort(conds)
    viz.mpl_lines(np.log10(conds[order]), {"ICA": [rows[i]["ica_si_sdr"] for i in order],
                                           "PCA only": [rows[i]["pca_si_sdr"] for i in order]},
                  f"E2 — {n_trials} random mixing matrices (30 dB SNR)", "log10 condition number of A",
                  "mean SI-SDR (dB)", FIG / "e2_random_matrices.png")
    ica_vals = np.array([x["ica_si_sdr"] for x in rows])
    clean_vals = np.array([x["ica_si_sdr_clean"] for x in rows])
    results["e2_matrices"] = {"random_trials": rows, "special": special,
                              "random_summary": {"clean_min": r3(clean_vals.min()), "clean_max": r3(clean_vals.max()),
                                                 "corr_cond_vs_sdr": r3(np.corrcoef(np.log10(conds), ica_vals)[0, 1]),"ica_median": r3(np.median(ica_vals)), "ica_min": r3(ica_vals.min()),
                                                 "ica_max": r3(ica_vals.max()),
                                                 "pca_median": r3(np.median([x["pca_si_sdr"] for x in rows])),
                                                 "frac_ica_above_20db": r3(np.mean(ica_vals > 20))}}


def exp_snr(results, seeds):
    """E3 — separation quality vs additive sensor noise, ICA vs PCA vs doing nothing."""
    names = ["Speech (female)", "Music", "Rain"]
    S = load(names)
    snrs = [None, 40, 30, 20, 10, 5, 0]
    out = []
    for snr in snrs:
        vals = {"ica": [], "pca": [], "input": [], "corr": []}
        for s in range(seeds):
            _, r, e, p, _ = separate_and_score(S, A_CUSTOM, names, snr_db=snr, seed=s)
            vals["ica"].append(e["summary"]["mean_si_sdr"])
            vals["pca"].append(p["summary"]["mean_si_sdr"])
            vals["input"].append(e["summary"]["mean_input_si_sdr"])
            vals["corr"].append(e["summary"]["mean_abs_corr"])
        out.append({"snr_db": "clean" if snr is None else snr, **{k: r3(np.mean(v)) for k, v in vals.items()}})
    labels = [str(o["snr_db"]) for o in out]
    viz.mpl_lines(range(len(out)), {"ICA": [o["ica"] for o in out], "PCA only": [o["pca"] for o in out],
                                    "best single mic": [o["input"] for o in out]},
                  "E3 — separation vs sensor noise", "input SNR (dB)", "mean SI-SDR (dB)",
                  FIG / "e3_snr_sweep.png", xticklabels=labels)
    results["e3_snr"] = out


def exp_contrasts(results):
    """E4 — contrast function and symmetric vs deflation."""
    names = ["Speech (female)", "Music", "Rain", "Speech (male)"]
    S = load(names)
    A = au.random_mixing_matrix(4, 4, seed=7)
    out = []
    for contrast in ["logcosh", "exp", "cube"]:
        for algo in ["symmetric", "deflation"]:
            _, r, e, _, dt = separate_and_score(S, A, names, snr_db=30, contrast=contrast, algorithm=algo)
            out.append({"contrast": contrast, "algorithm": algo, "mean_si_sdr": r3(e["summary"]["mean_si_sdr"]),
                        "min_si_sdr": r3(min(x["si_sdr"] for x in e["rows"])), "iterations": r.n_iter,
                        "converged": r.converged, "seconds": r3(dt)})
    viz.mpl_bar([f"{o['contrast']}\n{o['algorithm']}" for o in out],
                {"mean SI-SDR": [o["mean_si_sdr"] for o in out], "worst source": [o["min_si_sdr"] for o in out]},
                "E4 — contrast functions (4 sources, 30 dB SNR)", "SI-SDR (dB)", FIG / "e4_contrasts.png")
    results["e4_contrasts"] = out


def exp_nsources(results):
    """E5 — scaling from 2 to 5 sources (square mixing, 30 dB SNR)."""
    order = ["Speech (female)", "Music", "Rain", "Speech (male)", "Birds"]
    out = []
    for n in range(2, 6):
        names = order[:n]
        S = load(names)
        A = au.random_mixing_matrix(n, n, seed=100 + n)
        _, r, e, p, dt = separate_and_score(S, A, names, snr_db=30)
        out.append({"n_sources": n, "ica": r3(e["summary"]["mean_si_sdr"]), "pca": r3(p["summary"]["mean_si_sdr"]),
                    "input": r3(e["summary"]["mean_input_si_sdr"]), "iterations": r.n_iter, "seconds": r3(dt)})
        if n == 4:  # cocktail party: two speakers + music + rain -> export
            folder = SEP / "e5_cocktail_party_4src"
            Xn = au.add_noise(au.mix(S, A), 30, seed=0)
            for i, nm in enumerate(names):
                slug = nm.lower().replace(" ", "_").replace("(", "").replace(")", "")
                au.save_wav(folder / f"mixture_mic{i + 1}.wav", Xn[i], SR)
                au.save_wav(folder / f"recovered_{slug}.wav", e["aligned"][i], SR)
    viz.mpl_lines([o["n_sources"] for o in out], {"ICA": [o["ica"] for o in out], "PCA only": [o["pca"] for o in out],
                                                 "best single mic": [o["input"] for o in out]},
                  "E5 — number of sources (30 dB SNR)", "sources = microphones", "mean SI-SDR (dB)",
                  FIG / "e5_nsources.png")
    results["e5_nsources"] = out


def exp_gaussian(results):
    """E6 — identifiability: what happens when sources are Gaussian?"""
    rng = np.random.default_rng(3)
    T = int(SR * DURATION)
    real = load(["Speech (female)", "Music"])
    g1, g2 = rng.standard_normal(T), rng.standard_normal(T)
    cases = {
        "0 Gaussian (speech, music, rain)": (load(["Speech (female)", "Music", "Rain"]), ["Speech", "Music", "Rain"]),
        "1 Gaussian (speech, music, noise)": (np.vstack([real, g1]), ["Speech", "Music", "Gaussian A"]),
        "2 Gaussian (speech, noise, noise)": (np.vstack([real[:1], g1, g2]), ["Speech", "Gaussian A", "Gaussian B"]),
    }
    out = {}
    for label, (S, names) in cases.items():
        S = au.standardize(S)
        _, r, e, _, _ = separate_and_score(S, A_CUSTOM, names)
        out[label] = {x["source"]: r3(x["si_sdr"]) for x in e["rows"]}
        out[label]["amari"] = r3(e["summary"]["amari_index"])
        out[label]["kurtosis"] = [r3(k) for k in excess_kurtosis(S)]
    results["e6_gaussian"] = out


def exp_clt(results):
    """E7 — Central Limit Theorem: mixing makes signals more Gaussian."""
    names = list(SOURCE_FILES)
    S = load(names)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(1, 6):
        kurts = []
        for _ in range(30):
            idx = rng.choice(5, k, replace=False)
            w = rng.uniform(0.5, 1.0, k)
            kurts.append(excess_kurtosis(w @ S[idx])[0])
        rows.append({"n_mixed": k, "mean_excess_kurtosis": r3(np.mean(kurts))})
    viz.mpl_lines([r["n_mixed"] for r in rows], {"mean excess kurtosis": [r["mean_excess_kurtosis"] for r in rows]},
                  "E7 — CLT: summing more independent sources → closer to Gaussian (0)",
                  "number of sources summed", "excess kurtosis", FIG / "e7_clt.png")
    results["e7_clt"] = {"per_source_kurtosis": dict(zip(names, [r3(k) for k in excess_kurtosis(S)])), "sums": rows}


def exp_fingerprint(results):
    """E8 (optional extension) — identify the song before vs after ICA."""
    import fingerprint as fp
    db = fp.FingerprintDB.from_library()
    names = ["Speech (female)", "Music", "Rain"]
    S = load(names)
    A = np.array([[1.0, 0.35, 0.8], [0.7, 0.3, 1.0], [1.0, 0.4, 0.5]])  # music is quiet everywhere
    X = au.mix(S, A)
    r = run_ica(X)
    out = {"mixtures": [], "ica_components": []}
    for key, sigs in [("mixtures", X), ("ica_components", r.sources)]:
        for x in sigs:
            rk = db.identify(x, SR)
            out[key].append({"top": rk[0]["title"], "score": rk[0]["score"], "runner_up": rk[1]["score"],
                             "confidence": r3(fp.confidence(rk))})
    results["e8_fingerprint"] = out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, d=1):
    return "—" if x is None else f"{x:.{d}f}"


def write_report(R: dict, path: Path):
    e1 = R["e1_baseline"]
    L = ["# UNMIX — Experiment Report",
         "",
         "_Auto-generated by `python experiments.py` — every number below is computed, not typed. "
         f"Audio: {DURATION:.0f} s at {SR} Hz; sources standardised to unit variance; FastICA (log-cosh, symmetric) "
         "on PCA-whitened data; tolerance 1e-6._",
         "",
         "**Metrics.** |r| = absolute Pearson correlation of each source with its Hungarian-matched component. "
         "SI-SDR = scale-invariant signal-to-distortion ratio (dB, sign/scale-invariant). "
         "*Input* = SI-SDR of the best single microphone for that source (what you get without any processing). "
         "Amari index = 0 for perfect separation (uses the true A).",
         "",
         "## E1 — Baseline: speech + music + rain, custom 3×3 mixing matrix",
         "",
         f"Mixing matrix A (cond = {e1['cond_A']}):",
         "",
         "| | Speech | Music | Rain |", "|---|---|---|---|"]
    for i, row in enumerate(e1["A"]):
        L.append(f"| Mic {i + 1} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    L += ["", f"Whitening check: max |Cov(Z) − I| = {e1['whitened_cov_max_err']:.2e}. "
              f"FastICA converged = {e1['converged']} after {e1['n_iter']} iterations ({e1['seconds']:.2f} s).", "",
          "| Source | matched IC | sign | |r| | SI-SDR (dB) | SIR (dB) | input SI-SDR | PCA SI-SDR |",
          "|---|---|---|---|---|---|---|---|"]
    for a, b in zip(e1["ica"]["rows"], e1["pca"]["rows"]):
        L.append(f"| {a['source']} | {a['matched_component']} | {'+' if a['sign'] > 0 else '−'} | {a['abs_corr']:.4f} | "
                 f"{a['si_sdr']:.1f} | {a['sir']:.1f} | {a['input_si_sdr']:.1f} | {b['si_sdr']:.1f} |")
    s = e1["ica"]["summary"]
    L += ["", f"Mean SI-SDR: **ICA {s['mean_si_sdr']:.1f} dB** vs PCA {e1['pca']['summary']['mean_si_sdr']:.1f} dB vs "
              f"input {s['mean_input_si_sdr']:.1f} dB (improvement {s['mean_si_sdr_improvement']:.1f} dB). "
              f"Amari index {s['amari_index']:.4f}.", "",
          "Global matrix W·A (should be a scaled permutation):", "", "```"]
    L += ["  ".join(f"{v:+.4f}" for v in row) for row in e1["unmixing_times_A"]]
    L += ["```", "", "![reveal](outputs/figures/e1_reveal.png)", "", "![scatter](outputs/figures/e1_scatter.png)", "",
          "The scatter plots tell the whole story: the sources form a '+' shape (independent, sparse); mixing shears it; "
          "whitening makes it round in covariance but still rotated; the ICA rotation aligns the arms with the axes.", "",
          "![spectrograms](outputs/figures/e1_spectrograms.png)", "", "![pca vs ica](outputs/figures/e1_pca_vs_ica.png)", ""]

    e2 = R["e2_matrices"]
    rs = e2["random_summary"]
    L += ["## E2 — Different mixing matrices", "",
          f"{len(e2['random_trials'])} random matrices with entries U(−1, 1).", "",
          f"* **No noise:** ICA SI-SDR is {rs['clean_min']:.1f}–{rs['clean_max']:.1f} dB for *every* matrix. This is the "
          "*equivariance* property of FastICA: after whitening, the separation result does not depend on A at all "
          "(the algorithm only sees K·A·S, and K·A is always an orthogonal matrix).",
          f"* **30 dB sensor noise:** ICA median **{rs['ica_median']:.1f} dB** (min {rs['ica_min']:.1f}, max "
          f"{rs['ica_max']:.1f}); {rs['frac_ica_above_20db'] * 100:.0f}% of trials above 20 dB; PCA median "
          f"{rs['pca_median']:.1f} dB. Correlation between log10 cond(A) and SI-SDR: {rs['corr_cond_vs_sdr']:.2f}.", "",
          "| Condition | cond(A) | ICA SI-SDR | PCA SI-SDR | Amari |", "|---|---|---|---|---|"]
    for k, v in e2["special"].items():
        L.append(f"| {k} | {v['cond']:.0f} | {v['ica_si_sdr']:.1f} | {v['pca_si_sdr']:.1f} | {fmt(v['amari'], 4)} |")
    L += ["", "Without noise the condition number barely matters — ICA inverts A exactly in theory. With noise, "
              "an ill-conditioned A means the unmixing matrix has huge gains that amplify the noise. Extra microphones "
              "(overdetermined case) help because PCA keeps the 3 signal dimensions and discards noise-only ones.", "",
          "![random matrices](outputs/figures/e2_random_matrices.png)", ""]

    L += ["## E3 — Sensor-noise sweep (mean of seeds)", "",
          "| input SNR (dB) | ICA SI-SDR | PCA SI-SDR | best mic | ICA mean |r| |", "|---|---|---|---|---|"]
    for o in R["e3_snr"]:
        L.append(f"| {o['snr_db']} | {o['ica']:.1f} | {o['pca']:.1f} | {o['input']:.1f} | {o['corr']:.3f} |")
    L += ["", "The noiseless model X = AS is exactly invertible, so the clean result is limited only by finite-sample "
              "dependence between the sources. With additive noise, the unmixing matrix also passes noise through: "
              "ICA output quality tracks the input SNR — ICA separates *sources*, it does not denoise.", "",
          "![snr](outputs/figures/e3_snr_sweep.png)", ""]

    L += ["## E4 — Contrast functions & extraction strategy (4 sources, 30 dB SNR)", "",
          "| contrast | algorithm | mean SI-SDR | worst source | iterations | time (s) |", "|---|---|---|---|---|---|"]
    for o in R["e4_contrasts"]:
        L.append(f"| {o['contrast']} | {o['algorithm']} | {o['mean_si_sdr']:.1f} | {o['min_si_sdr']:.1f} | "
                 f"{o['iterations']} | {o['seconds']:.2f} |")
    L += ["", "![contrasts](outputs/figures/e4_contrasts.png)", ""]

    L += ["## E5 — Number of sources (square mixing, 30 dB SNR)", "",
          "| sources | ICA | PCA | best mic | iterations |", "|---|---|---|---|---|"]
    for o in R["e5_nsources"]:
        L.append(f"| {o['n_sources']} | {o['ica']:.1f} | {o['pca']:.1f} | {o['input']:.1f} | {o['iterations']} |")
    L += ["", "![n sources](outputs/figures/e5_nsources.png)", ""]

    L += ["## E6 — Identifiability: Gaussian sources (clean, custom A)", "",
          "| case | per-source SI-SDR (dB) | Amari | excess kurtosis of sources |", "|---|---|---|---|"]
    for k, v in R["e6_gaussian"].items():
        per = ", ".join(f"{n}: {x:.1f}" for n, x in v.items() if n not in ("amari", "kurtosis"))
        L.append(f"| {k} | {per} | {v['amari']:.3f} | {', '.join(f'{x:.2f}' for x in v['kurtosis'])} |")
    L += ["", "With one Gaussian source ICA still works (it is identified as 'whatever is left'). With two, any "
              "rotation of the two Gaussians is equally Gaussian, so ICA cannot find the original pair: their SI-SDR "
              "collapses while the non-Gaussian speech is still recovered.", ""]

    c = R["e7_clt"]
    L += ["## E7 — Central Limit Theorem in action", "",
          "Excess kurtosis of each clean source: " + ", ".join(f"{k} {v:.2f}" for k, v in c["per_source_kurtosis"].items()) + ".", "",
          "| sources summed | mean excess kurtosis |", "|---|---|"]
    L += [f"| {r['n_mixed']} | {r['mean_excess_kurtosis']:.2f} |" for r in c["sums"]]
    L += ["", "![clt](outputs/figures/e7_clt.png)", ""]

    fpr = R["e8_fingerprint"]
    L += ["## E8 — Optional extension: mini-Shazam before vs after ICA", "",
          "Library of 6 procedural songs; music mixed quietly under speech and rain.", "",
          "| signal | top match | votes | runner-up votes | confidence |", "|---|---|---|---|---|"]
    for k in ("mixtures", "ica_components"):
        for i, o in enumerate(fpr[k]):
            lab = f"Mic {i + 1}" if k == "mixtures" else f"ICA component {i + 1}"
            L.append(f"| {lab} | {o['top']} | {o['score']} | {o['runner_up']} | {o['confidence']:.2f} |")
    L += ["", "## Limitations", "",
          "* Mixtures are synthetic and *instantaneous* (X = AS). Real rooms add delays and reverberation "
          "(convolutive mixing), which this model does not capture — see README for the real-world mode caveats.",
          "* Metrics need ground truth; for real recordings only listening and independence proxies are available.",
          "* Sources are short (8 s); finite-sample dependence between them caps the clean SI-SDR.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    R: dict = {"config": {"sr": SR, "duration_s": DURATION, "contrast": "logcosh", "algorithm": "symmetric"}}
    steps = [("E1 baseline", lambda: exp_baseline(R)),
             ("E2 mixing matrices", lambda: exp_matrices(R, 8 if args.quick else 30)),
             ("E3 SNR sweep", lambda: exp_snr(R, 1 if args.quick else 3)),
             ("E4 contrasts", lambda: exp_contrasts(R)),
             ("E5 number of sources", lambda: exp_nsources(R)),
             ("E6 Gaussian sources", lambda: exp_gaussian(R)),
             ("E7 CLT", lambda: exp_clt(R)),
             ("E8 fingerprint", lambda: exp_fingerprint(R))]
    for name, fn in steps:
        t = time.perf_counter()
        fn()
        print(f"  ✓ {name:<22} {time.perf_counter() - t:5.1f} s")
    (OUT / "results.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    write_report(R, ROOT / "REPORT.md")
    print("Wrote outputs/results.json, outputs/figures/, outputs/separated/, REPORT.md")


if __name__ == "__main__":
    main()
