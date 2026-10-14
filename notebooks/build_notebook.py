"""Builds and executes notebooks/unmix_walkthrough.ipynb (so it ships with real outputs).

    python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# UNMIX — ICA walkthrough\n\n"
       "Step-by-step: mix three clean sources, **center**, **whiten by hand** (covariance + eigendecomposition), "
       "run **FastICA from scratch**, then evaluate against the originals while accounting for permutation, sign and "
       "scale ambiguity. Every cell calls the same modules the app uses (`whitening.py`, `ica.py`, `evaluation.py`)."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import numpy as np, matplotlib.pyplot as plt\n"
         "from IPython.display import Audio, display, Markdown\n"
         "import audio_utils as au, evaluation as ev, visualization as viz\n"
         "from whitening import center, covariance, pca_whiten, whiteness_error\n"
         "from ica import fastica, run_ica, run_pca, excess_kurtosis\n"
         "plt = viz.mpl_style(); import matplotlib; matplotlib.use('module://matplotlib_inline.backend_inline')\n"
         "SR = 16_000"),
    md("## 1 · Load sources (resample → mono → common length → standardise)"),
    code("names = ['Speech (female)', 'Music', 'Rain']\n"
         "files = ['speech_female.wav', 'music_neon_drift.wav', 'ambient_rain.wav']\n"
         "S = au.standardize(au.load_sources([au.SOURCES_DIR / f for f in files], SR, duration_s=8))\n"
         "print(S.shape, 'excess kurtosis:', excess_kurtosis(S).round(2))"),
    md("## 2 · Mix with our own matrix: X = A S"),
    code("A = np.array([[1.00, 0.60, 0.45],\n              [0.55, 1.00, 0.50],\n              [0.40, 0.65, 1.00]])\n"
         "X = au.mix(S, A)\nprint('cond(A) =', np.linalg.cond(A).round(2))\n"
         "print('mixture kurtosis:', excess_kurtosis(X).round(2), ' <- closer to 0 (Gaussian): CLT at work')\n"
         "display(Audio(X[0][:4*SR], rate=SR))"),
    md("## 3 · Center and whiten — by hand\n\n"
       "$C = X_c X_c^T/(T-1) = E D E^T$, $K = D^{-1/2}E^T$, $Z = K X_c$ ⇒ $\\mathrm{Cov}(Z)=I$."),
    code("Xc, mean = center(X)\nC = covariance(Xc)\nd, E = np.linalg.eigh(C)\n"
         "print('Cov(X):\\n', C.round(3))\nprint('eigenvalues:', d[::-1].round(3))\n"
         "wr = pca_whiten(X)\nprint('Cov(Z):\\n', np.cov(wr.Z).round(6))\nprint('max |Cov(Z)-I| =', whiteness_error(wr.Z))\n"
         "KA = wr.K @ A\n"
         "print('(K A)(K A)^T =\\n', (KA @ KA.T).round(3))\n"
         "print('≈ I: whitening turned the unknown mixing into a rotation (exact only if the finite-sample sources were exactly uncorrelated)')"),
    md("## 4 · FastICA (log-cosh contrast, symmetric decorrelation)"),
    code("W, n_iter, ok, hist = fastica(wr.Z, contrast='logcosh', algorithm='symmetric', seed=0)\n"
         "Y = W @ wr.Z\nprint(f'converged={ok} after {n_iter} iterations')\n"
         "print('W W^T =\\n', (W @ W.T).round(10))\nprint('global (W K) A =\\n', (W @ wr.K @ A).round(3))\n"
         "plt.figure(figsize=(6,2.6)); plt.semilogy(np.maximum(hist,1e-16), 'o-', color=viz.CYAN); plt.title('convergence'); plt.xlabel('iteration'); plt.show()"),
    md("## 5 · Evaluate — Hungarian matching, sign/scale-invariant metrics"),
    code("res = ev.evaluate(S, Y, X=X, W_total=W @ wr.K, A=A, names=names)\n"
         "pca = ev.evaluate(S, run_pca(X), X=X, names=names)\n"
         "import pandas as pd\n"
         "df = pd.DataFrame(res['rows'])[['source','matched_component','sign','abs_corr','si_sdr','sir','input_si_sdr']]\n"
         "df['pca_si_sdr'] = [r['si_sdr'] for r in pca['rows']]\ndisplay(df.round(3))\n"
         "print({k: round(v, 4) for k, v in res['summary'].items()})"),
    code("viz.mpl_reveal(S, X, res['aligned'], names, SR, 'reveal_nb.png')\n"
         "from IPython.display import Image; display(Image('reveal_nb.png'))"),
    md("### Listen: recovered sources (sign & scale aligned)"),
    code("for n, y in zip(names, res['aligned']):\n    display(Markdown(f'**{n}**')); display(Audio(y[:4*SR], rate=SR))"),
    md("## 6 · PCA vs ICA under noise"),
    code("rows = []\nfor snr in [None, 30, 20, 10, 0]:\n"
         "    Xn = au.add_noise(X, snr, seed=0)\n"
         "    r = run_ica(Xn)\n"
         "    rows.append({'SNR': 'clean' if snr is None else snr,\n"
         "                 'ICA SI-SDR': ev.evaluate(S, r.sources)['summary']['mean_si_sdr'],\n"
         "                 'PCA SI-SDR': ev.evaluate(S, run_pca(Xn))['summary']['mean_si_sdr'],\n"
         "                 'best mic': ev.evaluate(S, r.sources, X=Xn)['summary']['mean_input_si_sdr']})\n"
         "pd.DataFrame(rows).round(2)"),
    md("## 7 · Why Gaussian sources break ICA"),
    code("rng = np.random.default_rng(0)\n"
         "G = au.standardize(np.vstack([S[0], rng.standard_normal(S.shape[1]), rng.standard_normal(S.shape[1])]))\n"
         "r = run_ica(A @ G)\nfor row in ev.evaluate(G, r.sources, names=['speech','gauss A','gauss B'])['rows']:\n"
         "    print(f\"{row['source']:8s}  SI-SDR {row['si_sdr']:6.1f} dB\")\n"
         "print('-> the two Gaussian sources are not identifiable; the non-Gaussian speech still is.')"),
    md("See `REPORT.md` for the full experiment suite (random matrices, SNR sweep, contrasts, number of sources, CLT, "
       "fingerprinting)."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                               "language": "python"}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "unmix_walkthrough.ipynb")
(HERE / "reveal_nb.png").unlink(missing_ok=True)
print("wrote", HERE / "unmix_walkthrough.ipynb")
