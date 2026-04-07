import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import f
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import weibull_min


CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7hh_4594_k28.csv"


def load_draws(path):
    df = pd.read_csv(path)
    cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
    if all(c in df.columns for c in cols):
        d = df[cols].copy()
    else:
        d = df.iloc[:, :7].copy()
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return d.values.astype(int)


def prepare_stats(draws):
    counts = np.bincount(draws.ravel(), minlength=40)[1:].astype(float)  # 1..39
    mean_c = counts.mean()
    std_c = counts.std() if counts.std() > 0 else 1.0
    z = (counts - mean_c) / std_c

    gaps = np.zeros(39, dtype=float)
    n = len(draws)
    for num in range(1, 40):
        idx = np.where(draws == num)[0]
        gaps[num - 1] = n if len(idx) == 0 else (n - 1 - idx.max())

    return counts, z, gaps


def pick_top7(score, counts, gaps):
    # Deterministički tie-break: veći score, manji count, veći gap, manji broj
    order = sorted(
        range(1, 40),
        key=lambda n: (-float(score[n - 1]), float(counts[n - 1]), -float(gaps[n - 1]), n),
    )
    pred = np.array(sorted(order[:7]), dtype=int)
    return pred


def main():
    draws = load_draws(CSV_PATH)
    counts, z, gaps = prepare_stats(draws)

    # 1) Normal
    s_norm = norm.pdf(z, loc=np.mean(z), scale=np.std(z) if np.std(z) > 0 else 1.0)
    pred_norm = pick_top7(s_norm, counts, gaps)

    # 2) Chi-square
    x_chi = z - z.min() + 1e-6
    df_chi = max(2.0, float(np.mean(x_chi) * 2.0))
    s_chi2 = chi2.pdf(x_chi, df=df_chi)
    pred_chi2 = pick_top7(s_chi2, counts, gaps)

    # 3) F distribution
    x_f = (counts - counts.min()) + 1e-6
    dfn = 5.0
    dfd = max(10.0, float(len(draws) // 5))
    s_f = f.pdf(x_f / np.mean(x_f), dfn=dfn, dfd=dfd)
    pred_f = pick_top7(s_f, counts, gaps)

    # 4) Poisson — Pearson ostatak: |count−μ| / poisson.std(μ)=√μ; scipy.stats.poisson eksplicitno.
    mu = float(np.mean(counts))
    sig_p = float(poisson.std(mu))
    s_pois = np.abs(counts - mu) / (sig_p + 1e-12)
    pred_pois = pick_top7(s_pois, counts, gaps)

    # 5) Exponential — pdf(gap) je najveći za gap=0 (brojevi iz poslednjeg kola);
    #    cdf(gap) raste sa gap-om → ne kopira poslednju kombinaciju.
    x_exp = gaps - gaps.min()
    scale_exp = float(np.mean(x_exp)) if np.mean(x_exp) > 0 else 1.0
    s_exp = expon.cdf(x_exp, loc=0.0, scale=scale_exp)
    pred_exp = pick_top7(s_exp, counts, gaps)

    # 6) Weibull — samo cdf(gap) je isti red kao expon.cdf (obe strogo rastu po gap-u).
    #    cdf * (1 + k*z) meša čekanje i učestalost → drugačiji rang od expon; k mali, deterministički.
    x_w = np.maximum(gaps, 1e-6)
    c_w, loc_w, scale_w = weibull_min.fit(x_w, floc=0.0)
    s_weib = weibull_min.cdf(x_w, c=c_w, loc=loc_w, scale=scale_w) * (1.0 + 1e-3 * z)
    pred_weib = pick_top7(s_weib, counts, gaps)
    
    
    print()
    print("=== SLEDECE KOMBINACIJE (6 DISTRIBUCIJA) ===")
    print("CSV:", CSV_PATH)
    print("Broj kola:", len(draws))
    print()
    print("Next (norm):", pred_norm)
    print("Next (chi2):", pred_chi2)
    print("Next (f):", pred_f)
    print("Next (poisson):", pred_pois)
    print("Next (expon):", pred_exp)
    print("Next (weibull_min):", pred_weib)
    print()



if __name__ == "__main__":
    main()



"""
=== SLEDECE KOMBINACIJE (6 DISTRIBUCIJA) ===
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4594_k28.csv
Broj kola: 4594

Next (norm): [ 3  5 13 16 21 24 31]
Next (chi2): [13 16 21 24 25 31 38]
Next (f): [ 4  6 12 14 15 19 28]
Next (poisson): [ 1  8 17 20 23 27 30]
Next (expon): [ 5  7 12 15 21 27 34]
Next (weibull_min): [ 5  7 11 12 15 21 34]
"""
