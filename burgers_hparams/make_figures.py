"""Графики и сводная таблица по результатам run_experiments.py."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

INK, MUTED, GRID = "#14213D", "#5B6475", "#E4E2DA"
BLUE, ORANGE, BG = "#2F6DB5", "#D9661F", "#FFFFFF"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "legend.frameon": False,
})

R = {}
for p in glob.glob(os.path.join(RES, "*.json")):
    if p.endswith("timing.json"):
        continue
    r = json.load(open(p))
    R[r["name"]] = r


def arr(name):
    return np.load(os.path.join(RES, name + ".npz"))


d = np.load(os.path.join(HERE, "..", "burgers", "Burgers.npz"))
T, X, EXACT = d["t"].ravel(), d["x"].ravel(), d["usol"].T


def runs(n):
    """Все прогоны варианта по разным seed."""
    keys = ["base_s1", "base_s2", "base_s3"] if n == "base_s1" else [n, n + "_s2", n + "_s3"]
    return [R[k] for k in keys if k in R]


def l2(n):
    """Среднее геометрическое L2 по seed (ошибки различаются в разы, поэтому в логарифме)."""
    return float(np.exp(np.mean([np.log(r["l2"]) for r in runs(n)])))


def l2rng(n):
    v = [r["l2"] for r in runs(n)]
    return min(v), max(v)


TIMING_PATH = os.path.join(RES, "timing.json")
TIMING = json.load(open(TIMING_PATH)) if os.path.exists(TIMING_PATH) else {}


# Итерация L-BFGS в замере шумит (48–108 мс у одной и той же архитектуры: число вычислений
# на шаг разное), итерация Adam стабильна. Поэтому L-BFGS считаем как K итераций Adam,
# K — медиана отношения по всем замерам.
K_LBFGS = float(np.median([t["t_lbfgs_ms"] / t["t_adam_ms"] for t in TIMING.values()])) if TIMING else 6.0


def est_time(r, key):
    """Время прогона, с: мс/итерацию Adam × (шаги Adam + K × шаги L-BFGS)."""
    t = TIMING[key]
    adam = r["adam_iters"] + 1000 * r["rar_rounds"]
    if not r["lbfgs"]:
        adam, lbfgs = r["steps_total"], 0
    else:
        lbfgs = r.get("lbfgs_steps", max(r["steps_total"] - adam, 0))
    # у RMSprop/SGD/AdamW шаг оптимизатора другой, но сеть та же, что у базовой:
    # стоимость итерации L-BFGS берём от базовой архитектуры
    t_eval = TIMING["base_s1"]["t_adam_ms"] if r["optimizer"] != "adam" else t["t_adam_ms"]
    nncg = r.get("nncg_steps", 0) * t.get("t_nncg_ms", 0)
    return (t["t_adam_ms"] * adam + t_eval * K_LBFGS * lbfgs + nncg) / 1e3


def tmin(n):
    if n in TIMING:
        return float(np.mean([est_time(r, n) for r in runs(n)])) / 60
    return float(np.mean([r["time_total"] for r in runs(n)])) / 60


base = R["base_s1"]
seeds = [r["l2"] for r in runs("base_s1")]
lo, hi = min(seeds), max(seeds)


def pct(v):
    x = v * 100
    return f"{x:.0f}%" if x >= 10 else f"{x:.2g}%"


def pct_axis(ax):
    """Ось L2 в процентах с понятными делениями 0.1% / 0.2% / 0.5% / 1% / …"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [m * 10.0 ** e for e in range(-4, 1) for m in (1, 2, 5) if m * 10.0 ** e <= 1]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def seed_band(ax):
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.12, lw=0, label="разброс базовой по 3 seed")
    ax.axhline(l2("base_s1"), color=ORANGE, lw=1.5, ls="--", label="базовая (среднее)")


def errbars(ax, xs, names, color):
    for x, n in zip(xs, names):
        if len(runs(n)) > 1:
            a, b = l2rng(n)
            ax.plot([x, x], [a, b], color=color, lw=2, alpha=0.6)


def bars(names, labels, fname, title, xlabel=None, figsize=(12, 5.6)):
    names = [n for n in names if n in R]
    vals = [l2(n) for n in names]
    colors = [ORANGE if n == "base_s1" else BLUE for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    b = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    errbars(ax, range(len(vals)), names, INK)
    ax.set_yscale("log")
    pct_axis(ax)
    ax.set_xticks(range(len(vals)), [labels[n] + ("*" if len(runs(n)) > 1 else "") for n in names])
    ax.set_ylabel("относительная L2-ошибка")
    for rect, v, n in zip(b, vals, names):
        top = l2rng(n)[1] if len(runs(n)) > 1 else v
        ax.text(rect.get_x() + rect.get_width() / 2, top * 1.12, pct(v),
                ha="center", va="bottom", fontsize=14, color=INK)
    ax.set_ylim(min(min(l2rng(n)[0] for n in names), min(vals)) / 2,
                max(max(l2rng(n)[1] for n in names), max(vals)) * 3)
    ax.grid(axis="x", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK)
    save(fig, fname)


def sweep(names, xs, fname, xlabel, logx=False, cost="iter", categorical=False):
    names_xs = [(n, x) for n, x in zip(names, xs) if n in R]
    labels = [str(x) for _, x in names_xs]
    xs_ = list(range(len(names_xs))) if categorical else [x for _, x in names_xs]
    vals = [l2(n) for n, _ in names_xs]
    if cost == "iter":
        cy = [TIMING[n]["t_adam_ms"] for n, _ in names_xs]
        clabel, ctitle = "мс на итерацию", "Стоимость одной итерации"
    else:
        cy = [tmin(n) for n, _ in names_xs]
        clabel, ctitle = "время обучения, мин", "Полное время обучения"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    seed_band(a1)
    a1.plot(xs_, vals, "o-", color=BLUE, lw=2.5, ms=9)
    errbars(a1, xs_, [n for n, _ in names_xs], BLUE)
    for (n, _), x in zip(names_xs, xs_):
        if n == "base_s1":
            a1.plot([x], [l2(n)], "o", color=ORANGE, ms=13, zorder=5)
    a1.set_yscale("log"); pct_axis(a1); a1.set_xlabel(xlabel); a1.set_ylabel("относительная L2-ошибка")
    a1.set_title("Точность (меньше — лучше)", loc="left")
    a1.legend(loc="best", fontsize=13)
    a2.plot(xs_, cy, "s-", color=MUTED, lw=2.5, ms=8)
    for (n, _), x, c in zip(names_xs, xs_, cy):
        if n == "base_s1":
            a2.plot([x], [c], "s", color=ORANGE, ms=12, zorder=5)
    a2.set_xlabel(xlabel); a2.set_ylabel(clabel)
    a2.set_title(ctitle, loc="left")
    a2.set_ylim(0, max(cy) * 1.15)
    for a in (a1, a2):
        if logx and not categorical:
            a.set_xscale("log")
        a.set_xticks(xs_, labels)
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.tight_layout()
    save(fig, fname)


def heat(ax, Z, title, cmap="RdBu_r", vmin=None, vmax=None):
    im = ax.pcolormesh(T, X, Z.T, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title, loc="left"); ax.grid(False)
    return im


# 1. Эталон
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
im = heat(axs[0], EXACT, "Эталонное решение u(x, t)", vmin=-1, vmax=1)
fig.colorbar(im, ax=axs[0])
for tv, c in zip([0, 0.25, 0.5, 0.75], ["#9FB6D6", "#6A90C4", BLUE, INK]):
    i = np.argmin(np.abs(T - tv))
    axs[1].plot(X, EXACT[i], color=c, lw=2.5, label=f"t = {T[i]:.2f}")
axs[1].set_xlabel("x"); axs[1].set_ylabel("u"); axs[1].legend(fontsize=13)
axs[1].set_title("Срезы: волна «опрокидывается»", loc="left")
fig.tight_layout(); save(fig, "reference.png")

# 2. Базовый результат
U = arr("base_s1")["u_pred"]
err = np.abs(U - EXACT)
fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
fig.colorbar(heat(axs[0], EXACT, "Эталон", vmin=-1, vmax=1), ax=axs[0])
fig.colorbar(heat(axs[1], U, "PINN (базовая модель)", vmin=-1, vmax=1), ax=axs[1])
fig.colorbar(heat(axs[2], err, "|ошибка|", cmap="magma_r"), ax=axs[2])
fig.tight_layout(); save(fig, "base_result.png")

# 3. Глубина / ширина
sweep(["depth_1", "depth_2", "base_s1", "depth_4", "depth_6", "depth_8"], [1, 2, 3, 4, 6, 8],
      "depth.png", "число скрытых слоёв (по 20 нейронов)")
sweep(["width_5", "width_10", "base_s1", "width_40", "width_80"], [5, 10, 20, 40, 80],
      "width.png", "нейронов в слое (3 слоя)", logx=True)

# 4. Активации и инициализация
bars(["base_s1", "act_sin", "act_silu", "act_gelu", "act_sigmoid", "act_elu", "act_relu"],
     {"base_s1": "tanh", "act_sin": "sin", "act_silu": "SiLU", "act_gelu": "GELU",
      "act_sigmoid": "sigmoid", "act_elu": "ELU", "act_relu": "ReLU"},
     "activation.png", "Функция активации")
bars(["base_s1", "init_Glorot_uniform", "init_He_normal", "init_He_uniform", "init_zeros"],
     {"base_s1": "Glorot normal", "init_Glorot_uniform": "Glorot uniform",
      "init_He_normal": "He normal", "init_He_uniform": "He uniform", "init_zeros": "нули"},
     "init.png", "Инициализация весов", figsize=(11, 5.4))

# 5. Число точек
sweep(["points_250", "points_500", "points_1000", "base_s1", "points_5000", "points_10000"],
      [250, 500, 1000, 2540, 5000, 10000], "points.png", "точек коллокации внутри области",
      logx=True)

# 6. Распределения точек: как выглядят + точность
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, n, title in zip(axs, ["dist_pseudo", "base_s1", "dist_uniform"],
                        ["Случайные (pseudo)", "Hammersley (базовая)", "Равномерная сетка"]):
    if n in R:
        p = arr(n)["train_x"]
        ax.scatter(p[:, 1], p[:, 0], s=3, color=ORANGE if n == "base_s1" else BLUE)
        ax.set_title(f"{title}: L2 = {pct(R[n]['l2'])}", loc="left")
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.grid(False)
fig.tight_layout(); save(fig, "dist_points.png")
bars(["base_s1", "dist_Halton", "dist_Sobol", "dist_LHS", "dist_uniform", "dist_pseudo", "rar"],
     {"base_s1": "Hammersley", "dist_Halton": "Halton", "dist_Sobol": "Sobol", "dist_LHS": "LHS",
      "dist_uniform": "сетка", "dist_pseudo": "случайные", "rar": "Hammersley\n+ RAR"},
     "dist.png", "Способ расстановки точек", figsize=(14, 5.6))

# 7. RAR: куда добавлены точки
if "rar" in R:
    pb = arr("base_s1")["train_x"]
    pr = arr("rar")["train_x"]
    known = {tuple(np.round(v, 6)) for v in pb}
    added = np.array([v for v in pr if tuple(np.round(v, 6)) not in known])
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    im = heat(axs[0], err, "|ошибка| базовой модели", cmap="magma_r")
    fig.colorbar(im, ax=axs[0])
    if len(added):
        axs[0].scatter(added[:, 1], added[:, 0], s=14, color="#2EC4B6", edgecolor=INK, lw=0.3,
                       label=f"{len(added)} точек, добавленных RAR")
        axs[0].legend(loc="lower left", fontsize=13)
    Ur = arr("rar")["u_pred"]
    fig.colorbar(heat(axs[1], np.abs(Ur - EXACT), "|ошибка| после RAR", cmap="magma_r",
                      vmax=err.max()), ax=axs[1])
    for a in axs:
        a.set_ylim(-0.25, 0.25)
        a.set_ylabel("x (приближено к разрыву)")
    fig.tight_layout(); save(fig, "rar.png")

# 8. Оптимизаторы: кривые потерь
curves = [("base_s1", "Adam → L-BFGS (базовая)", ORANGE, "-"),
          ("opt_adam_only", "только Adam 15k", BLUE, "-"),
          ("opt_adam_only_50k", "только Adam 50k", BLUE, "--"),
          ("opt_lbfgs_only", "только L-BFGS", INK, "-"),
          ("opt_rmsprop", "RMSprop → L-BFGS", "#7A9CC6", "-"),
          ("opt_sgd", "SGD → L-BFGS", MUTED, ":")]
fig, ax = plt.subplots(figsize=(14, 5.8))
for n, lab, c, ls in curves:
    if n in R:
        a = arr(n)
        ax.semilogy(a["steps"], a["losses"].sum(1), ls, color=c, lw=2.4,
                    label=f"{lab}: L2 = {pct(R[n]['l2'])}")
ax.axvline(15000, color=MUTED, lw=1, ls="--")
ax.text(15400, ax.get_ylim()[1] * 0.4, "переход на L-BFGS", color=MUTED, fontsize=13)
ax.set_xlabel("итерация"); ax.set_ylabel("суммарный loss")
ax.legend(fontsize=13, loc="upper right")
ax.set_title("Оптимизатор: кривые обучения (seed 1)", loc="left")
save(fig, "optimizers.png")

# 9. Скорость обучения
fig, ax = plt.subplots(figsize=(13, 5.6))
for n, lab, c in [("lr_0.01", "lr = 1e-2", MUTED), ("base_s1", "lr = 1e-3 (базовая)", ORANGE),
                  ("lr_0.0001", "lr = 1e-4", BLUE)]:
    if n in R:
        a = arr(n)
        ax.semilogy(a["steps"], a["losses"].sum(1), color=c, lw=2.2,
                    label=f"{lab}: L2 = {pct(R[n]['l2'])}")
ax.axvline(15000, color=MUTED, lw=1, ls="--")
ax.set_xlabel("итерация"); ax.set_ylabel("суммарный loss"); ax.legend(fontsize=13)
ax.set_title("Шаг обучения Adam, потом L-BFGS (seed 1)", loc="left")
save(fig, "lr.png")

# 10. Сколько Adam перед L-BFGS
sweep(["opt_lbfgs_only", "adam_2000", "adam_5000", "base_s1", "adam_30000"],
      [0, 2000, 5000, 15000, 30000], "adam_iters.png", "итераций Adam перед L-BFGS", cost="total", categorical=True)

# 11. Компромисс точность / время
fig, ax = plt.subplots(figsize=(14, 6.4))
MAIN = [n for n in R if not n.endswith(("_s2", "_s3")) and not n.startswith("seed_f")]
for n in MAIN:
    c = ORANGE if n == "base_s1" else BLUE
    ax.scatter(tmin(n), l2(n), s=160 if n == "base_s1" else 60, color=c,
               zorder=5 if n == "base_s1" else 3, alpha=0.9)
lab = {"base_s1": ("базовая", (10, -18)), "points_10000": ("10 000 точек", (-40, 12)),
       "points_5000": ("5000 точек", (8, -16)), "cfg_tanh_gu_6x20_30k": ("tanh GU 6×20, Adam 30k", (-60, 12)),
       "cfg_tanh_gu_4x20": ("tanh GU 4×20", (8, -16)), "dist_Sobol": ("Sobol", (-52, -4)),
       "opt_adam_only": ("только Adam", (8, 4)), "act_relu": ("ReLU", (8, 0)), "depth_1": ("1 слой", (8, -12)),
       "act_sin": ("sin", (8, 0)), "width_80": ("80 нейронов", (8, 4))}
for n, (t, off) in lab.items():
    if n in R:
        ax.annotate(t, (tmin(n), l2(n)), xytext=off, textcoords="offset points", fontsize=13, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_yscale("log"); pct_axis(ax); ax.set_xlabel("время обучения, мин"); ax.set_ylabel("относительная L2-ошибка")
ax.set_title("Все эксперименты: точность против времени (левый нижний угол — лучше)", loc="left")
save(fig, "tradeoff.png")

# 12. Срезы плохих и хорошей конфигураций
fig, ax = plt.subplots(figsize=(13, 5.4))
i = np.argmin(np.abs(T - 0.75))
ax.plot(X, EXACT[i], color=INK, lw=4, label="эталон")
for n, labl, c, ls in [("base_s1", "базовая", ORANGE, "--"), ("depth_1", "1 слой", BLUE, "-"),
                       ("act_relu", "ReLU", MUTED, "-."), ("points_250", "250 точек", "#7A9CC6", ":")]:
    if n in R:
        ax.plot(X, arr(n)["u_pred"][i], ls, color=c, lw=2.6, label=f"{labl} ({pct(R[n]['l2'])})")
ax.set_xlim(-0.6, 0.6); ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title(f"Срез при t = {T[i]:.2f} (seed 1): у разрыва видно, кто справился", loc="left")
ax.legend(fontsize=13)
save(fig, "slices.png")

# ===================== вторая серия: подробные сравнения =====================
from matplotlib.colors import LogNorm


def err_maps(items, fname, title):
    """Карты |ошибки| по всей области для нескольких вариантов (seed 1), общая лог-шкала."""
    items = [(n, l) for n, l in items if n in R]
    fig, axs = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.4), sharey=True)
    norm = LogNorm(vmin=1e-4, vmax=1)
    for ax, (n, lab) in zip(axs, items):
        e = np.abs(arr(n)["u_pred"] - EXACT)
        im = ax.pcolormesh(T, X, np.clip(e, 1e-4, 1).T, cmap="magma_r", norm=norm, shading="auto",
                           rasterized=True)
        ax.set_title(f"{lab}: {pct(R[n]['l2'])}", loc="left")
        ax.set_xlabel("t"); ax.grid(False)
    axs[0].set_ylabel("x")
    cb = fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01)
    cb.set_label("|ошибка|")
    save(fig, fname)


def curves(items, fname, title, vline=15000):
    """Кривые суммарного loss (seed 1)."""
    fig, ax = plt.subplots(figsize=(14, 5.6))
    pal = iter([BLUE, INK, "#7A9CC6", MUTED, "#2EC4B6", "#8E6CC0", "#A3B18A"])
    for n, lab in [(n, l) for n, l in items if n in R]:
        c = ORANGE if n == "base_s1" else next(pal)
        a = arr(n)
        ax.semilogy(a["steps"], a["losses"].sum(1), color=c, lw=2.4 if n == "base_s1" else 2,
                    label=f"{lab}: L2 = {pct(R[n]['l2'])}")
    if vline:
        ax.axvline(vline, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("итерация"); ax.set_ylabel("суммарный loss"); ax.legend(fontsize=13, loc="upper right")
    ax.set_title(title, loc="left")
    save(fig, fname)


err_maps([("depth_1", "1 слой"), ("depth_2", "2 слоя"), ("base_s1", "3 слоя"), ("depth_6", "6 слоёв")],
         "maps_depth.png", "Ошибка по всей области при разной глубине (seed 1)")
err_maps([("width_5", "5 нейронов"), ("width_10", "10"), ("base_s1", "20"), ("width_80", "80")],
         "maps_width.png", "Ошибка по всей области при разной ширине слоя (seed 1)")
err_maps([("base_s1", "tanh"), ("act_sigmoid", "sigmoid"), ("act_sin", "sin"), ("act_relu", "ReLU")],
         "maps_act.png", "Ошибка по всей области при разных активациях (seed 1)")
err_maps([("points_250", "250 точек"), ("points_1000", "1000"), ("base_s1", "2540"), ("points_5000", "5000")],
         "maps_points.png", "Ошибка по всей области при разном числе точек (seed 1)")
err_maps([("opt_adam_only", "только Adam"), ("opt_lbfgs_only", "только L-BFGS"),
          ("base_s1", "Adam → L-BFGS"), ("opt_rmsprop", "RMSprop → L-BFGS")],
         "maps_optim.png", "Ошибка по всей области при разных оптимизаторах (seed 1)")

curves([("depth_1", "1 слой"), ("depth_2", "2 слоя"), ("base_s1", "3 слоя (базовая)"),
        ("depth_4", "4 слоя"), ("depth_6", "6 слоёв"), ("depth_8", "8 слоёв")],
       "curves_depth.png", "Кривые обучения при разной глубине (seed 1)")
curves([("width_5", "5 нейронов"), ("width_10", "10"), ("base_s1", "20 (базовая)"),
        ("width_40", "40"), ("width_80", "80")],
       "curves_width.png", "Кривые обучения при разной ширине (seed 1)")
curves([("base_s1", "tanh (базовая)"), ("act_sigmoid", "sigmoid"), ("act_silu", "SiLU"),
        ("act_gelu", "GELU"), ("act_sin", "sin"), ("act_elu", "ELU"), ("act_relu", "ReLU")],
       "curves_act.png", "Кривые обучения при разных активациях (seed 1)")
curves([("points_250", "250 точек"), ("points_500", "500"), ("points_1000", "1000"),
        ("base_s1", "2540 (базовая)"), ("points_5000", "5000"), ("points_10000", "10 000")],
       "curves_points.png", "Кривые обучения при разном числе точек (seed 1)")

# Сетка слои × нейроны
GRID = {(2, 10): "grid_2x10", (2, 20): "depth_2", (2, 40): "grid_2x40",
        (3, 10): "width_10", (3, 20): "base_s1", (3, 40): "width_40",
        (4, 10): "grid_4x10", (4, 20): "depth_4", (4, 40): "grid_4x40",
        (6, 10): "grid_6x10", (6, 20): "depth_6", (6, 40): "grid_6x40"}
if all(v in R for v in GRID.values()):
    Ls, Ws = [2, 3, 4, 6], [10, 20, 40]
    M = np.array([[l2(GRID[(L, W)]) for W in Ws] for L in Ls])
    C = np.array([[TIMING.get(GRID[(L, W)], {}).get("t_adam_ms", np.nan) for W in Ws] for L in Ls])
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, Z, ttl, cmap, fmt, norm in [
            (axs[0], M, "L2-ошибка (среднее по 3 seed)", "magma_r", pct, LogNorm(vmin=M.min() / 1.5, vmax=M.max())),
            (axs[1], C, "Стоимость итерации, мс", "Blues", lambda v: f"{v:.1f}", None)]:
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(Ls)):
            for j in range(len(Ws)):
                v = Z[i, j]
                dark = (np.log(v) - np.log(Z.min())) / (np.log(Z.max()) - np.log(Z.min()) + 1e-9) > 0.55 if norm else v > np.nanmean(Z)
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=17,
                        color="white" if dark else INK, fontweight="bold" if (Ls[i], Ws[j]) == (3, 20) else "normal")
        ax.set_xticks(range(len(Ws)), [str(w) for w in Ws]); ax.set_yticks(range(len(Ls)), [str(l) for l in Ls])
        ax.set_xlabel("нейронов в слое"); ax.set_ylabel("скрытых слоёв"); ax.set_title(ttl, loc="left"); ax.grid(False)
        ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=ORANGE, lw=4))
    fig.tight_layout(); save(fig, "grid.png")

# Итерации: только Adam против Adam → L-BFGS
ADAM_ONLY = [(5000, "adam_only_5k"), (15000, "opt_adam_only"), (30000, "adam_only_30k"),
             (50000, "opt_adam_only_50k"), (100000, "adam_only_100k")]
ADAM_LB = [(0, "opt_lbfgs_only"), (2000, "adam_2000"), (5000, "adam_5000"), (15000, "base_s1"), (30000, "adam_30000")]
if all(n in R for _, n in ADAM_ONLY):
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ao = [(it, n) for it, n in ADAM_ONLY if n in R]
    al = [(max(it, 500), n) for it, n in ADAM_LB if n in R]
    ax.plot([i for i, _ in ao], [l2(n) for _, n in ao], "o-", color=BLUE, lw=2.5, ms=9, label="только Adam")
    errbars(ax, [i for i, _ in ao], [n for _, n in ao], BLUE)
    ax.plot([i for i, _ in al], [l2(n) for _, n in al], "s-", color=ORANGE, lw=2.5, ms=9,
            label="Adam, затем L-BFGS до сходимости")
    errbars(ax, [i for i, _ in al], [n for _, n in al], ORANGE)
    ax.set_xscale("log"); ax.set_yscale("log"); pct_axis(ax)
    ax.set_xticks([500, 2000, 5000, 15000, 30000, 50000, 100000],
                  ["0", "2k", "5k", "15k", "30k", "50k", "100k"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("итераций Adam"); ax.set_ylabel("относительная L2-ошибка")
    ax.set_title("Сколько итераций нужно: Adam сам по себе против Adam + L-BFGS", loc="left")
    ax.legend(fontsize=14)
    save(fig, "iterations.png")

# Ограничение шагов L-BFGS
CAP = [("0 (только Adam)", "opt_adam_only"), (500, "lbfgs_max_500"), (1000, "lbfgs_max_1000"),
       (2000, "lbfgs_max_2000"), ("до сходимости", "base_s1")]
if all(n in R for _, n in CAP):
    bars([n for _, n in CAP], {n: (f"{c}" if c != "до сходимости" else "до\nсходимости") for c, n in CAP},
         "lbfgs_cap.png", "Сколько шагов L-BFGS после 15 000 итераций Adam", xlabel="шагов L-BFGS",
         figsize=(12, 5.4))

# Оптимизаторы — расширенно
OPT = {"base_s1": "Adam\n→ L-BFGS", "opt_rmsprop": "RMSprop\n→ L-BFGS", "opt_adamw": "AdamW\n→ L-BFGS",
       "opt_adam_decay": "Adam+спад\n→ L-BFGS", "opt_nncg": "+ NNCG\nпосле L-BFGS",
       "opt_sgd": "SGD\n→ L-BFGS", "opt_adam_only_50k": "только Adam\n50k",
       "opt_adam_only": "только Adam\n15k", "opt_lbfgs_only": "только\nL-BFGS"}
bars(list(OPT), OPT, "optim_all.png", "Все проверенные оптимизаторы", figsize=(19, 6))

# Точки на границе и в начальный момент
BCIC = {"bc_20": "граница 20", "base_s1": "граница 80,\nt=0 160", "bc_320": "граница 320",
        "ic_40": "t=0: 40", "ic_640": "t=0: 640"}
bars(list(BCIC), BCIC, "bcic.png", "Точки на границе и в начальный момент (внутри всегда 2540)",
     figsize=(13, 5.6))

# Пересэмплирование
RS = {"dist_pseudo": "случайные,\nфиксированные", "resample_pseudo": "случайные,\nновые каждые 1000 ит.",
      "base_s1": "Hammersley", "rar": "Hammersley\n+ RAR"}
bars(list(RS), RS, "resample.png", "Фиксированные точки, пересэмплирование и RAR", figsize=(14, 5.6))

# Активации с LAAF
ACT = {"base_s1": "tanh", "act_laaf5_tanh": "LAAF-5 tanh", "act_sigmoid": "sigmoid", "act_silu": "SiLU",
       "act_gelu": "GELU", "act_sin": "sin", "act_elu": "ELU", "act_relu": "ReLU"}
bars(list(ACT), ACT, "activation_all.png", "Функция активации, включая адаптивную LAAF", figsize=(15, 5.6))

# Чувствительность: во сколько раз худший вариант группы хуже лучшего
SENS = [("Активация", ["base_s1", "act_sigmoid", "act_silu", "act_gelu", "act_sin", "act_elu", "act_relu"]),
        ("Слои", ["depth_1", "depth_2", "base_s1", "depth_4", "depth_6", "depth_8"]),
        ("Нейроны", ["width_5", "width_10", "base_s1", "width_40", "width_80"]),
        ("Число точек", ["points_250", "points_500", "points_1000", "base_s1", "points_5000", "points_10000"]),
        ("Расстановка", ["base_s1", "dist_pseudo", "dist_uniform", "dist_LHS", "dist_Halton", "dist_Sobol"]),
        ("Оптимизатор", ["base_s1", "opt_rmsprop", "opt_sgd", "opt_adam_only", "opt_lbfgs_only"]),
        ("Итерации Adam", ["opt_lbfgs_only", "adam_2000", "adam_5000", "base_s1", "adam_30000"]),
        ("Инициализация*", ["base_s1", "init_Glorot_uniform", "init_He_normal", "init_He_uniform"]),
        ("Шаг lr", ["base_s1", "lr_0.01", "lr_0.0001"]),
        ("Точки границы/IC", ["base_s1", "bc_20", "bc_320", "ic_40", "ic_640"])]
rows = []
for g, names in SENS:
    vals = [l2(n) for n in names if n in R]
    if len(vals) > 1:
        rows.append((g, max(vals) / min(vals)))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[BLUE if r[1] > 10 else MUTED for r in rows], height=0.6)
for i, (_, v) in enumerate(rows):
    ax.text(v * 1.08, i, f"×{v:.0f}" if v >= 10 else f"×{v:.1f}", va="center", fontsize=15, color=INK)
ax.set_xscale("log"); ax.set_xlim(1, max(r[1] for r in rows) * 3)
_f32 = [R[n] for n in ["base_s1", "base_s2", "base_s3"] + [f"seed_f32_{k}" for k in range(4, 11)] if n in R]
_ok = [r["l2"] for r in _f32 if not (r["lbfgs"] and r.get("lbfgs_steps", 999) < 200)]
SEED_SPREAD = max(_ok) / min(_ok)
ax.axvline(SEED_SPREAD, color=ORANGE, ls="--", lw=1.5)
ax.text(SEED_SPREAD * 1.05, -0.6, f"разброс от seed без срывов L-BFGS (×{SEED_SPREAD:.0f})", color=ORANGE, fontsize=13)
ax.set_xlabel("во сколько раз худший вариант хуже лучшего (по L2)")
ax.set_title("Какой гиперпараметр влияет сильнее всего", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "sensitivity.png")
print("sensitivity:", [(g, round(v, 1)) for g, v in rows])


# ===================== третья серия: комбинации =====================
def est_parts(n):
    """Средние по seed секунды Adam и L-BFGS(+NNCG) по честному замеру."""
    if n not in TIMING:
        return float("nan"), float("nan")
    t = TIMING[n]
    a_s, l_s = [], []
    for r in runs(n):
        adam = r["adam_iters"] + 1000 * r["rar_rounds"]
        if not r["lbfgs"]:
            adam, lb = r["steps_total"], 0
        else:
            lb = r.get("lbfgs_steps", max(r["steps_total"] - adam, 0))
        t_eval = TIMING["base_s1"]["t_adam_ms"] if r["optimizer"] != "adam" else t["t_adam_ms"]
        a_s.append(t["t_adam_ms"] * adam / 1e3)
        l_s.append((t_eval * K_LBFGS * lb + r.get("nncg_steps", 0) * t.get("t_nncg_ms", 0)) / 1e3)
    return float(np.mean(a_s)), float(np.mean(l_s))


def matrix(row_keys, col_keys, name_of, fname, title, row_lab, col_lab, xlabel, ylabel):
    M = np.full((len(row_keys), len(col_keys)), np.nan)
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            n = name_of(rk, ck)
            if n in R:
                M[i, j] = l2(n)
    fig, ax = plt.subplots(figsize=(3.0 * len(col_keys) + 3, 1.25 * len(row_keys) + 1.6))
    lm = np.log10(M)
    ax.imshow(lm, cmap="magma_r", vmin=np.log10(0.002), vmax=np.log10(1.0), aspect="auto")
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            v = M[i, j]
            if np.isnan(v):
                continue
            mark = " ✓" if v < 0.01 else ""
            ax.text(j, i, pct(v) + mark, ha="center", va="center", fontsize=17,
                    color="white" if lm[i, j] > np.log10(0.05) else INK,
                    fontweight="bold" if (i, j) == (0, 0) else "normal")
    ax.set_xticks(range(len(col_keys)), col_lab); ax.set_yticks(range(len(row_keys)), row_lab)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left"); ax.grid(False)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=ORANGE, lw=4))
    save(fig, fname)
    return M


ACTS5 = ["tanh", "sin", "sigmoid", "silu", "gelu"]
ACT_LAB = ["tanh", "sin", "sigmoid", "SiLU", "GELU"]
INIT_KEYS = ["GN", "GU", "HN", "HU"]


def ai_name(a, k):
    if a == "tanh":
        return {"GN": "base_s1", "GU": "init_Glorot_uniform", "HN": "init_He_normal", "HU": "init_He_uniform"}[k]
    return f"act_{a}" if k == "GN" else f"ai_{a}_{k}"


MAI = matrix(ACTS5, INIT_KEYS, ai_name, "act_init.png",
             "L2-ошибка: активация × инициализация (среднее по 3 seed, ✓ — меньше 1 %)", ACT_LAB,
             ["Glorot normal", "Glorot uniform", "He normal", "He uniform"], "инициализация весов", "активация")

OPT_KEYS = ["adamlbfgs", "rmsprop", "adamw", "adamonly"]


def ao_name(a, k):
    if a == "tanh":
        return {"adamlbfgs": "base_s1", "rmsprop": "opt_rmsprop", "adamw": "opt_adamw", "adamonly": "opt_adam_only"}[k]
    return f"act_{a}" if k == "adamlbfgs" else f"ao_{a}_{k}"


MAO = matrix(ACTS5, OPT_KEYS, ao_name, "act_opt.png",
             "L2-ошибка: активация × оптимизатор (среднее по 3 seed, ✓ — меньше 1 %)", ACT_LAB,
             ["Adam → L-BFGS", "RMSprop → L-BFGS", "AdamW → L-BFGS", "только Adam"], "оптимизатор", "активация")

COMBOS = ["base_s1", "cfg_tanh_gu_4x20", "cfg_tanh_gu_6x20_30k", "cfg_tanh_gu_4x20_5000pts", "cfg_tanh_2x20_fast",
          "cfg_sin_gu_4x20", "cfg_sin_gu_4x40", "cfg_sin_gu_rmsprop_4x20", "cfg_sigmoid_gu_4x20",
          "cfg_silu_hn_rmsprop_4x20", "cfg_gelu_hu_adamw", "cfg_laaf_gu_4x20", "cfg_relu_hn", "cfg_elu_hn"]
COMBO_LAB = {"base_s1": "базовая: tanh, GN, 3×20", "cfg_tanh_gu_4x20": "tanh, GU, 4×20",
             "cfg_tanh_gu_6x20_30k": "tanh, GU, 6×20, Adam 30k", "cfg_tanh_gu_4x20_5000pts": "tanh, GU, 4×20, 5000 точек",
             "cfg_tanh_2x20_fast": "tanh, 2×20, Adam 5k", "cfg_sin_gu_4x20": "sin, GU, 4×20",
             "cfg_sin_gu_4x40": "sin, GU, 4×40", "cfg_sin_gu_rmsprop_4x20": "sin, GU, 4×20, RMSprop",
             "cfg_sigmoid_gu_4x20": "sigmoid, GU, 4×20", "cfg_silu_hn_rmsprop_4x20": "SiLU, HN, 4×20, RMSprop",
             "cfg_gelu_hu_adamw": "GELU, HU, 3×20, AdamW", "cfg_laaf_gu_4x20": "LAAF-tanh, GU, 4×20",
             "cfg_relu_hn": "ReLU, HN, 3×20", "cfg_elu_hn": "ELU, HN, 3×20"}
cs = sorted([n for n in COMBOS if n in R], key=l2)
if cs:
    fig, ax = plt.subplots(figsize=(14, 0.62 * len(cs) + 1.4))
    ys = range(len(cs))[::-1]
    ax.barh(list(ys), [l2(n) for n in cs], color=[ORANGE if n == "base_s1" else (BLUE if l2(n) < 0.01 else MUTED) for n in cs], height=0.62)
    for y, n in zip(ys, cs):
        a, b = l2rng(n)
        ax.plot([a, b], [y, y], color=INK, lw=2, alpha=0.6)
        ax.text(b * 1.15, y, pct(l2(n)) + (" ✓" if l2(n) < 0.01 else ""), va="center", fontsize=14, color=INK)
    ax.set_yticks(list(ys), [COMBO_LAB[n] for n in cs])
    ax.set_xscale("log"); ax.axvline(0.01, color=INK, ls=":", lw=1.2)
    ax.text(0.0105, len(cs) - 0.4, "1 %", color=INK, fontsize=13)
    ax.set_xlabel("относительная L2-ошибка (среднее по 3 seed, черта — разброс)")
    ax.set_title("Готовые конфигурации: меняется сразу несколько параметров", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "combos.png")

# Таблицы в формате образца → tables.json
TABLES = {}
for n in sorted(set(MAIN)):
    a_s, l_s = est_parts(n)
    v = l2(n); lo_, hi_ = l2rng(n)
    TABLES[n] = dict(group=R[n]["group"], seeds=len(runs(n)), l2=v, l2_min=lo_, l2_max=hi_,
                     acc=100 * (1 - v), adam_s=a_s, lbfgs_s=l_s, total_min=(a_s + l_s) / 60,
                     ok=v < 0.01, layers=R[n]["layers"], width=R[n]["width"], act=R[n]["activation"],
                     init=R[n]["init"], opt=R[n]["optimizer"], lbfgs=R[n]["lbfgs"], adam_iters=R[n]["adam_iters"],
                     points=R[n]["num_domain"], params=R[n]["n_params"])
json.dump(TABLES, open(os.path.join(HERE, "tables.json"), "w"), ensure_ascii=False, indent=1)


# ===================== стабильность: float32 против float64 =====================
F32 = ["base_s1", "base_s2", "base_s3"] + [f"seed_f32_{k}" for k in range(4, 11)]
F64 = [f"seed_f64_{k}" for k in range(1, 11)]


def stalled(r):
    return r["lbfgs"] and r.get("lbfgs_steps", 999) < 200


STAB = {}
if all(n in R for n in F32 + F64):
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for x, (names, lab, c) in enumerate([(F32, "float32 (по умолчанию)", BLUE), (F64, "float64", ORANGE)]):
        vals = [R[n]["l2"] for n in names]
        st = [stalled(R[n]) for n in names]
        jit = np.linspace(-0.18, 0.18, len(vals))
        ax.scatter(x + jit, vals, s=110, color=[MUTED if s else c for s in st], edgecolor=INK, lw=0.6, zorder=3)
        gm = float(np.exp(np.mean(np.log(vals))))
        ax.plot([x - 0.3, x + 0.3], [gm, gm], color=c, lw=3)
        ax.text(x + 0.33, gm, f"среднее {pct(gm)}", va="center", fontsize=14, color=INK)
        STAB[lab] = dict(gm=gm, med=float(np.median(vals)), lo=min(vals), hi=max(vals), stalls=int(sum(st)),
                         ok=int(sum(v < 0.01 for v in vals)), n=len(vals),
                         t_min=float(np.mean([R[n]["time_total"] for n in names])) / 60)
    ax.scatter([], [], s=110, color=MUTED, edgecolor=INK, label="L-BFGS остановился за < 200 шагов")
    ax.set_yscale("log"); pct_axis(ax); ax.axhline(0.01, color=INK, ls=":", lw=1.2)
    ax.set_xticks([0, 1], ["float32 (по умолчанию)", "float64"]); ax.set_xlim(-0.6, 1.9)
    ax.set_ylabel("относительная L2-ошибка"); ax.legend(fontsize=13, loc="upper right")
    ax.set_title("Базовая модель на 10 seed: точность вычислений и срывы L-BFGS", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "stability.png")
    json.dump(STAB, open(os.path.join(HERE, "stability.json"), "w"), ensure_ascii=False, indent=1)
    print("stability:", STAB)


# сводка: по варианту — среднее по seed и разброс
rows = sorted((R[n] for n in MAIN), key=lambda r: (r["group"], l2(r["name"])))
with open(os.path.join(HERE, "summary.md"), "w") as f:
    f.write("| эксперимент | группа | seed | L2 среднее, % | L2 min–max, % | невязка | шагов | время, мин | параметров |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        n = r["name"]
        a, b = l2rng(n)
        res = np.mean([x["mean_residual"] for x in runs(n)])
        f.write(f"| {n} | {r['group']} | {len(runs(n))} | {l2(n) * 100:.3g} | "
                f"{a * 100:.3g}–{b * 100:.3g} | {res:.2e} | {r['steps_total']} | "
                f"{tmin(n):.1f} | {r['n_params']} |\n")
print(open(os.path.join(HERE, "summary.md")).read())
print("seeds:", seeds)
