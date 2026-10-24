"""Анализ гиперпараметров PINN для уравнения Бюргерса.

Каждый эксперимент меняет один параметр относительно базового примера DeepXDE
(3x20, tanh, Glorot normal, Adam lr=1e-3 15000 итераций -> L-BFGS,
2540/80/160 точек, распределение Hammersley).

Запуск всех экспериментов в 8 процессах:
    python run_experiments.py --workers 8
Один эксперимент:
    python run_experiments.py --only base_s1
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATASET = os.path.join(HERE, "..", "burgers", "Burgers.npz")

BASE = dict(
    layers=3, width=20, activation="tanh", init="Glorot normal",
    optimizer="adam", lr=1e-3, adam_iters=15000, lbfgs=True,
    num_domain=2540, num_boundary=80, num_initial=160, dist="Hammersley",
    rar_rounds=0, seed=1,
    lbfgs_maxiter=None, decay=None, weight_decay=0, nncg_iters=0, resample=0, float64=False,
)


def experiments():
    exps = []

    def add(group, name, **kw):
        exps.append(dict(BASE, group=group, name=name, **kw))

    for s in (1, 2, 3):
        add("base", f"base_s{s}", seed=s)
    for n in (1, 2, 4, 6, 8):
        add("depth", f"depth_{n}", layers=n)
    for w in (5, 10, 40, 80):
        add("width", f"width_{w}", width=w)
    for a in ("sin", "sigmoid", "silu", "gelu", "relu", "elu"):
        add("activation", f"act_{a}", activation=a)
    for i in ("Glorot uniform", "He normal", "He uniform", "zeros"):
        add("init", "init_" + i.replace(" ", "_"), init=i)
    for n in (250, 500, 1000, 5000, 10000):
        add("points", f"points_{n}", num_domain=n)
    for d in ("pseudo", "uniform", "LHS", "Halton", "Sobol"):
        add("dist", f"dist_{d}", dist=d)
    add("dist", "rar", rar_rounds=3)
    add("optimizer", "opt_adam_only", lbfgs=False)
    add("optimizer", "opt_adam_only_50k", lbfgs=False, adam_iters=50000)
    add("optimizer", "opt_lbfgs_only", adam_iters=0)
    add("optimizer", "opt_rmsprop", optimizer="rmsprop")
    add("optimizer", "opt_sgd", optimizer="sgd")
    for lr in (1e-2, 1e-4):
        add("lr", f"lr_{lr:g}", lr=lr)
    for it in (2000, 5000, 30000):
        add("adam_iters", f"adam_{it}", adam_iters=it)

    # --- вторая серия ---
    # сетка «слои × нейроны» (вместе с depth_*, width_* и базовой даёт {2,3,4,6} × {10,20,40})
    for nl in (2, 4, 6):
        for w in (10, 40):
            add("grid", f"grid_{nl}x{w}", layers=nl, width=w)
    # только Adam, разное число итераций (есть 15k и 50k)
    for it in (5000, 30000, 100000):
        add("adam_only", f"adam_only_{it // 1000}k", lbfgs=False, adam_iters=it)
    # ограничение числа шагов L-BFGS после 15k Adam
    for m in (500, 1000, 2000):
        add("lbfgs_cap", f"lbfgs_max_{m}", lbfgs_maxiter=m)
    # ещё оптимизаторы
    add("optimizer", "opt_adamw", optimizer="adamw", weight_decay=1e-4)
    add("optimizer", "opt_adam_decay", decay=["inverse time", 1000, 0.5])
    add("optimizer", "opt_nncg", nncg_iters=1000)
    # точки на границе и в начальный момент
    for nb in (20, 320):
        add("bcic", f"bc_{nb}", num_boundary=nb)
    for ni in (40, 640):
        add("bcic", f"ic_{ni}", num_initial=ni)
    # случайные точки, пересэмплируемые каждые 1000 итераций Adam
    add("dist", "resample_pseudo", dist="pseudo", resample=1000)
    # адаптивная активация L-LAAF
    add("activation", "act_laaf5_tanh", activation="LAAF-5 tanh")

    # --- третья серия: комбинации параметров ---
    acts = ("sin", "sigmoid", "silu", "gelu")
    inits = {"GU": "Glorot uniform", "HN": "He normal", "HU": "He uniform"}
    opts = {"rmsprop": dict(optimizer="rmsprop"), "adamw": dict(optimizer="adamw", weight_decay=1e-4),
            "adamonly": dict(lbfgs=False)}
    # активация × инициализация (tanh × init и act × Glorot normal уже есть выше)
    for a in acts:
        for k, i in inits.items():
            add("act_init", f"ai_{a}_{k}", activation=a, init=i)
    # активация × оптимизатор (tanh × opt и act × Adam→L-BFGS уже есть выше)
    for a in acts:
        for k, o in opts.items():
            add("act_opt", f"ao_{a}_{k}", activation=a, **o)
    # готовые конфигурации: меняется сразу несколько параметров
    combos = {
        "cfg_tanh_gu_4x20": dict(init="Glorot uniform", layers=4),
        "cfg_tanh_gu_6x20_30k": dict(init="Glorot uniform", layers=6, adam_iters=30000),
        "cfg_tanh_gu_4x20_5000pts": dict(init="Glorot uniform", layers=4, num_domain=5000),
        "cfg_tanh_2x20_fast": dict(layers=2, adam_iters=5000),
        "cfg_sin_gu_4x20": dict(activation="sin", init="Glorot uniform", layers=4),
        "cfg_sin_gu_4x40": dict(activation="sin", init="Glorot uniform", layers=4, width=40),
        "cfg_sin_gu_rmsprop_4x20": dict(activation="sin", init="Glorot uniform", layers=4, optimizer="rmsprop"),
        "cfg_sigmoid_gu_4x20": dict(activation="sigmoid", init="Glorot uniform", layers=4),
        "cfg_silu_hn_rmsprop_4x20": dict(activation="silu", init="He normal", layers=4, optimizer="rmsprop"),
        "cfg_gelu_hu_adamw": dict(activation="gelu", init="He uniform", optimizer="adamw", weight_decay=1e-4),
        "cfg_relu_hn": dict(activation="relu", init="He normal"),
        "cfg_elu_hn": dict(activation="elu", init="He normal"),
        "cfg_laaf_gu_4x20": dict(activation="LAAF-5 tanh", init="Glorot uniform", layers=4),
    }
    for k, v in combos.items():
        add("combo", k, **v)

    # --- стабильность: базовая модель на 10 seed во float32 и float64 ---
    # (документация DeepXDE: если L-BFGS останавливается раньше времени — float64)
    for k in range(4, 11):
        add("precision", f"seed_f32_{k}", seed=k)
    for k in range(1, 11):
        add("precision", f"seed_f64_{k}", seed=k, float64=True)

    # варианты, близкие к базовой модели: ещё 2 seed, чтобы отделить эффект от шума
    contenders = ("depth_4", "depth_6", "width_40", "act_sigmoid", "init_Glorot_uniform",
                  "init_He_normal", "init_He_uniform", "points_5000", "dist_LHS",
                  "dist_pseudo", "rar", "opt_rmsprop",
                  # здесь L-BFGS остановился за 0–5 шагов: проверяем, повторяется ли
                  "dist_Sobol", "points_10000", "width_10",
                  # вторая серия
                  "depth_2", "adam_2000", "adam_5000", "adam_30000", "opt_sgd", "opt_adam_only",
                  "opt_lbfgs_only", "points_500", "points_1000",
                  "grid_2x10", "grid_2x40", "grid_4x10", "grid_4x40", "grid_6x10", "grid_6x40",
                  "lbfgs_max_500", "lbfgs_max_1000", "lbfgs_max_2000",
                  "opt_adamw", "opt_adam_decay", "opt_nncg",
                  "bc_20", "bc_320", "ic_40", "ic_640", "resample_pseudo", "act_laaf5_tanh")
    for e in list(exps):
        if e["name"] in contenders or e["group"] in ("act_init", "act_opt", "combo"):
            for s in (2, 3):
                exps.append(dict(e, name=f"{e['name']}_s{s}", seed=s, base_name=e["name"]))
    return exps


def run(cfg):
    os.environ["DDE_BACKEND"] = "pytorch"
    # DeepXDE считает на GPU Apple (MPS); NNCG нужна eigh, которой на MPS нет — пусть идёт на CPU
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    import deepxde as dde

    # DeepXDE сам выбирает GPU Apple (MPS); для такой маленькой сети CPU в 3–20 раз быстрее,
    # а главное — каждый процесс получает своё ядро и время честно сравнимо
    torch.set_default_device("cpu")

    dde.config.set_random_seed(cfg["seed"])
    if cfg.get("float64"):
        dde.config.set_default_float("float64")

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_b: on_b)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_i: on_i)
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=cfg["num_domain"], num_boundary=cfg["num_boundary"],
        num_initial=cfg["num_initial"], train_distribution=cfg["dist"],
    )
    reg = ["l2", cfg["weight_decay"]] if cfg.get("weight_decay") else None
    net = dde.nn.FNN([2] + [cfg["width"]] * cfg["layers"] + [1], cfg["activation"], cfg["init"],
                     regularization=reg)
    model = dde.Model(data, net)
    if cfg.get("lbfgs_maxiter"):
        m = cfg["lbfgs_maxiter"]
        dde.optimizers.set_LBFGS_options(maxiter=m)
        # DeepXDE (PyTorch) гоняет L-BFGS порциями по iter_per_step шагов и считает
        # размер порции один раз при импорте — обновляем вручную
        from deepxde.optimizers.config import LBFGS_options as LO
        LO["iter_per_step"] = min(1000, m)
        LO["fun_per_step"] = LO["maxfun"] * LO["iter_per_step"] // m
    n_params = sum(p.numel() for p in net.parameters())

    steps, losses = [], []

    def record(lh):
        for s, l in zip(lh.steps, lh.loss_train):
            if steps and s <= steps[-1]:
                continue
            steps.append(int(s))
            losses.append([float(v) for v in l])

    t_adam = t_lbfgs = 0.0
    lbfgs_steps = nncg_steps = 0

    def train_round(adam_iters):
        nonlocal t_adam, t_lbfgs, lbfgs_steps
        if adam_iters:
            model.compile(cfg["optimizer"], lr=cfg["lr"], decay=cfg.get("decay"))
            cbs = [dde.callbacks.PDEPointResampler(period=cfg["resample"])] if cfg.get("resample") else []
            t0 = time.time()
            lh, _ = model.train(iterations=adam_iters, display_every=500, verbose=0, callbacks=cbs)
            t_adam += time.time() - t0
            record(lh)
        if cfg["lbfgs"]:
            net.regularizer = None  # L-BFGS в DeepXDE не поддерживает weight decay
            model.compile("L-BFGS")
            s0 = model.train_state.step
            t0 = time.time()
            lh, _ = model.train(display_every=500, verbose=0)
            t_lbfgs += time.time() - t0
            lbfgs_steps += model.train_state.step - s0
            record(lh)

    train_round(cfg["adam_iters"])
    if cfg.get("nncg_iters"):
        dde.optimizers.set_NNCG_options(rank=50, mu=1e-1)
        model.compile("NNCG")
        s0 = model.train_state.step
        lh, _ = model.train(iterations=cfg["nncg_iters"], display_every=100, verbose=0)
        nncg_steps = model.train_state.step - s0
        record(lh)
    n_added = 0
    for _ in range(cfg["rar_rounds"]):
        X = geomtime.random_points(100000)
        err = np.abs(model.predict(X, operator=pde)).ravel()
        top = X[np.argsort(-err)[:100]]
        data.add_anchors(top)
        n_added += len(top)
        train_round(1000)

    d = np.load(DATASET)
    t, x, exact = d["t"], d["x"], d["usol"].T
    xx, tt = np.meshgrid(x, t)
    Xg = np.vstack((xx.ravel(), tt.ravel())).T
    u_true = exact.ravel()[:, None]
    u_pred = model.predict(Xg)
    res = np.abs(model.predict(Xg, operator=pde))

    out = dict(
        cfg, n_params=int(n_params),
        l2=float(dde.metrics.l2_relative_error(u_true, u_pred)),
        max_err=float(np.max(np.abs(u_true - u_pred))),
        mean_residual=float(res.mean()),
        final_loss=float(sum(losses[-1])),
        steps_total=steps[-1], time_adam=t_adam, time_lbfgs=t_lbfgs,
        time_total=t_adam + t_lbfgs, rar_added=n_added,
        lbfgs_steps=int(lbfgs_steps), nncg_steps=int(nncg_steps),
    )
    np.savez_compressed(
        os.path.join(RESULTS, cfg["name"] + ".npz"),
        u_pred=u_pred.reshape(xx.shape).astype(np.float32),
        steps=np.array(steps), losses=np.array(losses),
        train_x=data.train_x_all.astype(np.float32),
    )
    with open(os.path.join(RESULTS, cfg["name"] + ".json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    todo = [e for e in experiments()
            if (not args.only or e["name"] in args.only)
            and not os.path.exists(os.path.join(RESULTS, e["name"] + ".json"))]
    # тяжёлые эксперименты первыми, чтобы не ждать их в конце
    heavy = ("adam_only_100k", "opt_nncg", "opt_nncg_s2", "opt_nncg_s3", "opt_adam_only_50k", "depth_8",
             "width_80", "points_10000", "points_10000_s2", "points_10000_s3", "grid_6x40", "grid_6x40_s2",
             "grid_6x40_s3")
    todo.sort(key=lambda e: e["name"] not in heavy)
    print(f"Экспериментов к запуску: {len(todo)}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, e): e["name"] for e in todo}
        for f in as_completed(futs):
            name = futs[f]
            try:
                r = f.result()
                print(f"[ok] {name}: L2={r['l2']:.3e}, time={r['time_total']:.0f}s", flush=True)
            except Exception as ex:
                print(f"[FAIL] {name}: {ex!r}", flush=True)


if __name__ == "__main__":
    main()
