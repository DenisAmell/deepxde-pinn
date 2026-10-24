"""Честная оценка времени обучения каждой конфигурации.

Во время run_experiments.py параллельно шло до 14 процессов на ядрах разной мощности,
поэтому замеренное там время несопоставимо. Здесь каждая конфигурация запускается
по очереди, одна на машине: меряем время одной итерации Adam и одной итерации L-BFGS
и умножаем на фактическое число шагов из results/<name>.json.

    python bench_time.py      # пишет results/timing.json
"""

import json
import os
import time

import run_experiments as R

ADAM_N, LBFGS_N = 200, 40


def bench(cfg):
    os.environ["DDE_BACKEND"] = "pytorch"
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
    dde.optimizers.set_LBFGS_options(maxiter=LBFGS_N)

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, 0.99))
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_b: on_b)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_i: on_i)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=cfg["num_domain"],
                            num_boundary=cfg["num_boundary"], num_initial=cfg["num_initial"],
                            train_distribution=cfg["dist"])
    reg = ["l2", cfg["weight_decay"]] if cfg.get("weight_decay") else None
    net = dde.nn.FNN([2] + [cfg["width"]] * cfg["layers"] + [1], cfg["activation"], cfg["init"],
                     regularization=reg)
    model = dde.Model(data, net)
    opt = cfg["optimizer"]
    model.compile(opt, lr=cfg["lr"])
    model.train(iterations=20, verbose=0)  # прогрев
    # мерим 5 раз и берём минимум: отсекаем случайные помехи и энергоэффективные ядра M4
    chunks = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.train(iterations=ADAM_N, verbose=0, display_every=ADAM_N)
        chunks.append((time.perf_counter() - t0) / ADAM_N)
    t_adam = min(chunks)
    net.regularizer = None  # L-BFGS в DeepXDE не поддерживает weight decay
    model.compile("L-BFGS")
    s0 = model.train_state.step
    t0 = time.perf_counter()
    model.train(verbose=0, display_every=LBFGS_N)
    n = max(model.train_state.step - s0, 1)
    t_lbfgs = (time.perf_counter() - t0) / n
    t_nncg = 0.0
    if cfg.get("nncg_iters"):
        dde.optimizers.set_NNCG_options(rank=50, mu=1e-1)
        model.compile("NNCG")
        model.train(iterations=2, verbose=0)
        t0 = time.perf_counter()
        model.train(iterations=20, verbose=0)
        t_nncg = (time.perf_counter() - t0) / 20
    return t_adam, t_lbfgs, t_nncg


def main():
    out_path = os.path.join(R.RESULTS, "timing.json")
    timing = json.load(open(out_path)) if os.path.exists(out_path) else {}
    # время зависит только от архитектуры, точек и оптимизатора — seed не нужен
    for cfg in R.experiments():
        if "base_name" in cfg or cfg["name"] in timing or cfg["name"] in ("base_s2", "base_s3"):
            continue
        # стабильность: время одинаково для всех seed — мерим один float32 (= base_s1) и один float64
        if cfg["name"].startswith("seed_f32_") or (cfg["name"].startswith("seed_f64_") and cfg["name"] != "seed_f64_1"):
            continue
        if not os.path.exists(os.path.join(R.RESULTS, cfg["name"] + ".json")):
            continue
        res = json.load(open(os.path.join(R.RESULTS, cfg["name"] + ".json")))
        t_adam, t_lbfgs, t_nncg = bench(cfg)
        adam_steps = res["adam_iters"] + 1000 * res["rar_rounds"]
        lbfgs_steps = max(res["steps_total"] - adam_steps, 0) if res["lbfgs"] else 0
        if not res["lbfgs"]:
            adam_steps = res["steps_total"]
        est = t_adam * adam_steps + t_lbfgs * lbfgs_steps
        timing[cfg["name"]] = dict(t_adam_ms=t_adam * 1e3, t_lbfgs_ms=t_lbfgs * 1e3, t_nncg_ms=t_nncg * 1e3,
                                   adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, est_min=est / 60)
        print(f"{cfg['name']}: Adam {t_adam * 1e3:.1f} мс/ит, L-BFGS {t_lbfgs * 1e3:.1f} мс/ит, "
              f"оценка {est / 60:.1f} мин", flush=True)
        json.dump(timing, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
