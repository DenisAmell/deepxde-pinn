# PINN-задачи на DeepXDE

Две задачи из примеров [DeepXDE](https://github.com/lululxvi/deepxde/tree/master/examples), решённые физически-информированными нейросетями (PINN) на бэкенде PyTorch.

| Задача | Файл | Что ищем | Результат |
|---|---|---|---|
| **Прямая** — уравнение Бюргерса | `burgers/burgers_forward.py` | решение $u(x,t)$ по уравнению, начальному и граничным условиям | относительная L2-ошибка ≈ 0.4 % |
| **Обратная** — уравнение диффузии | `diffusion_inverse/diffusion_inverse.py` | неизвестный коэффициент диффузии $C$ по 10 измерениям (истинное $C = 1$) | $C \approx 0.998$ |

Источники: `examples/pinn_forward/Burgers.py` и `examples/pinn_inverse/diffusion_1d_inverse.py`.

## Структура

```
.
├── PINN_DeepXDE.ipynb          # ноутбук: теория PINN, разбор кода, графики
├── requirements.txt
├── burgers/
│   ├── burgers_forward.py      # прямая задача
│   ├── Burgers.npz             # эталонное решение для проверки
│   └── loss.dat, train.dat, test.dat   # результаты прогона
└── diffusion_inverse/
    ├── diffusion_inverse.py    # обратная задача
    └── loss.dat, train.dat, test.dat   # результаты прогона
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Прямая задача (15 000 итераций Adam, затем L-BFGS, ~8 мин):

```bash
cd burgers && python burgers_forward.py && cd ..
```

Обратная задача (50 000 итераций Adam, ~5 мин):

```bash
cd diffusion_inverse && python diffusion_inverse.py && cd ..
```

Каждый скрипт сохраняет `loss.dat`, `train.dat`, `test.dat` в папку запуска и в конце открывает графики. Без окон с графиками: `MPLBACKEND=Agg python ...`.

Ноутбук:

```bash
pip install jupyter
jupyter notebook PINN_DeepXDE.ipynb
```

## Кратко о методе

Нейросеть $u_\theta(x,t)$ обучается так, чтобы в наборе точек выполнялись уравнение (невязка PDE → 0), граничные (BC) и начальные (IC) условия. Производные считаются автоматическим дифференцированием. В обратной задаче неизвестный $C$ объявлен обучаемой переменной (`dde.Variable`) и подбирается вместе с весами сети. Однозначно определить его позволяют измерения `PointSetBC`.
