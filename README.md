
# Оптимальне резервування — Dash демо

Це мінімальний застосунок для моделювання оптимального страхового резерву:

- Модель сукупних збитків L = \sum_{i=1}^N X_i, де N ~ Poisson(λ), X_i ~ Lognormal(μ, σ).
- Мінімізується J(c) = k·c + p·E[(L - c)_+].
- Результат: оптимальний резерв c* та супутні метрики.

<img width="1316" height="831" alt="image" src="https://github.com/user-attachments/assets/b172fa76-4430-4784-b545-b57a0b4adce9" />

<img width="1281" height="480" alt="image" src="https://github.com/user-attachments/assets/9173e151-9cc6-4c0e-a266-5913d9571c14" />

<img width="987" height="539" alt="image" src="https://github.com/user-attachments/assets/b9453fdf-f88b-46df-a9d0-ef4952a4fa14" />


## Вимоги

- Python 3.10 (рекомендовано віртуальне середовище)
- Пакети з `requirements.txt`

## Інсталяція та запуск

```bash
cd /mnt/data/optimal_reserving_app
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt

python app.py
```

Після запуску відкрийте http://127.0.0.1:8050 у браузері.

## Налаштування

- λ — інтенсивність подій (Poisson)
- μ, σ — параметри логнормального розподілу збитку
- k — вартість капіталу на одиницю резерву
- p — штраф за одиницю дефіциту (expected shortfall)
- Cmax — верхня межа резерву (ґратка від 0 до Cmax)
- Сценарії Монте-Карло — кількість симуляцій
- Крок ґратки резерву — дискретизація c

## Примітки

- Розрахунок J(c) побудований на очікуваному дефіциті E[(L-c)_+].
- Для більш «регуляторних» метрик (VaR/CVaR) можна розширити модель.
