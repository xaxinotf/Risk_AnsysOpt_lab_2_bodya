
import numpy as np
from scipy.stats import poisson, lognorm
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.graph_objs as go

"""
Управління ризиком — оптимальне резервування (інтерактивне демо)

Модель:
- Сукупні збитки L за період: L = \sum_{i=1}^{N} X_i, де N ~ Poisson(λ),
  X_i ~ Lognormal(μ, σ) незалежні.
- Резерв c >= 0.
- Критерій (мінімізувати): J(c) = k * c + p * E[(L - c)_+],
  де k — ставка/вартість капіталу (річна), p — "штраф" за дефіцит резерву.
- Додатково показуються: P(L > c), E[(L - c)_+] та графік J(c).

Обчислення:
- Монте-Карло моделювання L (N сценаріїв).
- Для ґратки c у [0, Cmax] обчислюємо метрики.
"""

def simulate_aggregate_losses(lmbda=5.0, mu=8.0, sigma=1.0, n_sims=50000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    N = rng.poisson(lmbda, size=n_sims)
    losses = np.zeros(n_sims, dtype=float)
    # параметри для lognorm у scipy: shape=sigma, scale=exp(mu)
    shape = sigma
    scale = np.exp(mu)
    # генеруємо суми збитків
    for i in range(n_sims):
        n_i = N[i]
        if n_i > 0:
            severities = rng.lognormal(mean=mu, sigma=sigma, size=n_i)
            losses[i] = severities.sum()
        # якщо n_i=0, losses[i]=0
    return losses

def compute_metrics(losses, c_grid, k=0.02, p=1.0):
    # E[(L - c)+] та P(L > c) для кожного c
    L = losses[:, None]  # (n,1)
    C = c_grid[None, :]  # (1,m)
    shortfall = np.clip(L - C, 0.0, None)  # (n,m)
    es = shortfall.mean(axis=0)            # E[(L - c)+]
    ruin = (L > C).mean(axis=0)            # P(L > c)
    J = k * c_grid + p * es                # k*c + p*E[(L - c)+]
    return es, ruin, J

app = Dash(__name__)
server = app.server

app.layout = html.Div(className="container", children=[
    html.H1("Оптимальне резервування як інструмент мінімізації ризику"),
    html.P("Інтерактивна демонстрація моделі оптимального резерву з Монте-Карло моделюванням."),
    html.Div(className="controls", children=[
        html.Div(className="row", children=[
            html.Div(children=[
                html.Label("Інтенсивність збиткових подій λ (Poisson)"),
                dcc.Slider(id="lambda", min=0.1, max=20.0, step=0.1, value=5.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1, "paddingRight": "12px"}),
            html.Div(children=[
                html.Label("μ логнормального розподілу (середнє в лог-просторі)"),
                dcc.Slider(id="mu", min=5.0, max=12.0, step=0.1, value=8.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1, "paddingRight": "12px"}),
            html.Div(children=[
                html.Label("σ логнормального розподілу"),
                dcc.Slider(id="sigma", min=0.1, max=2.5, step=0.1, value=1.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        html.Div(className="row", children=[
            html.Div(children=[
                html.Label("Вартість капіталу k (за період)"),
                dcc.Slider(id="k", min=0.0, max=0.2, step=0.005, value=0.02,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1, "paddingRight": "12px"}),
            html.Div(children=[
                html.Label("Штраф p за одиницю дефіциту"),
                dcc.Slider(id="p", min=0.1, max=10.0, step=0.1, value=1.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1, "paddingRight": "12px"}),
            html.Div(children=[
                html.Label("Верхня межа резерву Cmax"),
                dcc.Slider(id="cmax", min=0.0, max=5_000_000.0, step=50_000.0, value=1_500_000.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        html.Div(className="row", children=[
            html.Div(children=[
                html.Label("Сценарії Монте-Карло"),
                dcc.Slider(id="nsims", min=5000, max=150000, step=5000, value=50000,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1, "paddingRight": "12px"}),
            html.Div(children=[
                html.Label("Крок ґратки резерву (тис.)"),
                dcc.Slider(id="cstep", min=10_000.0, max=250_000.0, step=5_000.0, value=25_000.0,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": 1}),
        ], style={"display": "flex", "marginBottom": "8px"}),
    ]),
    html.Hr(),
    html.Div(className="charts", children=[
        dcc.Graph(id="objective-graph"),
        dcc.Graph(id="prob-es-graph"),
        dcc.Graph(id="loss-hist"),
    ]),
    html.Div(id="opt-summary", style={"marginTop": "8px", "fontWeight": "600"}),
    html.Footer(children=html.Small("© Оптимальне резервування • Dash demo"))
], style={"maxWidth": "1150px", "margin": "0 auto", "padding": "16px"})

@callback(
    Output("objective-graph", "figure"),
    Output("prob-es-graph", "figure"),
    Output("loss-hist", "figure"),
    Output("opt-summary", "children"),
    Input("lambda", "value"),
    Input("mu", "value"),
    Input("sigma", "value"),
    Input("k", "value"),
    Input("p", "value"),
    Input("cmax", "value"),
    Input("nsims", "value"),
    Input("cstep", "value"),
)
def update(lmbda, mu, sigma, k, p, cmax, nsims, cstep):
    # Симуляція збитків
    losses = simulate_aggregate_losses(lmbda=float(lmbda), mu=float(mu), sigma=float(sigma),
                                       n_sims=int(nsims), rng_seed=123)
    # Ґратка резерву
    c_grid = np.arange(0.0, float(cmax) + 1.0, float(cstep))
    if len(c_grid) < 2:
        c_grid = np.array([0.0, float(cmax)])
    es, ruin, J = compute_metrics(losses, c_grid, k=float(k), p=float(p))
    j_min_idx = int(np.argmin(J))
    c_star = float(c_grid[j_min_idx])
    J_star = float(J[j_min_idx])
    es_star = float(es[j_min_idx])
    ruin_star = float(ruin[j_min_idx])

    # Фігура 1: J(c)
    fig_obj = go.Figure()
    fig_obj.add_trace(go.Scatter(x=c_grid, y=J, mode="lines", name="J(c) = k·c + p·E[(L-c)+]"))
    fig_obj.add_trace(go.Scatter(x=[c_star], y=[J_star], mode="markers", name="c*", marker={"size": 12}))
    fig_obj.update_layout(title="Цільова функція J(c) та оптимум c*",
                          xaxis_title="Резерв c",
                          yaxis_title="J(c)",
                          hovermode="x unified")

    # Фігура 2: P(L>c) та E[(L-c)+]
    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(x=c_grid, y=ruin, mode="lines", name="Ймовірність дефіциту P(L>c)"))
    fig_pe.add_trace(go.Scatter(x=c_grid, y=es, mode="lines", name="Очікуваний дефіцит E[(L-c)+]", yaxis="y2"))
    fig_pe.update_layout(title="Ймовірність дефіциту та очікуваний дефіцит",
                         xaxis_title="Резерв c",
                         yaxis=dict(title="P(L>c)"),
                         yaxis2=dict(title="E[(L-c)+]", overlaying="y", side="right"),
                         hovermode="x unified")

    # Фігура 3: Розподіл L
    hist_values = losses
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=hist_values, nbinsx=60, name="L"))
    fig_hist.update_layout(title="Емпіричний розподіл сукупних збитків L (Монте-Карло)",
                           xaxis_title="L", yaxis_title="Частота",
                           barmode="overlay")

    summary = f"Оптимальний резерв c* = {c_star:,.0f};  J(c*) = {J_star:,.2f};  P(L>c*) = {ruin_star:.4f};  E[(L-c*)+] = {es_star:,.0f}"

    return fig_obj, fig_pe, fig_hist, summary

if __name__ == "__main__":
    app.run(debug=True)
