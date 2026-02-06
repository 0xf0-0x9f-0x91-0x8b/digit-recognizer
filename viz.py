import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_img(t):
    t = t.detach().cpu().squeeze(0).numpy()
    fig = px.imshow(t, color_continuous_scale="gray")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.show()

def display_imgs(t1, t2):
    t1 = t1.detach().cpu().squeeze(0).numpy()
    t2 = t2.detach().cpu().squeeze(0).numpy()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02)
    fig.add_trace(
        go.Heatmap(z=t1, colorscale="gray", showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=t2, colorscale="gray", showscale=False),
        row=1, col=2
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()