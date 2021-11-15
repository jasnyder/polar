import plotly.graph_objects as go

from .plotcore import load, build_df, select

def plot(df):
    fig = go.Figure(data=[go.Cone(
        x=df['x1'],
        y=df['x2'],
        z=df['x3'],
        u=df['q1'],
        v=df['q2'],
        w=df['q3'],
        sizemode='absolute',
        sizeref=2,
    )])

    def fun(scene):
        scene.aspectmode = 'data'
        return
    fig.for_each_scene(fun)
    return fig

def save(fig, fname):
    fig.write_html(f'animations/{fname[5:-4]}_vectorfield.html',
                   include_plotlyjs='directory', full_html=False, animation_opts={'frame': {'duration': 100}})

if __name__ == "__main__":
    fname = input('Enter data filename: ')  # 'data/test1.pkl'
    T_plot = int(input('timestep to plot: ') or -1)
    data, kwargs = load(fname)
    df, kwargs = build_df(data, kwargs)
    df_t = select(df, T_plot, kwargs)
    fig = plot(df_t)
    save(fig, fname)
