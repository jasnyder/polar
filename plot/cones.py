import plotly.graph_objects as go

try:
    from .plotcore import load, build_df, select
except ModuleNotFoundError:
    from plotcore import load, build_df, select

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

def save(fig, fname, T_plot):
    fig.write_html(fname.replace('data','animations').replace('.pkl',f'_vectorfield_t_{T_plot}.html'),
                   include_plotlyjs='directory', full_html=False, animation_opts={'frame': {'duration': 100}})

if __name__ == "__main__":
    fname = input('Enter data filename (default: most recent): ')  or 'most recent'# 'data/test1.pkl'
    data, kwargs, fname = load(fname)
    T_plot = int(input('timestep to plot (default: last): ') or -1)
    df, kwargs = build_df(data, kwargs)
    df_t = select(df, T_plot, kwargs)
    fig = plot(df_t)
    save(fig, fname, T_plot)
