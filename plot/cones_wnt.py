from .plotcore import load, build_df_wnt, select
import plotly.graph_objects as go

def plot(df):
    fig = go.Figure(data=[go.Cone(
        x=df.loc[df]['x1'],
        y=df.loc[df]['x2'],
        z=df.loc[df]['x3'],
        u=df.loc[df]['q1'],
        v=df.loc[df]['q2'],
        w=df.loc[df]['q3'],
        sizemode='absolute',
        sizeref=2,
        color=df.loc[df]['w']
    )])

    def fun(scene):
        scene.aspectmode = 'data'
        return
    fig.for_each_scene(fun)
    return fig

def save(fig):
    fig.write_html(fname.replace('data','animations').replace('.pkl','_vectorfield.html'),
                   include_plotlyjs='directory', full_html=False, animation_opts={'frame': {'duration': 100}})

if __name__ == "__main__":
    fname = input('Enter data filename: ')  # 'data/test1.pkl'
    T_plot = int(input('timestep to plot: ') or -1)
    data, kwargs, fname = load(fname)
    df, kwargs = build_df_wnt(data, kwargs)
    df_t = select(df, T_plot, kwargs)
    fig = plot(df_t)
    save(fig, fname)