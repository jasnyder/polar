try:
    from .plotcore import load, select, build_dfs_wnt_ligand
except:
    from plotcore import load, select, build_dfs_wnt_ligand
import plotly.graph_objects as go

def plot(df):
    fig = go.Figure(data=[go.Cone(
        x=df['x1'],
        y=df['x2'],
        z=df['x3'],
        u=df['q1'],
        v=df['q2'],
        w=df['q3'],
        sizemode='absolute',
        sizeref=2
    )])

    def fun(scene):
        scene.aspectmode = 'data'
        return
    fig.for_each_scene(fun)
    return fig

def save(fig, fname):
    fig.write_html(fname.replace('data','animations').replace('.pkl','_vectorfield.html'),
                   include_plotlyjs='directory', full_html=False, animation_opts={'frame': {'duration': 100}})

if __name__ == "__main__":
    fname = input('Enter data filename (default: most recent): ') or 'most recent'
    T_plot = int(input('timestep to plot: ') or -1)
    data, kwargs, fname = load(fname)
    df, df_lig, kwargs = build_dfs_wnt_ligand(data, kwargs)
    df_t = select(df, T_plot, kwargs)
    fig = plot(df_t)
    save(fig, fname)