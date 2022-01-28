from unittest import skip
import plotly.express as px

try:
    from .plotcore import load, build_dfs_wnt_ligand
except ModuleNotFoundError:
    from plotcore import load, build_dfs_wnt_ligand


def plot(df, color = 'x1'):
    range_x = (df['x1'].min(), df['x1'].max())
    range_y = (df['x2'].min(), df['x2'].max())
    range_z = (df['x3'].min(), df['x3'].max())
    fig = px.scatter_3d(df, x='x1', y='x2', z='x3', animation_frame='t',
                        color = color, range_x=range_x, range_y=range_y, range_z=range_z)

    def fun(scene):
        scene.aspectmode = 'data'
        return
    fig.for_each_scene(fun)
    return fig


def save(fig, fname):
    fig.write_html(fname.replace('data','animations').replace('.pkl','_ligand.html'), include_plotlyjs='directory',
                   full_html=False, animation_opts={'frame': {'duration': 50}})


if __name__ == '__main__':
    fname = input('Enter data filename (default: most recent): ') or 'most recent' # 'data/test1.pkl'
    skipframes = input('plot ever n frames? (default: 1) ') or 1
    color = input('color by? (default: x1) ') or 'x1'
    data, kwargs, fname = load(fname)
    df, df_lig, kwargs = build_dfs_wnt_ligand(data, kwargs, skipframes = int(skipframes))
    fig = plot(df_lig, color=color)
    save(fig, fname)
