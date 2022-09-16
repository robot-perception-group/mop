import plotly.graph_objects as go

def viz_smpl_seq(V,F,color='#dba773'):

    fig = go.Figure(
        data=[go.Mesh3d(x=V[0,:,0],y=V[0,:,1],z=V[0,:,2],
                i=F[:,0],j=F[:,1],k=F[:,2],color=color)],
        layout=go.Layout(
            updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
        frames=[go.Frame(data=[go.Mesh3d(x=V[idx,:,0],y=V[idx,:,1],z=V[idx,:,2],
                i=F[:,0],j=F[:,1],k=F[:,2],color=color)]) for idx in range(1,V.shape[0])]
    )

    return fig
