import plotly.graph_objects as go
import plotly.express as px
from networkx import node_link_graph, node_link_data
from networkx.readwrite import json_graph

import networkx as nx


def node_trace(x, y, text: str, size: int):
    return go.Scatter(
        x=(x,), y=(y,),
        mode='markers',
        text=text,
        # hovertext=hover_data,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color='#B0BEC5',
            size=min(size, 40),
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))


def plot_network(g: nx.DiGraph) -> go.Figure:
    # fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
    #                  hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    # fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))
    #
    # # Update hover order
    # fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
    #                                              "%{customdata[1]}",
    #                                              "Size: %{customdata[2]}"]))

    edge_x = []
    edge_y = []
    w = list()
    for u, v in g.edges:
        ux, uy = g.nodes[u]['pos']
        vx, vy = g.nodes[v]['pos']
        edge_x.append(ux)
        edge_x.append(vx)
        edge_x.append(None)
        edge_y.append(uy)
        edge_y.append(vy)
        edge_y.append(None)
        w.append(w)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hovertext=w,
        text=w,
        hoverinfo='text',
        mode='lines')

    node_x = []
    node_y = []
    hover_data = []
    sizes = []
    # TODO: assign color by year/bin
    for node in g.nodes:
        print(node)
        x, y = g.nodes[node]['pos']
        node_x.append(float(x))
        node_y.append(float(y))
        s = int(g.nodes[node]['size'])
        hover_data.append(g.nodes[node]['name'] + f'\nSize: {s}')
        sizes.append(s)

    assert len(hover_data) == len(node_y) == len(node_x), "Wrong lengths in arrays"

    # fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
    #                  hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    # fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    # fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
    #                                              "%{customdata[1]}",
    #                                              "Size: %{customdata[2]}"]))

    node_ts = [node_trace(x, y, t, s) for x, y, t, s in zip(node_x, node_y, hover_data, sizes)]

    fig = go.Figure(data=[edge_trace, *node_ts],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Prepare figure range
    # print(edge_y)
    # print(edge_x)
    # x_range = (min(edge_x) - abs(min(edge_x) * .15), max(edge_x) + abs((max(edge_x)) * .15))
    # y_range = (min(edge_y) - abs(min(edge_y) * .15), max(edge_y) + abs((max(edge_y)) * .15))

    # Update axes ranges
    # fig.update_xaxes(range=x_range)
    # fig.update_yaxes(range=y_range)

    return fig
