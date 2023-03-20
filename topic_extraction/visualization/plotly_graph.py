import plotly.graph_objects as go

import networkx as nx


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
    for edge in g.edges():
        x0, y0 = g.nodes[edge[0]]['pos']
        x1, y1 = g.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    hover_data = []
    for node in g.nodes():
        x, y = g.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        hover_data.append(g.nodes[node]['name'])

    assert len(hover_data) == len(node_y) == len(node_x), "Wrong lengths in arrays"

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=hover_data,
        hoverinfo='skip',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
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
    print(edge_y)
    print(edge_x)
    x_range = (min(edge_x) - abs(min(edge_x) * .15), max(edge_x) + abs((max(edge_x)) * .15))
    y_range = (min(edge_y) - abs(min(edge_y) * .15), max(edge_y) + abs((max(edge_y)) * .15))

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    return fig
