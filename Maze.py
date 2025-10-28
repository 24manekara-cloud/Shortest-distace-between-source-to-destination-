import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Distance-Vector Routing Simulator (DP Matrix)", layout="wide")

INF = float('inf')


# -------------------------------- Helper Functions --------------------------------
def ensure_router_tables_from_graph(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    graph = {i: {} for i in range(len(nodes))}
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        ui, vi = idx[u], idx[v]
        graph[ui][vi] = w
        graph[vi][ui] = w
    return nodes, idx, graph


def bellman_ford_dp_matrix(graph, num_nodes, source_idx):
    dp = [[INF for _ in range(num_nodes)] for _ in range(num_nodes)]
    parent = [[-1 for _ in range(num_nodes)] for _ in range(num_nodes)]

    dp[0][source_idx] = 0

    for k in range(1, num_nodes):
        for v in range(num_nodes):
            dp[k][v] = dp[k - 1][v]
            parent[k][v] = parent[k - 1][v]

        for u, edges in graph.items():
            for v, w in edges.items():
                if dp[k - 1][u] != INF and dp[k - 1][u] + w < dp[k][v]:
                    dp[k][v] = dp[k - 1][u] + w
                    parent[k][v] = u

    return dp, parent


def trace_path(parent, source_idx, dest_idx, k):
    path = []
    cur = dest_idx
    i = k

    while i >= 0 and cur != -1 and cur != source_idx:
        path.append(cur)
        cur = parent[i][cur]
        i -= 1

    if cur == source_idx:
        path.append(source_idx)
        path.reverse()
        return path

    return None


# -------------------------------- UI Layout --------------------------------
st.title("Distance-Vector Routing Simulator — Bellman-Ford as DP Matrix")

col1, col2 = st.columns([1, 2])

# -----------------------------------------------------------------------------
# Left Controls
# -----------------------------------------------------------------------------
with col1:
    st.header("Controls")

    if 'G' not in st.session_state:
        st.session_state.G = nx.Graph()
    if 'dp_computed' not in st.session_state:
        st.session_state.dp_computed = False

    if st.button("Create sample topology"):
        st.session_state.G = nx.Graph()
        st.session_state.G.add_weighted_edges_from([
            ("A", "B", 1),
            ("A", "C", 5),
            ("B", "C", 2),
            ("B", "D", 4),
            ("C", "D", 1),
            ("C", "E", 3),
            ("D", "E", 2)
        ])
        st.session_state.dp_computed = False

    st.write("---")
    st.subheader("Edit topology")

    new_node = st.text_input("Add router", value="")
    if st.button("Add router") and new_node:
        if new_node in st.session_state.G.nodes():
            st.warning("Router already exists")
        else:
            st.session_state.G.add_node(new_node)
            st.session_state.dp_computed = False

    nodes_list = list(st.session_state.G.nodes())
    if len(nodes_list) >= 2:
        u = st.selectbox("From", options=nodes_list)
        v = st.selectbox("To", options=nodes_list)
        w = st.number_input("Cost", min_value=0.0, value=1.0)

        if st.button("Add / Update link"):
            if u != v:
                st.session_state.G.add_edge(u, v, weight=float(w))
                st.session_state.dp_computed = False

    if st.button("Clear topology"):
        st.session_state.G = nx.Graph()
        st.session_state.dp_computed = False

    st.write("---")
    st.subheader("Run Bellman-Ford DP")

    if nodes_list:
        source = st.selectbox("Source", options=nodes_list)
        if st.button("Compute DP"):
            nodes_order, idx_map, graph_int = ensure_router_tables_from_graph(st.session_state.G)
            n = len(nodes_order)
            s_idx = idx_map[source]
            dp, parent = bellman_ford_dp_matrix(graph_int, n, s_idx)

            st.session_state.dp = dp
            st.session_state.parent = parent
            st.session_state.nodes_order = nodes_order
            st.session_state.idx_map = idx_map
            st.session_state.dp_computed = True

            st.success("DP computed successfully")


# -----------------------------------------------------------------------------
# Right Visualization
# -----------------------------------------------------------------------------
with col2:
    st.header("Visualizations")

    if len(st.session_state.G.nodes()) > 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        pos = nx.spring_layout(st.session_state.G, seed=42)
        nx.draw(st.session_state.G, pos, with_labels=True, node_size=700)
        labels = nx.get_edge_attributes(st.session_state.G, "weight")
        nx.draw_networkx_edge_labels(st.session_state.G, pos, edge_labels=labels)
        ax.set_axis_off()
        st.pyplot(fig)

    st.write("---")

    if st.session_state.dp_computed:
        dp = st.session_state.dp
        parent = st.session_state.parent
        nodes_order = st.session_state.nodes_order
        idx_map = st.session_state.idx_map
        n = len(nodes_order)

        st.subheader("DP Matrix")
        k_sel = st.slider("Iteration k", 0, n - 1, n - 1)

        df = pd.DataFrame(dp, columns=nodes_order, index=[f"k={i}" for i in range(n)])

        # display dp table
        html = df.to_html().replace("inf", "∞")
        st.markdown(html, unsafe_allow_html=True)

        st.write("---")
        st.subheader("Parent Matrix")
        dfp = pd.DataFrame([
            [(nodes_order[parent[k][j]] if parent[k][j] != -1 else "-")
             for j in range(n)]
            for k in range(n)
        ], index=[f"k={i}" for i in range(n)], columns=nodes_order)
        st.dataframe(dfp)

        st.write("---")
        st.subheader("Highlight shortest path")

        dest = st.selectbox("Destination", options=nodes_order)
        dest_idx = idx_map[dest]

        # Find source again
        source_idx = None
        for j in range(n):
            if dp[0][j] == 0:
                source_idx = j

        path_idx = trace_path(parent, source_idx, dest_idx, k_sel)
        if path_idx:
            path_names = [nodes_order[i] for i in path_idx]
            st.success(f"Path at k={k_sel}: {' -> '.join(path_names)}")
        else:
            st.error("No path yet")

        # highlight in graph
        edges_in_path = [(nodes_order[path_idx[i]], nodes_order[path_idx[i + 1]])
                         for i in range(len(path_idx) - 1)] if path_idx else []

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        pos2 = nx.spring_layout(st.session_state.G, seed=42)

        edge_colors = ['red' if e in edges_in_path or (e[1], e[0]) in edges_in_path else 'black'
                       for e in st.session_state.G.edges()]

        nx.draw(st.session_state.G, pos2, with_labels=True, node_color="lightblue",
                edge_color=edge_colors, node_size=600)
        nx.draw_networkx_edge_labels(
            st.session_state.G,
            pos2,
            edge_labels=nx.get_edge_attributes(st.session_state.G, 'weight')
        )
        ax2.set_axis_off()
        st.pyplot(fig2)
