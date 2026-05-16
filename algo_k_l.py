import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import random
from itertools import cycle

# Загрузка данных
def load_data_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
        if not lines:
            raise ValueError(f"Файл {filename} пустой!")
        labels = lines[0].split()
        matrix = [list(map(float, line.split())) for line in lines[1:]]
        mat = np.array(matrix)
        
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Матрица не квадратная: {mat.shape}")
        if len(labels) != mat.shape[0]:
            raise ValueError(f"Меток {len(labels)}, размер матрицы {mat.shape[0]}")
        
        return mat, labels


# Комбинация матриц
def combine_matrices(A_co, A_cit, threshold=30.0, preview_size=100):
    n = A_co.shape[0]
    A_weighted = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            co = A_co[i, j]
            cit_ij = A_cit[i, j]
            cit_ji = A_cit[j, i]
            mutual = min(cit_ij, cit_ji)
            oneway_ij = max(cit_ij - mutual, 0)
            A_weighted[i, j] = 4.5 * co + 2.5 * mutual + 0.6 * oneway_ij
    
    A_bin = (A_weighted >= threshold).astype(int)
    
    nonzero = A_weighted[A_weighted > 0]
    mean_val = nonzero.mean() if nonzero.size > 0 else 0.0
    
    print(f"\nВзвешенная матрица: max = {A_weighted.max():.2f} | "
          f"ср. ненулевых = {mean_val:.2f} | плотность = {A_bin.mean():.4f}")
    print(f"Бинарная матрица (>= {threshold}): рёбер = {A_bin.sum()}")
    
    show = min(preview_size, n)
    print("\n" + "="*80)
    print("ФРАГМЕНТ ВЗВЕШЕННОЙ МАТРИЦЫ")
    print("="*80)
    print("Метки:", " ".join(f"{lbl:>8}" for lbl in labels[:show]))
    for i in range(show):
        row_str = f"{labels[i]:<8} "
        for j in range(show):
            val = A_weighted[i, j]
            row_str += f"{val:8.2f}" if val > 0 else "      - "
        print(row_str)
    
    return A_bin, A_weighted


# Кластеризация (компоненты связности)
def simple_clusterization(A_bin, labels):
    A_sym = np.maximum(A_bin, A_bin.T)
    G_undir = nx.from_numpy_array(A_sym)
    components = sorted(nx.connected_components(G_undir), key=len, reverse=True)
    return [[labels[i] for i in comp] for comp in components]


# Алгоритм Калининой-Лежниной
def floyd_warshall_transitive_closure(A):
    n = A.shape[0]
    T = A.astype(int).copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                T[i,j] |= (T[i,k] & T[k,j])
    return T


def canonical_form_by_sums(T, A):
    row_sums = T.sum(axis=1)
    order = np.argsort(-row_sums)
    P = np.eye(len(T), dtype=int)[order]
    return P @ A @ P.T, P @ T @ P.T, order


def check_connectivity_type(T):
    if np.all(T == 1): return "Сильно связный (неразложимая матрица)"
    if np.all(T == T.T):
        return "Слабо связный" if np.any(T == 0) else "Односторонне связный"
    return "Несвязный граф"


def extract_clusters_from_T(T_tilde, labels, order):
    n = T_tilde.shape[0]
    clusters = []
    cluster = [labels[order[0]]]
    for i in range(1, n):
        if T_tilde[i, i-1] == 0 and T_tilde[i-1, i] == 0:
            clusters.append(cluster)
            cluster = [labels[order[i]]]
        else:
            cluster.append(labels[order[i]])
    clusters.append(cluster)
    return clusters


def hierarchical_matrix_clustering(A_bin, labels):
    T = floyd_warshall_transitive_closure(A_bin)
    conn_type = check_connectivity_type(T)
    
    if np.all(T == 1):
        return A_bin, [labels], T, conn_type   # один большой кластер
    
    A_tilde, T_tilde, order = canonical_form_by_sums(T, A_bin)
    clusters = extract_clusters_from_T(T_tilde, labels, order)
    return A_tilde, clusters, T_tilde, conn_type


# Визуализация

def visualize_simple_groups(labels, clusters, title="Простые кластеры"):
    color_cycle = cycle(plt.cm.tab20.colors)
    node_color_dict = {}
    for cluster in clusters:
        col = next(color_cycle)
        for node in cluster:
            node_color_dict[node] = col

    A_sym = np.maximum(A_bin, A_bin.T)
    G = nx.from_numpy_array(A_sym)
    mapping = {i: labels[i] for i in range(len(labels))}
    G = nx.relabel_nodes(G, mapping)

    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42)

    plt.figure(figsize=(15, 11))
    nx.draw(G, pos, 
            node_color=[node_color_dict[n] for n in G.nodes()],
            node_size=700, alpha=0.9, with_labels=True, font_size=8,
            edge_color='gray', width=1)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return node_color_dict


def assign_unique_colors(clusters):
    """
    Цвета только для крупных кластеров
    """
    color_cycle = cycle(plt.cm.tab20.colors)
    node_color_dict = {}
    for cluster in clusters:
        if len(cluster) > 1:         
            col = next(color_cycle)
            for node in cluster:
                node_color_dict[node] = col
    return node_color_dict


def visualize_influence_hierarchy_strict(A_tilde, clusters, connectivity_type, node_colors_from_simple):
    """
    2D — крупные кластеры + все связи между ними
    """
    large_clusters = [c for c in clusters if len(c) > 1]
    if not large_clusters:
        print("Нет крупных кластеров для 2D визуализации")
        return

    all_nodes = [node for cluster in large_clusters for node in cluster]

    G = nx.DiGraph()
    for node in all_nodes:
        G.add_node(node)

    node_to_idx = {label: idx for idx, label in enumerate(labels)}
    for u in all_nodes:
        for v in all_nodes:
            if A_bin[node_to_idx[u], node_to_idx[v]] > 0:
                G.add_edge(u, v)

    pos = {}
    level = 0
    for cluster in large_clusters:
        n_nodes = len(cluster)
        start_x = -(n_nodes - 1) * 2.0 / 2
        for idx, node in enumerate(cluster):
            x = start_x + idx * 2.0
            y = -level * 4.0
            pos[node] = (x, y)
        level += 1

    node_colors = [node_colors_from_simple.get(n, '#aaaaaa') for n in all_nodes]

    plt.figure(figsize=(19, 13))
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                           arrowsize=14, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=1050, alpha=0.95, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title(f"Иерархическая структура (только крупные кластеры)\n"
              f"Все связи между ними\nТип связности: {connectivity_type}", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_influence_hierarchy_3d(A_tilde, clusters, connectivity_type, node_colors_from_simple):
    """
    3D — все кластеры + все связи из A_bin
    """
    all_nodes = [node for cluster in clusters for node in cluster]

    G = nx.DiGraph()
    for node in all_nodes:
        G.add_node(node)

    node_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for i, u in enumerate(all_nodes):
        for j, v in enumerate(all_nodes):
            if i != j and A_bin[node_to_idx[u], node_to_idx[v]] > 0:
                G.add_edge(u, v)

    pos_3d = {}
    level = 0
    for cluster in clusters:
        n = len(cluster)
        base_x = - (n - 1) * 0.9 / 2 if n > 1 else 0
        for idx, node in enumerate(cluster):
            x = base_x + idx * 0.9 + random.uniform(-0.25, 0.25)
            y = -level * 3.6
            z = random.uniform(-1.6, 1.6)
            pos_3d[node] = (x, y, z)
        level += 1

    node_colors = []
    for n in all_nodes:
        node_colors.append(node_colors_from_simple.get(n, '#777777'))

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        if u in pos_3d and v in pos_3d:   
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(90, 90, 90, 0.75)', width=2.0),
        hoverinfo='none',
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[pos_3d[n][0] for n in all_nodes],
        y=[pos_3d[n][1] for n in all_nodes],
        z=[pos_3d[n][2] for n in all_nodes],
        mode='markers+text',
        marker=dict(
            size=8,
            color=node_colors,
            opacity=0.92,
            line=dict(width=1.2, color='black')
        ),
        text=all_nodes,
        textposition="middle center",
        textfont=dict(size=10),
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"3D Полная иерархическая структура<br>"
              f"(все кластеры включая одиночные + все связи)<br>"
              f"Тип связности: {connectivity_type}",
        scene=dict(
            xaxis_title='Разброс внутри уровня',
            yaxis_title='Уровень влияния (сверху ↓ вниз)',
            zaxis_title='Глубина',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridcolor='lightgray')
        ),
        width=1250,
        height=880,
        margin=dict(l=0, r=0, b=0, t=100)
    )

    fig.write_html("3d_full_hierarchy_all_clusters.html")
    print("3D визуализация успешно сохранена в файл: 3d_full_hierarchy_all_clusters.html")

# Запуск
coauth_file = 'coauthorship_matrix.txt'
cite_file   = 'citation_matrix.txt'
threshold   = 30.0
preview     = 100

A_co, labels_co = load_data_from_txt(coauth_file)
A_cit, labels_cit = load_data_from_txt(cite_file)

labels = labels_co
A_bin, A_weighted = combine_matrices(A_co, A_cit, threshold=threshold, preview_size=preview)

simple_clusters = simple_clusterization(A_bin, labels)
print("\nКластеры (простые):")
for i, c in enumerate(simple_clusters, 1):
    print(f"C{i} ({len(c)} авторов): {c}")
node_colors_simple = visualize_simple_groups(labels, simple_clusters, "Простые кластеры")

canonical_A, hier_clusters, T, connectivity_type = hierarchical_matrix_clustering(A_bin, labels)

print("\nКластеры (включая одиночные):")
for i, c in enumerate(hier_clusters, 1):
    print(f"C{i} ({len(c)} авторов): {c}")

node_colors_large = assign_unique_colors(hier_clusters)

visualize_influence_hierarchy_strict(canonical_A, hier_clusters, connectivity_type, node_colors_large)
visualize_influence_hierarchy_3d(canonical_A, hier_clusters, connectivity_type, node_colors_large)