import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import warnings
import os

warnings.filterwarnings('ignore')


def load_matrix(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    labels = lines[0].split()
    matrix = np.array([list(map(float, l.split())) for l in lines[1:]])
    return matrix, labels


# Комбинирование матриц
def combine_matrices(A_co, A_cit, threshold=30.0):
    n = A_co.shape[0]
    A_weighted = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            co      = A_co[i, j]
            cit_ij  = A_cit[i, j]
            cit_ji  = A_cit[j, i]
            mutual  = min(cit_ij, cit_ji)
            oneway  = max(cit_ij - mutual, 0)
            A_weighted[i, j] = 4.5 * co + 2.5 * mutual + 0.6 * oneway
    A_bin = (A_weighted >= threshold).astype(int)
    return A_bin


#  Транзитивное замыкание
def floyd_warshall_tc(A):
    n = A.shape[0]
    T = (A > 0).astype(np.uint8)
    np.fill_diagonal(T, 1)
    steps = int(np.ceil(np.log2(n + 1)))
    for _ in range(steps):
        T_new = np.clip(T @ T, 0, 1).astype(np.uint8)
        if np.array_equal(T_new, T):
            break
        T = T_new
    return T


#  АЛГОРИТМ ЛЕЖНИНОЙ–КАЛИНИНОЙ
def lk_clustering(A_bin, labels, max_nodes=600):
    n = A_bin.shape[0]
    used_labels = list(labels)

    if n > max_nodes:
        print(f"  [LK] Граф {n} узлов — обрезаем до {max_nodes} (по степени)")
        degrees = A_bin.sum(axis=1)
        idx = np.argsort(-degrees)[:max_nodes]
        A_bin = A_bin[np.ix_(idx, idx)]
        used_labels = [labels[i] for i in idx]

    T = floyd_warshall_tc(A_bin)
    row_sums = T.sum(axis=1)
    order = np.argsort(-row_sums)
    T_sorted = T[np.ix_(order, order)]

    clusters = []
    current = [used_labels[order[0]]]

    for i in range(1, len(order)):
        if T_sorted[i, i-1] == 0 and T_sorted[i-1, i] == 0:
            clusters.append(current)
            current = [used_labels[order[i]]]
        else:
            current.append(used_labels[order[i]])
    clusters.append(current)

    if np.all(T == 1):
        conn = "Сильно связный"
    elif np.all(T == T.T):
        conn = "Слабо связный"
    else:
        conn = "Несвязный"

    return clusters, conn


#  AGNES
def agglomerative_clustering(A_bin, labels, n_clusters):
    A_sym = np.maximum(A_bin, A_bin.T)
    dist  = 1 - A_sym
    np.clip(dist, 0, 1, out=dist)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    pred = model.fit_predict(dist)

    clusters = {}
    for l, c in zip(labels, pred):
        clusters.setdefault(c, []).append(l)
    return list(clusters.values())


#  DIANA
def diana_clustering(A_bin, labels, max_nodes=250):
    n = A_bin.shape[0]
    used_labels = list(labels)

    if n > max_nodes:
        print(f"  [DIANA] Граф {n} узлов — обрезаем до {max_nodes}")
        degrees = A_bin.sum(axis=1)
        idx = np.argsort(-degrees)[:max_nodes]
        A_bin = A_bin[np.ix_(idx, idx)]
        used_labels = [labels[i] for i in idx]

    A_sym = np.maximum(A_bin, A_bin.T)
    dist  = 1 - A_sym

    def split(cluster):
        if len(cluster) <= 1:
            return [cluster]
        max_d, seed = -1, cluster[0]
        for i in cluster:
            for j in cluster:
                if dist[i, j] > max_d:
                    max_d = dist[i, j]
                    seed  = i
        group1 = [seed]
        group2 = [x for x in cluster if x != seed]
        changed = True
        while changed:
            changed = False
            for x in group2[:]:
                d1 = np.mean([dist[x, y] for y in group1])
                d2 = (np.mean([dist[x, y] for y in group2 if y != x])
                      if len(group2) > 1 else 0)
                if d1 < d2:
                    group1.append(x)
                    group2.remove(x)
                    changed = True
        return [group1, group2]

    stack = [list(range(len(used_labels)))]
    final = []
    while stack:
        cl = stack.pop()
        if len(cl) <= 2:
            final.append(cl)
            continue
        res = split(cl)
        if len(res) == 1:
            final.append(cl)
        else:
            stack.extend(res)

    return [[used_labels[i] for i in cl] for cl in final]


#  LOUVAIN
def louvain_clustering(A_bin, labels):
    A_sym = np.maximum(A_bin, A_bin.T)
    G = nx.from_numpy_array(A_sym)
    mapping = {i: labels[i] for i in range(len(labels))}
    G = nx.relabel_nodes(G, mapping)

    try:
        import community as community_louvain
        part = community_louvain.best_partition(G)
    except Exception:
        comms = list(nx.community.greedy_modularity_communities(G))
        part  = {}
        for i, c in enumerate(comms):
            for node in c:
                part[node] = i

    clusters = {}
    for node, cl in part.items():
        clusters.setdefault(cl, []).append(node)
    return list(clusters.values())


# Метрики
def compute_metrics(clusters):
    sizes = [len(c) for c in clusters]
    return {
        "кластеров": len(clusters),
        "макс":      max(sizes),
        "мин":       min(sizes),
        "средний":   round(np.mean(sizes), 2)
    }



def print_clusters(name, clusters):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    for i, cl in enumerate(sorted(clusters, key=lambda x: -len(x)), 1):
        print(f"\nКластер {i} (размер {len(cl)}):")
        if len(cl) > 30:
            print(", ".join(cl[:30]) + " ...")
        else:
            print(", ".join(cl))


from itertools import cycle

def visualize_simple_groups(A_bin, labels, clusters, title="Простые кластеры",
                             save_path=None):
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

    node_to_idx = {labels[i]: i for i in range(len(labels))}
    num_labels  = {node: node_to_idx[node] for node in G.nodes()
                   if node in node_to_idx}

    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42)

    plt.figure(figsize=(15, 11))
    nx.draw(G, pos,
            node_color=[node_color_dict[n] for n in G.nodes()],
            node_size=700, alpha=0.9, with_labels=False,  
            edge_color='gray', width=1)
    nx.draw_networkx_labels(G, pos, labels=num_labels,    
                            font_size=8, font_weight='bold')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
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


def visualize_influence_hierarchy_strict(A_bin, labels, clusters,
                                         connectivity_type,
                                         node_colors_from_simple,
                                         save_path=None):
    """
    2D — кластеры + все связи между ними
    """
    large_clusters = [c for c in clusters if len(c) > 1]
    if not large_clusters:
        print("Нет крупных кластеров для 2D визуализации")
        return

    all_nodes   = [node for cluster in large_clusters for node in cluster]
    node_to_idx = {label: idx for idx, label in enumerate(labels)}

    num_labels = {node: node_to_idx[node] for node in all_nodes
                  if node in node_to_idx}

    G = nx.DiGraph()
    for node in all_nodes:
        G.add_node(node)

    for u in all_nodes:
        for v in all_nodes:
            if A_bin[node_to_idx[u], node_to_idx[v]] > 0:
                G.add_edge(u, v)

    pos   = {}
    level = 0
    for cluster in large_clusters:
        n_nodes = len(cluster)
        start_x = -(n_nodes - 1) * 2.0 / 2
        for idx, node in enumerate(cluster):
            pos[node] = (start_x + idx * 2.0, -level * 4.0)
        level += 1

    node_colors = [node_colors_from_simple.get(n, '#aaaaaa') for n in all_nodes]

    plt.figure(figsize=(19, 13))
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                           arrowsize=14, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=1050, alpha=0.95,
                           edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels=num_labels,    # номера
                            font_size=10, font_weight='bold')

    plt.title(f"Иерархическая структура (только крупные кластеры)\n"
              f"Все связи между ними\nТип связности: {connectivity_type}",
              fontsize=15)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()



if __name__ == "__main__":
    print("Загрузка матриц...")
    A_co,  labels = load_matrix("coauthorship_matrix.txt")
    A_cit, _      = load_matrix("citation_matrix.txt")

    n = A_co.shape[0]
    print(f"Авторов: {n}  ({n}×{n} матрица)")

    print("\nКомбинирование матриц...")
    A_bin = combine_matrices(A_co, A_cit)
    nnz   = int(A_bin.sum()) // 2
    print(f"Рёбер в графе: {nnz}  (плотность {nnz/(n*n):.5f})")


    print("\n[1] Лежнина–Калинина...")
    clusters_lk, conn = lk_clustering(A_bin, labels)

    print("\n[2] AGNES...")
    clusters_agg = agglomerative_clustering(A_bin, labels, len(clusters_lk))

    print("\n[3] Louvain...")
    clusters_lou = louvain_clustering(A_bin, labels)

    print("\n[4] DIANA...")
    clusters_diana = diana_clustering(A_bin, labels)

    # Результаты 
    print(f"\nТип связности: {conn}")

    print_clusters("Лежнина–Калинина", clusters_lk)
    print_clusters("AGNES",            clusters_agg)
    print_clusters("Louvain",          clusters_lou)
    print_clusters("DIANA",            clusters_diana)

    print("\nМетрики:")
    for name, cl in [("LK",     clusters_lk),
                     ("AGNES",  clusters_agg),
                     ("Louvain",clusters_lou),
                     ("DIANA",  clusters_diana)]:
        print(f"  {name:8s}: {compute_metrics(cl)}")

    # Простые графы с номерами
    color_map_lk = visualize_simple_groups(
        A_bin, labels, clusters_lk,
        title="Лежнина–Калинина",
        save_path='plots/lk.png'
    )
    visualize_simple_groups(
        A_bin, labels, clusters_agg,
        title="AGNES",
        save_path='plots/agnes.png'
    )
    visualize_simple_groups(
        A_bin, labels, clusters_lou,
        title="Louvain",
        save_path='plots/louvain.png'
    )
    visualize_simple_groups(
        A_bin, labels, clusters_diana,
        title="DIANA",
        save_path='plots/diana.png'
    )

    # Иерархия для каждого алгоритма
    for clusters, name in [
        (clusters_lk,    "lk"),
        (clusters_agg,   "agnes"),
        (clusters_lou,   "louvain"),
        (clusters_diana, "diana"),
    ]:
        colors = assign_unique_colors(clusters)
        visualize_influence_hierarchy_strict(
            A_bin, labels, clusters,
            connectivity_type=conn,
            node_colors_from_simple=colors,
            save_path=f'plots/hierarchy_{name}.png'
        )