import heapq
import math
import random


# ---------- grafos ----------

def generateGraph(n=10, neighbors=2, seed=0, maxWeight=20):
    """n nodos al azar en el plano; cada uno se une con sus `neighbors` vecinos más cercanos.

    Regresa (pos, edges): pos[v] = (x, y) en [0, 1]² y edges = [(u, v, peso), ...] sin dirección.
    Si quedan varias componentes se unen por su par de nodos más cercano, así todo es alcanzable."""
    rng = random.Random(seed)
    minDist = 0.6 / math.sqrt(n)  # separación mínima para que el dibujo no se encime
    pos, tries = [], 0
    while len(pos) < n:
        p = (rng.random(), rng.random())
        tries += 1
        if tries > 20_000 or all(math.dist(p, q) >= minDist for q in pos):
            pos.append(p)

    pairs = set()
    for u in range(n):
        near = sorted((v for v in range(n) if v != u), key=lambda v: math.dist(pos[u], pos[v]))
        for v in near[:neighbors]:
            pairs.add((min(u, v), max(u, v)))

    # Unir componentes sueltas
    group = list(range(n))

    def root(v):
        while group[v] != v:
            v = group[v]
        return v

    for u, v in pairs:
        group[root(v)] = root(u)
    while len({root(v) for v in range(n)}) > 1:
        main = [v for v in range(n) if root(v) == root(0)]
        rest = [v for v in range(n) if root(v) != root(0)]
        u, v = min(((a, b) for a in main for b in rest), key=lambda e: math.dist(pos[e[0]], pos[e[1]]))
        pairs.add((min(u, v), max(u, v)))
        group[root(v)] = root(u)

    edges = [(u, v, rng.randint(1, maxWeight)) for u, v in sorted(pairs)]
    return pos, edges


def adjacency(n, edges, directed=False):
    """Lista de adyacencia: adj[u] = [(v, peso), ...]."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    for row in adj:
        row.sort()
    return adj


def randomGraph(n, density, seed=0, maxWeight=100):
    """Grafo dirigido para medir tiempos: una cadena 0 → 1 → … → n − 1 (todo alcanzable desde 0)
    más cada arista u → v con probabilidad `density`."""
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    for v in range(n - 1):
        adj[v].append((v + 1, rng.randint(1, maxWeight)))
    for u in range(n):
        row, rand = adj[u], rng.random
        for v in range(n):
            if u != v and rand() < density:
                row.append((v, rng.randint(1, maxWeight)))
    return adj


# ---------- Dijkstra ----------

def dijkstraArray(adj, source):
    """Versión con arreglo: en cada vuelta busca el nodo sin visitar con menor distancia, O(V²).

    Regresa (dist, prev, ops); ops cuenta nodos revisados al buscar el mínimo y aristas relajadas."""
    n = len(adj)
    dist, prev, visited = [math.inf] * n, [None] * n, [False] * n
    dist[source] = 0
    scanned = relaxed = 0
    for _ in range(n):
        u, best = -1, math.inf
        for v in range(n):  # búsqueda lineal del mínimo
            if not visited[v] and dist[v] < best:
                u, best = v, dist[v]
        scanned += n
        if u < 0:
            break  # lo que queda es inalcanzable
        visited[u] = True
        relaxed += len(adj[u])
        for v, w in adj[u]:
            if best + w < dist[v]:
                dist[v] = best + w
                prev[v] = u
    return dist, prev, {"scanned": scanned, "relaxed": relaxed}


def dijkstraHeap(adj, source):
    """Versión con montículo (heapq): saca el mínimo en O(log V), O((V + E) log V) en total.

    Regresa (dist, prev, ops); ops cuenta operaciones del heap, entradas viejas y aristas relajadas."""
    n = len(adj)
    dist, prev = [math.inf] * n, [None] * n
    dist[source] = 0
    heap = [(0, source)]
    pushes, pops, stale, relaxed = 1, 0, 0, 0
    while heap:
        d, u = heapq.heappop(heap)
        pops += 1
        if d > dist[u]:
            stale += 1  # entrada vieja: ya se encontró algo mejor para u
            continue
        relaxed += len(adj[u])
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
                pushes += 1
    return dist, prev, {"pushes": pushes, "pops": pops, "stale": stale, "relaxed": relaxed}


def dijkstraSteps(adj, source):
    """Dijkstra con heap guardando cada vez que se saca algo del heap, para verlo paso a paso.

    Cada paso guarda el nodo sacado, si era una entrada vieja, las aristas que relajó y el estado
    después: distancias, padres, visitados y el contenido del heap."""
    n = len(adj)
    dist, prev, visited = [math.inf] * n, [None] * n, [False] * n
    dist[source] = 0
    heap = [(0, source)]
    steps = []
    while heap:
        d, u = heapq.heappop(heap)
        relax = []
        if d > dist[u] or visited[u]:
            kind = "stale"
        else:
            kind = "settle"
            visited[u] = True
            for v, w in adj[u]:
                old = dist[v]
                better = d + w < old
                relax.append({"to": v, "w": w, "old": old, "new": d + w, "better": better, "done": visited[v]})
                if better:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))
        steps.append({"type": kind, "node": u, "d": d, "relax": relax, "dist": dist[:], "prev": prev[:],
                      "visited": visited[:], "heap": heap[:]})  # el arreglo del heap tal cual (árbol binario)
    return steps


def shortestPath(prev, target):
    """Sigue a los padres desde `target` hasta el origen."""
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    return path[::-1]


def generateDirected(n=10, neighbors=2, seed=0, source=0, negatives=0):
    """Versión dirigida de generateGraph para Bellman-Ford.

    Las aristas del árbol BFS desde `source` apuntan hacia afuera (así todo es alcanzable) y las
    demás toman una dirección al azar. `negatives` arcos al azar reciben un peso negativo; eso
    puede formar un ciclo negativo, que Bellman-Ford detecta."""
    pos, edges = generateGraph(n, neighbors, seed)
    rng = random.Random(seed * 1_000_003 + n)
    adj = adjacency(n, edges)
    parent, queue = {source: None}, [source]
    for u in queue:
        for v, _ in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    arcs = []
    for u, v, w in edges:
        if parent.get(v) == u or (parent.get(u) != v and rng.random() < 0.5):
            arcs.append([u, v, w])
        else:
            arcs.append([v, u, w])
    for i in rng.sample(range(len(arcs)), min(negatives, len(arcs))):
        arcs[i][2] = -rng.randint(5, 15)
    return pos, [tuple(a) for a in arcs]


def bellmanFordOps(adj, source, order=None):
    """Bellman-Ford sobre lista de adyacencia con corte temprano: rondas de relajar todas las
    aristas hasta que una ronda no cambia nada (a lo más V − 1), O(VE) en el peor caso.

    `order` es el orden en que se recorren los nodos en cada ronda. Regresa (dist, prev, ops)."""
    n = len(adj)
    order = range(n) if order is None else order
    dist, prev = [math.inf] * n, [None] * n
    dist[source] = 0
    rounds = relaxed = 0
    for _ in range(n - 1):
        rounds += 1
        changed = False
        for u in order:
            du = dist[u]
            if du == math.inf:
                continue
            relaxed += len(adj[u])
            for v, w in adj[u]:
                if du + w < dist[v]:
                    dist[v] = du + w
                    prev[v] = u
                    changed = True
        if not changed:
            break
    return dist, prev, {"rounds": rounds, "relaxed": relaxed}


def bellmanFordSteps(n, arcs, source):
    """Bellman-Ford ronda por ronda sobre la lista de arcos dirigidos (u, v, peso).

    Cada ronda guarda todas las revisiones de arcos (`checks`), las mejoras (en el orden en que
    ocurrieron) y las distancias y padres al final. Si tras n − 1 rondas una ronda más todavía mejora algo, hay un ciclo negativo
    alcanzable: se regresa uno de sus ciclos en `negativeCycle`."""
    dist, prev = [math.inf] * n, [None] * n
    dist[source] = 0
    rounds = []
    for _ in range(n - 1):
        improved, checks = [], []
        for u, v, w in arcs:
            better = dist[u] + w < dist[v]
            # Cada revisión: distancia de la salida, candidato y distancia que tenía la llegada
            checks.append({"u": u, "v": v, "w": w, "du": dist[u], "old": dist[v], "new": dist[u] + w, "better": better})
            if better:
                improved.append({"u": u, "v": v, "w": w, "old": dist[v], "new": dist[u] + w})
                dist[v] = dist[u] + w
                prev[v] = u
        rounds.append({"improved": improved, "checks": checks, "dist": dist[:], "prev": prev[:]})
        if not improved:
            return {"rounds": rounds, "negativeCycle": None}

    # Ronda n: si algo todavía mejora, hay un ciclo negativo
    for u, v, w in arcs:
        if dist[u] + w < dist[v]:
            prev[v] = u
            x = v
            for _ in range(n):  # retroceder n padres asegura caer dentro del ciclo
                x = prev[x]
            cycle, y = [x], prev[x]
            while y != x:
                cycle.append(y)
                y = prev[y]
            return {"rounds": rounds, "negativeCycle": cycle[::-1]}
    return {"rounds": rounds, "negativeCycle": None}


def johnson(n, arcs):
    """Johnson: caminos más cortos entre todos los pares con pesos negativos (sin ciclos negativos).

    1. Agrega un nodo q con arcos de peso 0 hacia todos.
    2. Bellman-Ford desde q da h(v) (≤ 0).
    3. Repondera w'(u, v) = w(u, v) + h(u) − h(v) ≥ 0: los caminos más cortos no cambian.
    4. Dijkstra desde cada nodo con w'.
    5. Corrige d(u, v) = d'(u, v) − h(u) + h(v).
    Costo O(VE + V(V + E) log V). Si hay ciclo negativo regresa h = None."""
    q = n
    bf = bellmanFordSteps(n + 1, list(arcs) + [(q, v, 0) for v in range(n)], q)
    # Rondas de Bellman-Ford desde q, solo con los n nodos originales (para verlo paso a paso)
    hRounds = [{"h": r["dist"][:n], "improved": r["improved"], "checks": r["checks"]} for r in bf["rounds"]]
    if bf["negativeCycle"]:
        return {"h": None, "bfRounds": len(bf["rounds"]), "hRounds": hRounds, "negativeCycle": bf["negativeCycle"]}
    h = bf["rounds"][-1]["dist"][:n]
    reweighted = [(u, v, w + h[u] - h[v]) for u, v, w in arcs]
    adj = adjacency(n, reweighted, directed=True)
    dprime, dist, prev = [], [], []
    for s in range(n):
        d, p, _ = dijkstraHeap(adj, s)
        dprime.append(d)
        prev.append(p)
        dist.append([d[v] - h[s] + h[v] if d[v] != math.inf else math.inf for v in range(n)])
    # Dijkstra paso a paso desde cada nodo con los pesos ajustados (para verlo como en la pestaña de Dijkstra)
    dSteps = [dijkstraSteps(adj, s) for s in range(n)]
    return {"h": h, "bfRounds": len(bf["rounds"]), "hRounds": hRounds, "negativeCycle": None, "reweighted": reweighted,
            "dprime": dprime, "dist": dist, "prev": prev, "dSteps": dSteps}


def bellmanFord(n, edges, source, directed=False):
    """Referencia para las pruebas: relaja todas las aristas n − 1 veces, O(VE)."""
    arcs = [(u, v, w) for u, v, w in edges] + ([] if directed else [(v, u, w) for u, v, w in edges])
    dist = [math.inf] * n
    dist[source] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in arcs:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    return dist


if __name__ == "__main__":
    pos, edges = generateGraph(8, 2, 0)
    adj = adjacency(8, edges)
    print("Aristas (u, v, peso):", edges)
    dist, prev, _ = dijkstraHeap(adj, 0)
    for v in range(8):
        print(f"0 → {v}: distancia {dist[v]:>3}, camino {' → '.join(map(str, shortestPath(prev, v)))}")
    print("Igual con arreglo:", dijkstraArray(adj, 0)[0] == dist)
