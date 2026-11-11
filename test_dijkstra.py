import math
import unittest

import random

from dijkstra import (adjacency, bellmanFord, bellmanFordOps, bellmanFordSteps, dijkstraArray, dijkstraHeap,
                      dijkstraSteps, generateDirected, generateGraph, johnson, randomGraph, shortestPath)


def graph(n, neighbors, seed):
    pos, edges = generateGraph(n, neighbors, seed)
    return edges, adjacency(n, edges)


class TestCorrectness(unittest.TestCase):
    """Distancias correctas"""

    def test_matches_bellman_ford(self):
        """Arreglo y heap dan lo mismo que Bellman-Ford"""
        for seed in range(40):
            n = 5 + seed % 20
            edges, adj = graph(n, 1 + seed % 3, seed)
            expected = bellmanFord(n, edges, 0)
            with self.subTest(seed=seed):
                self.assertEqual(dijkstraArray(adj, 0)[0], expected)
                self.assertEqual(dijkstraHeap(adj, 0)[0], expected)

    def test_any_source(self):
        """Funciona desde cualquier nodo de origen"""
        edges, adj = graph(12, 2, 7)
        for s in range(12):
            with self.subTest(source=s):
                self.assertEqual(dijkstraHeap(adj, s)[0], bellmanFord(12, edges, s))

    def test_directed_dense(self):
        """Arreglo y heap coinciden en grafos dirigidos densos"""
        for seed in range(5):
            adj = randomGraph(80, 0.3, seed)
            with self.subTest(seed=seed):
                self.assertEqual(dijkstraArray(adj, 0)[0], dijkstraHeap(adj, 0)[0])

    def test_unreachable(self):
        """Los nodos inalcanzables quedan en ∞ y sin padre"""
        adj = adjacency(4, [(0, 1, 3), (2, 3, 1)])
        for algo in (dijkstraArray, dijkstraHeap):
            dist, prev, _ = algo(adj, 0)
            with self.subTest(algo=algo.__name__):
                self.assertEqual(dist, [0, 3, math.inf, math.inf])
                self.assertEqual(prev, [None, 0, None, None])

    def test_known_example(self):
        """Ejemplo a mano: el camino barato no es el de menos aristas"""
        edges = [(0, 1, 10), (0, 2, 1), (2, 3, 1), (3, 1, 1)]
        adj = adjacency(4, edges, directed=True)
        dist, prev, _ = dijkstraHeap(adj, 0)
        self.assertEqual(dist, [0, 3, 1, 2])
        self.assertEqual(shortestPath(prev, 1), [0, 2, 3, 1])


class TestPaths(unittest.TestCase):
    """Caminos más cortos"""

    def test_path_cost_is_distance(self):
        """La suma de pesos del camino reconstruido es la distancia"""
        edges, adj = graph(15, 2, 3)
        weight = {}
        for u, v, w in edges:
            weight[u, v] = weight[v, u] = w
        dist, prev, _ = dijkstraHeap(adj, 0)
        for t in range(15):
            p = shortestPath(prev, t)
            with self.subTest(target=t):
                self.assertEqual(p[0], 0)
                self.assertEqual(p[-1], t)
                self.assertEqual(sum(weight[a, b] for a, b in zip(p, p[1:])), dist[t])

    def test_source_path(self):
        """El camino al origen es solo el origen"""
        _, adj = graph(6, 2, 1)
        self.assertEqual(shortestPath(dijkstraHeap(adj, 4)[1], 4), [4])


class TestSteps(unittest.TestCase):
    """Paso a paso"""

    def test_settles_every_node_once(self):
        """Cada nodo se fija exactamente una vez"""
        _, adj = graph(14, 2, 4)
        settled = [s["node"] for s in dijkstraSteps(adj, 0) if s["type"] == "settle"]
        self.assertEqual(sorted(settled), list(range(14)))

    def test_nondecreasing_order(self):
        """Los nodos se fijan en orden de distancia no decreciente"""
        for seed in range(20):
            _, adj = graph(16, 2, seed)
            ds = [s["d"] for s in dijkstraSteps(adj, 0) if s["type"] == "settle"]
            with self.subTest(seed=seed):
                self.assertEqual(ds, sorted(ds))

    def test_final_state_matches(self):
        """El último paso tiene las mismas distancias que dijkstraHeap"""
        _, adj = graph(18, 3, 9)
        steps = dijkstraSteps(adj, 0)
        self.assertEqual(steps[-1]["dist"], dijkstraHeap(adj, 0)[0])
        self.assertEqual(steps[-1]["heap"], [])

    def test_stale_entries(self):
        """Una entrada vieja no relaja nada y su distancia es mayor a la definitiva"""
        for seed in range(30):
            _, adj = graph(16, 3, seed)
            for s in dijkstraSteps(adj, 0):
                if s["type"] == "stale":
                    with self.subTest(seed=seed):
                        self.assertEqual(s["relax"], [])
                        self.assertGreater(s["d"], s["dist"][s["node"]])

    def test_heap_property(self):
        """El heap guardado en cada paso cumple que cada padre es menor o igual que sus hijos"""
        for seed in range(30):
            _, adj = graph(14, 3, seed)
            for s in dijkstraSteps(adj, 0):
                h = s["heap"]
                with self.subTest(seed=seed):
                    self.assertTrue(all(h[(i - 1) // 2] <= h[i] for i in range(1, len(h))))

    def test_stale_count_matches(self):
        """Las entradas viejas de los pasos coinciden con las que cuenta dijkstraHeap"""
        _, adj = graph(20, 3, 2)
        steps = dijkstraSteps(adj, 0)
        ops = dijkstraHeap(adj, 0)[2]
        self.assertEqual(sum(s["type"] == "stale" for s in steps), ops["stale"])
        self.assertEqual(len(steps), ops["pops"])


class TestOps(unittest.TestCase):
    """Conteo de operaciones"""

    def test_array_scans_v_squared(self):
        """El arreglo revisa V nodos en cada una de las V vueltas"""
        adj = randomGraph(50, 0.1, 0)
        self.assertEqual(dijkstraArray(adj, 0)[2]["scanned"], 50 * 50)

    def test_both_relax_every_edge_once(self):
        """Ambas versiones relajan cada arista alcanzable una sola vez"""
        adj = randomGraph(60, 0.2, 1)
        e = sum(len(row) for row in adj)
        self.assertEqual(dijkstraArray(adj, 0)[2]["relaxed"], e)
        self.assertEqual(dijkstraHeap(adj, 0)[2]["relaxed"], e)

    def test_pushes_equal_pops(self):
        """Todo lo que entra al heap sale"""
        adj = randomGraph(60, 0.2, 2)
        ops = dijkstraHeap(adj, 0)[2]
        self.assertEqual(ops["pushes"], ops["pops"])


class TestGraph(unittest.TestCase):
    """Generador de grafos"""

    def test_connected(self):
        """El grafo generado siempre es conexo"""
        for seed in range(30):
            edges, adj = graph(4 + seed, 1, seed)
            with self.subTest(seed=seed):
                self.assertNotIn(math.inf, dijkstraHeap(adj, 0)[0])

    def test_deterministic(self):
        """La misma semilla da el mismo grafo"""
        self.assertEqual(generateGraph(12, 2, 5), generateGraph(12, 2, 5))

    def test_positive_weights(self):
        """Todos los pesos son positivos y no hay lazos"""
        _, edges = generateGraph(20, 3, 1)
        self.assertTrue(all(w > 0 and u != v for u, v, w in edges))


class TestBellmanFord(unittest.TestCase):
    """Bellman-Ford"""

    def test_same_as_dijkstra_without_negatives(self):
        """Sin pesos negativos da lo mismo que Dijkstra"""
        for seed in range(20):
            adj = randomGraph(60, 0.1, seed)
            order = list(range(60))
            random.Random(seed).shuffle(order)
            with self.subTest(seed=seed):
                self.assertEqual(bellmanFordOps(adj, 0, order)[0], dijkstraHeap(adj, 0)[0])

    def test_steps_match_reference(self):
        """Ronda por ronda llega a las mismas distancias que la versión de referencia"""
        for seed in range(60):
            n = 6 + seed % 15
            _, arcs = generateDirected(n, 2, seed, 0, seed % 4)
            run = bellmanFordSteps(n, arcs, 0)
            if run["negativeCycle"]:
                continue
            with self.subTest(seed=seed):
                self.assertEqual(run["rounds"][-1]["dist"], bellmanFord(n, arcs, 0, directed=True))

    def test_at_most_n_minus_1_rounds(self):
        """Sin ciclo negativo termina en a lo más n − 1 rondas y la última no mejora nada"""
        for seed in range(60):
            _, arcs = generateDirected(12, 2, seed, 0, 2)
            run = bellmanFordSteps(12, arcs, 0)
            if run["negativeCycle"]:
                continue
            with self.subTest(seed=seed):
                self.assertLessEqual(len(run["rounds"]), 11)
                if len(run["rounds"]) < 11:
                    self.assertEqual(run["rounds"][-1]["improved"], [])

    def test_negative_edges_beat_dijkstra(self):
        """Con pesos negativos Dijkstra puede equivocarse y Bellman-Ford no"""
        arcs = [(0, 1, 2), (0, 2, 5), (2, 1, -4), (1, 3, 1)]
        run = bellmanFordSteps(4, arcs, 0)
        self.assertIsNone(run["negativeCycle"])
        self.assertEqual(run["rounds"][-1]["dist"], [0, 1, 5, 2])
        self.assertNotEqual(dijkstraArray(adjacency(4, arcs, directed=True), 0)[0], [0, 1, 5, 2])

    def test_checks_every_arc_each_round(self):
        """Cada ronda revisa todos los arcos en orden y sus mejoras son las revisiones exitosas"""
        for seed in range(30):
            _, arcs = generateDirected(8, 2, seed, 0, 2)
            for r in bellmanFordSteps(8, arcs, 0)["rounds"]:
                with self.subTest(seed=seed):
                    self.assertEqual([(c["u"], c["v"], c["w"]) for c in r["checks"]], list(arcs))
                    self.assertEqual([(c["u"], c["v"], c["new"]) for c in r["checks"] if c["better"]],
                                     [(m["u"], m["v"], m["new"]) for m in r["improved"]])

    def test_detects_negative_cycle(self):
        """Detecta un ciclo negativo y el ciclo que regresa suma menos de 0"""
        arcs = [(0, 1, 1), (1, 2, 2), (2, 3, -6), (3, 1, 1), (3, 4, 2)]
        cycle = bellmanFordSteps(5, arcs, 0)["negativeCycle"]
        self.assertEqual(sorted(cycle), [1, 2, 3])
        w = {(u, v): x for u, v, x in arcs}
        self.assertLess(sum(w[a, b] for a, b in zip(cycle, cycle[1:] + cycle[:1])), 0)

    def test_generated_cycles_are_negative(self):
        """Cada ciclo reportado en grafos generados es un ciclo real de peso negativo"""
        found = 0
        for seed in range(300):
            _, arcs = generateDirected(10, 2, seed, 0, 4)
            cycle = bellmanFordSteps(10, arcs, 0)["negativeCycle"]
            if not cycle:
                continue
            found += 1
            w = {(u, v): x for u, v, x in arcs}
            with self.subTest(seed=seed):
                self.assertLess(sum(w[a, b] for a, b in zip(cycle, cycle[1:] + cycle[:1])), 0)
        self.assertGreater(found, 0)

    def test_directed_reachable(self):
        """En el grafo dirigido generado todo es alcanzable desde el origen"""
        for seed in range(30):
            for source in (0, 3):
                _, arcs = generateDirected(10, 2, seed, source, 0)
                with self.subTest(seed=seed, source=source):
                    self.assertNotIn(math.inf, bellmanFord(10, arcs, source, directed=True))


class TestJohnson(unittest.TestCase):
    """Johnson"""

    def cases(self, count=60, negatives=3):
        for seed in range(count):
            n = 5 + seed % 15
            _, arcs = generateDirected(n, 2, seed, 0, negatives)
            yield seed, n, arcs, johnson(n, arcs)

    def test_matches_bellman_ford_all_pairs(self):
        """Todas las distancias coinciden con Bellman-Ford desde cada nodo"""
        for seed, n, arcs, res in self.cases():
            if res["h"] is None:
                continue
            for s in range(n):
                with self.subTest(seed=seed, source=s):
                    self.assertEqual(res["dist"][s], bellmanFord(n, arcs, s, directed=True))

    def test_reweighted_nonnegative(self):
        """Los pesos reponderados w' = w + h(u) − h(v) nunca son negativos"""
        for seed, n, arcs, res in self.cases():
            if res["h"] is not None:
                with self.subTest(seed=seed):
                    self.assertTrue(all(w >= 0 for _, _, w in res["reweighted"]))

    def test_h_nonpositive(self):
        """h(v) ≤ 0 porque el arco q → v pesa 0"""
        for seed, n, arcs, res in self.cases():
            if res["h"] is not None:
                with self.subTest(seed=seed):
                    self.assertTrue(all(x <= 0 for x in res["h"]))

    def test_paths_preserved(self):
        """Reponderar no cambia los caminos: el costo original del camino es la distancia"""
        for seed, n, arcs, res in self.cases(30):
            if res["h"] is None:
                continue
            w = {(u, v): x for u, v, x in arcs}
            for s in range(n):
                for t in range(n):
                    p = shortestPath(res["prev"][s], t)
                    if res["dist"][s][t] == math.inf:
                        continue
                    with self.subTest(seed=seed, s=s, t=t):
                        self.assertEqual(sum(w[a, b] for a, b in zip(p, p[1:])), res["dist"][s][t])

    def test_h_rounds_end_in_h(self):
        """La última ronda de Bellman-Ford desde q termina justo en h"""
        for seed, n, arcs, res in self.cases(30):
            if res["h"] is not None:
                with self.subTest(seed=seed):
                    self.assertEqual(res["hRounds"][-1]["h"], res["h"])
                    self.assertEqual(res["hRounds"][-1]["improved"], [])

    def test_dijkstra_steps_match(self):
        """El Dijkstra paso a paso desde cada nodo termina en los costos ajustados d′"""
        for seed, n, arcs, res in self.cases(20):
            if res["h"] is None:
                continue
            for s in range(n):
                with self.subTest(seed=seed, source=s):
                    self.assertEqual(res["dSteps"][s][-1]["dist"], res["dprime"][s])

    def test_negative_cycle(self):
        """Con un ciclo negativo no hay h y se reporta el ciclo"""
        arcs = [(0, 1, 1), (1, 2, 2), (2, 3, -6), (3, 1, 1)]
        res = johnson(4, arcs)
        self.assertIsNone(res["h"])
        self.assertEqual(sorted(res["negativeCycle"]), [1, 2, 3])

    def test_no_negatives_equals_dijkstra(self):
        """Sin pesos negativos h = 0 y da lo mismo que Dijkstra desde cada nodo"""
        _, arcs = generateDirected(12, 2, 4, 0, 0)
        res = johnson(12, arcs)
        self.assertEqual(res["h"], [0] * 12)
        adj = adjacency(12, arcs, directed=True)
        self.assertEqual(res["dist"], [dijkstraHeap(adj, s)[0] for s in range(12)])


if __name__ == "__main__":
    unittest.main()
