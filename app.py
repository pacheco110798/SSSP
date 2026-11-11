import math
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from dijkstra import (adjacency, bellmanFord, bellmanFordSteps, dijkstraArray, dijkstraSteps, generateDirected,
                      generateGraph, johnson)

STATIC = Path(__file__).parent / "static"
MAX_N = 26  # una letra por nodo (A–Z)

app = FastAPI(title="Dijkstra, Bellman-Ford y Johnson: caminos más cortos")


def finite(xs):
    """JSON no acepta infinito: ∞ se manda como null."""
    return [None if x == math.inf else x for x in xs]


def cleanSteps(steps):
    """Pasos de dijkstraSteps listos para JSON (∞ → null)."""
    for s in steps:
        s["dist"] = finite(s["dist"])
        for r in s["relax"]:
            r["old"] = None if r["old"] == math.inf else r["old"]
    return steps


def clean(d):
    """Copia de un dict con ∞ cambiado por null en sus valores numéricos."""
    return {k: None if isinstance(x, float) and x == math.inf else x for k, x in d.items()}


@app.get("/")
def index():
    # Sin caché, para que el navegador siempre cargue la versión actual de la página
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/run")
def apiRun(n: int = Query(5, ge=3, le=MAX_N), neighbors: int = Query(2, ge=1, le=4),
           seed: int = Query(0, ge=0), source: int = Query(0, ge=0)):
    """El grafo de (n, neighbors, seed) y Dijkstra con heap desde `source`, paso a paso."""
    source = min(source, n - 1)
    pos, edges = generateGraph(n, neighbors, seed)
    steps = cleanSteps(dijkstraSteps(adjacency(n, edges), source))
    return {
        "n": n,
        "neighbors": neighbors,
        "seed": seed,
        "source": source,
        "pos": pos,
        "edges": edges,
        "steps": steps,
        "stale": sum(s["type"] == "stale" for s in steps),
        "improved": sum(r["better"] for s in steps for r in s["relax"]),
    }


def bellmanCase(n, neighbors, seed, source, negatives):
    """Grafo dirigido, Bellman-Ford ronda por ronda y lo que diría Dijkstra en el mismo grafo."""
    pos, arcs = generateDirected(n, neighbors, seed, source, negatives)
    run = bellmanFordSteps(n, arcs, source)
    dijkstra = dijkstraArray(adjacency(n, arcs, directed=True), source)[0]
    final = run["rounds"][-1]["dist"]
    wrong = [] if run["negativeCycle"] else [v for v in range(n) if dijkstra[v] != final[v]]
    return pos, arcs, run, dijkstra, wrong


@app.get("/api/bellman")
def apiBellman(n: int = Query(5, ge=3, le=MAX_N), neighbors: int = Query(2, ge=1, le=4),
               seed: int = Query(0, ge=0), source: int = Query(0, ge=0), negatives: int = Query(1, ge=0, le=8),
               find: str = Query("none", pattern="^(none|fail|cycle)$")):
    """Bellman-Ford en un grafo dirigido con `negatives` arcos negativos.

    Con find = fail | cycle busca, desde `seed` en adelante, la primera semilla donde Dijkstra se
    equivoca o donde hay un ciclo negativo."""
    source = min(source, n - 1)
    negatives = max(negatives, 1) if find != "none" else negatives
    for s in range(seed, seed + (3000 if find != "none" else 1)):
        pos, arcs, run, dijkstra, wrong = bellmanCase(n, neighbors, s, source, negatives)
        if find == "none" or (find == "fail" and wrong) or (find == "cycle" and run["negativeCycle"]):
            break
    else:
        return {"found": False}
    for r in run["rounds"]:
        r["dist"] = finite(r["dist"])
        for m in r["improved"]:
            m["old"] = None if m["old"] == math.inf else m["old"]
        r["checks"] = [clean(c) for c in r["checks"]]
    return {
        "found": True,
        "n": n,
        "neighbors": neighbors,
        "seed": s,
        "source": source,
        "negatives": negatives,
        "pos": pos,
        "arcs": arcs,
        "rounds": run["rounds"],
        "negativeCycle": run["negativeCycle"],
        "dijkstra": finite(dijkstra),
        "wrong": wrong,
    }


def johnsonCase(n, neighbors, seed, source, negatives):
    """Johnson en el grafo dirigido y, para comparar, Dijkstra directo (sin reponderar) desde cada nodo."""
    pos, arcs = generateDirected(n, neighbors, seed, source, negatives)
    res = johnson(n, arcs)
    adj = adjacency(n, arcs, directed=True)
    naive = [dijkstraArray(adj, s)[0] for s in range(n)]
    wrong = 0 if res["h"] is None else sum(naive[s][v] != res["dist"][s][v] for s in range(n) for v in range(n))
    return pos, arcs, res, naive, wrong


@app.get("/api/johnson")
def apiJohnson(n: int = Query(5, ge=3, le=MAX_N), neighbors: int = Query(2, ge=1, le=4),
               seed: int = Query(0, ge=0), source: int = Query(0, ge=0), negatives: int = Query(1, ge=0, le=8),
               find: str = Query("none", pattern="^(none|fail|cycle)$")):
    """Johnson (todos los pares) en el mismo grafo dirigido que /api/bellman.

    Con find = fail | cycle busca la primera semilla desde `seed` donde Dijkstra directo se
    equivoca en algún par o donde hay un ciclo negativo."""
    source = min(source, n - 1)
    negatives = max(negatives, 1) if find != "none" else negatives
    for s in range(seed, seed + (3000 if find != "none" else 1)):
        pos, arcs, res, naive, wrong = johnsonCase(n, neighbors, s, source, negatives)
        if find == "none" or (find == "fail" and wrong) or (find == "cycle" and res["negativeCycle"]):
            break
    else:
        return {"found": False}
    out = {"found": True, "n": n, "neighbors": neighbors, "seed": s, "source": source, "negatives": negatives,
           "pos": pos, "arcs": arcs, "h": res["h"], "bfRounds": res["bfRounds"],
           "negativeCycle": res["negativeCycle"], "naive": [finite(r) for r in naive], "wrongPairs": wrong,
           "hRounds": [{"h": finite(r["h"]),
                        # u = None: el arco viene de q
                        "improved": [clean({**m, "u": None if m["u"] == n else m["u"]}) for m in r["improved"]],
                        "checks": [clean({**c, "u": None if c["u"] == n else c["u"]}) for c in r["checks"]]}
                       for r in res["hRounds"]]}
    if res["h"] is not None:
        out.update({
            "reweighted": res["reweighted"],
            "dprime": [finite(r) for r in res["dprime"]],
            "dist": [finite(r) for r in res["dist"]],
            "prev": res["prev"],
            "dSteps": [cleanSteps(st) for st in res["dSteps"]],
            # Comprobación: Bellman-Ford desde cada nodo con los pesos originales
            "verified": all(bellmanFord(n, arcs, s, directed=True) == res["dist"][s] for s in range(n)),
        })
    return out
