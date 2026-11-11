# Caminos más cortos: Dijkstra, Bellman-Ford y Johnson

Andrea Georgina Rodriguez Pacheco

Los tres se basan en **relajar** aristas: si `dist[u] + w(u, v) < dist[v]`, se actualiza `dist[v]` y el padre de `v`.

| Algoritmo | Qué resuelve | Pesos negativos | Complejidad |
|---|---|---|---|
| Dijkstra (arreglo) | un origen | no | O(V²) |
| Dijkstra (heap) | un origen | no | O((V + E) log V) |
| Bellman-Ford | un origen | sí, y detecta ciclos negativos | O(VE) |
| Johnson | todos los pares | sí, y detecta ciclos negativos | O(VE + V(V + E) log V) |

- **Dijkstra** fija siempre el nodo no visitado más cercano y relaja sus aristas. Con un arco negativo puede fijar un nodo demasiado pronto y equivocarse.
- **Bellman-Ford** relaja todas las aristas en rondas; n − 1 rondas bastan. Si la ronda n todavía mejora algo, hay un ciclo negativo.
- **Johnson** agrega un nodo q con arcos de peso 0, corre Bellman-Ford desde q para obtener h(v), repondera `w'(u, v) = w(u, v) + h(u) − h(v) ≥ 0`, corre Dijkstra desde cada nodo y corrige `d(u, v) = d'(u, v) − h(u) + h(v)`.

La app web tiene las pestañas:

- **Dijkstra**: paso a paso con heap en un grafo no dirigido (heap, relajaciones, tabla de distancias; clic en un nodo para ver su camino).
- **Bellman-Ford**: peso por peso en un grafo dirigido con pesos negativos, comparado con Dijkstra; botones para buscar un caso donde Dijkstra falla o uno con ciclo negativo.
- **Johnson**: las 6 fases (0 a 5) sobre el mismo grafo dirigido, con Bellman-Ford desde q peso por peso, Dijkstra desde cada ciudad paso a paso y la matriz de distancias entre todos los pares.

Cada pestaña se recorre con **◀ Anterior**, **Siguiente ▶** y una barra deslizante.

## Instalar

```bash
pip install -r requirements.txt
```

## Usar

| Qué | Comando |
|---|---|
| App web (http://127.0.0.1:8000) | `python3 -m uvicorn app:app --reload` |
| Ejemplo en consola | `python dijkstra.py` |
| Pruebas | `python -m unittest test_dijkstra` |
| Notebook | abrir `dijkstra.ipynb` |
