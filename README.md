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

- **Paso a paso**: Dijkstra con heap en un grafo no dirigido (heap, relajaciones, tabla de distancias; clic en un nodo para ver su camino).
- **Bellman-Ford**: ronda por ronda en un grafo dirigido con arcos negativos, comparado con Dijkstra; botones para buscar un caso donde Dijkstra falla o uno con ciclo negativo.
- **Johnson**: las 5 fases sobre el mismo grafo dirigido y la matriz de distancias entre todos los pares (clic en una celda para ver el camino).
- **Arreglo vs heap**: tiempos y operaciones de Dijkstra (arreglo y heap) y Bellman-Ford en grafos dispersos y densos.
- **Pruebas**: las pruebas unitarias.

Cada reproductor tiene control de velocidad, y hay un interruptor de **modo noche** (ambos se recuerdan en el navegador).

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
