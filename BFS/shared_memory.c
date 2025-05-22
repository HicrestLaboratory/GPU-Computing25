#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

#define INF INT_MAX

typedef struct {
    int nnodes;
    int nedges;
    int *row_ptr;
    int *col_idx;
} GraphCSR;

void free_graph(GraphCSR* g) {
    free(g->row_ptr);
    free(g->col_idx);
}

void bfs_csr_parallel(GraphCSR* graph, int start, int* dist) {
    int n = graph->nnodes;
    int* row_ptr = graph->row_ptr;
    int* col_idx = graph->col_idx;

    for (int i = 0; i < n; i++)
        dist[i] = INF;

    dist[start] = 0;

    // Allocate frontiers
    int* frontier = (int*)malloc(n * sizeof(int));
    int* next_frontier = (int*)malloc(n * sizeof(int));

    int frontier_size = 1;
    frontier[0] = start;

    while (frontier_size > 0) {
        int next_frontier_size = 0;

        #pragma omp parallel
        {
            int* local_frontier = (int*)malloc(n * sizeof(int));
            int local_size = 0;

            #pragma omp for nowait schedule(dynamic)
            for (int i = 0; i < frontier_size; i++) {
                int u = frontier[i];
                for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
                    int v = col_idx[j];
                    if (__sync_bool_compare_and_swap(&dist[v], INF, dist[u] + 1)) {
                        local_frontier[local_size++] = v;
                    }
                }
            }

            // Merge local frontiers safely
            #pragma omp critical
            {
                for (int i = 0; i < local_size; i++) {
                    next_frontier[next_frontier_size++] = local_frontier[i];
                }
            }

            free(local_frontier);
        }

        // Swap frontiers
        int* temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        frontier_size = next_frontier_size;
    }

    free(frontier);
    free(next_frontier);
}

GraphCSR* defaultgraph_1(void) {
    GraphCSR *graph = (GraphCSR*)malloc(sizeof(GraphCSR));
    graph->nnodes = 9;
    graph->nedges = 22;

    graph->row_ptr = (int*)malloc(sizeof(int) * (graph->nnodes + 1));
    graph->col_idx = (int*)malloc(sizeof(int) * graph->nedges);

    int tmp_row[10] = {0,2,6,9,11,14,16,18,21,22};
    int tmp_col[22] = {1,2,0,2,3,4,0,1,5,1,6,1,6,7,2,7,3,4,4,5,8,7};

    for (int i = 0; i < 10; i++) graph->row_ptr[i] = tmp_row[i];
    for (int i = 0; i < 22; i++) graph->col_idx[i] = tmp_col[i];

    for (int i = 0; i < graph->nnodes; i++) {
        fprintf(stdout, "Node %d: ", i);
        for (int j = graph->row_ptr[i]; j < graph->row_ptr[i + 1]; j++) {
            fprintf(stdout, "%d ", graph->col_idx[j]);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
    }

    return graph;
}

int main() {
    GraphCSR *graph = defaultgraph_1();
    int* dist = (int*)malloc(graph->nnodes * sizeof(int));

    int start;
    printf("Enter start node: ");
    scanf("%d", &start);

    bfs_csr_parallel(graph, start, dist);

    printf("Distances from node %d:\n", start);
    for (int i = 0; i < graph->nnodes; i++) {
        if (dist[i] == INF)
            printf("Node %d: unreachable\n", i);
        else
            printf("Node %d: %d\n", i, dist[i]);
    }

    free_graph(graph);
    free(graph);
    free(dist);
    return 0;
}

