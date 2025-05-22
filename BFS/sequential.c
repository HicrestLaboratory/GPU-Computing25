#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <glib.h>

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

void bfs_csr(GraphCSR* graph, int start, int* dist) {
    int n = graph->nnodes, *row_ptr = graph->row_ptr, *col_idx = graph->col_idx;

    for (int i = 0; i < n; i++)
        dist[i] = INF;

    dist[start] = 0;

    GQueue* queue = g_queue_new();
    g_queue_push_tail(queue, GINT_TO_POINTER(start));

    while (!g_queue_is_empty(queue)) {
        int u = GPOINTER_TO_INT(g_queue_pop_head(queue));

        for (int idx = row_ptr[u]; idx < row_ptr[u + 1]; idx++) {
            int v = col_idx[idx];
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                g_queue_push_tail(queue, GINT_TO_POINTER(v));
            }
        }
    }

    g_queue_free(queue);
}

GraphCSR* read_graph(void) {
    GraphCSR *graph = (GraphCSR*)malloc(sizeof(GraphCSR));

    int n, m;                                                                                                                                        printf("Enter number of nodes and edges: ");
    scanf("%d %d", &n, &m);                                                                                                                                                                                                                                                                           int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(m * sizeof(int));

    printf("Enter CSR row_ptr (%d values):\n", n + 1);
    for (int i = 0; i <= n; i++)
        scanf("%d", &row_ptr[i]);

    printf("Enter CSR col_idx (%d values):\n", m);
    for (int i = 0; i < m; i++)
        scanf("%d", &col_idx[i]);

    graph->nnodes = n;
    graph->nedges = m;
    graph->row_ptr = row_ptr;
    graph->col_idx = col_idx;
    return(graph);
}

void print_graph(GraphCSR* graph) {
    for(int i=0; i<graph->nnodes; i++) {
        fprintf(stdout, "Node %d: ", i);
        for (int j=graph->row_ptr[i]; j<graph->row_ptr[i+1]; j++) {
            fprintf(stdout, "%d ", graph->col_idx[j]);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
    }
}

GraphCSR* defaultgraph_1 (void) {
    GraphCSR *graph = (GraphCSR*)malloc(sizeof(GraphCSR));
    graph->nnodes = 9;
    graph->nedges = 22;

    graph->row_ptr = (int*)malloc(sizeof(int)*graph->nnodes);
    graph->col_idx = (int*)malloc(sizeof(int)*graph->nedges);

    int tmp_row[10] = {0,2,6,9,11,14,16,18,21,22};
    int tmp_col[22] = {1,2,0,2,3,4,0,1,5,1,6,1,6,7,2,7,3,4,4,5,8,7};
    for(int i=0; i<10; i++) graph->row_ptr[i] = tmp_row[i];
    for(int i=0; i<22; i++) graph->col_idx[i] = tmp_col[i];

    print_graph(graph);
    return(graph);
}

GraphCSR* defaultgraph_2(void) {
    // Node count (from 0 to 33, excluding 1X/2X/0X)
    const int nnodes = 34;

    // Hardcoded adjacency list derived from LaTeX
    // All edges are added in both directions (undirected)
    const int edges[][2] = {
	{0,1}, {0,2}, {0,10}, {0,30},
	{1,2},
        {10,11}, {10,13}, {10,20},
        {11,12},
        {12,13},
	{20,23}, {20,24}, {20,30},
        {21,22}, {21,23},
        {22,23},
        {23,24},
        {30,32},
        {31,32},
        {32,33},
    };

    const int nedges = sizeof(edges) / sizeof(edges[0]);
    const int nedges_undirected = nedges * 2;

    // Allocate temp degree array to compute CSR structure
    int degree[nnodes];
    for (int i = 0; i < nnodes; i++) degree[i] = 0;

    // Count degrees
    for (int i = 0; i < nedges; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        degree[u]++;
        degree[v]++;
    }

    // Allocate CSR arrays
    int* row_ptr = (int*)malloc((nnodes + 1) * sizeof(int));
    int* col_idx = (int*)malloc(nedges_undirected * sizeof(int));

    // Build row_ptr
    row_ptr[0] = 0;
    for (int i = 0; i < nnodes; i++) {
        row_ptr[i + 1] = row_ptr[i] + degree[i];
        degree[i] = 0; // reuse to count insert position
    }

    // Fill col_idx using degree[] as offset
    for (int i = 0; i < nedges; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        col_idx[row_ptr[u] + degree[u]++] = v;
        col_idx[row_ptr[v] + degree[v]++] = u;
    }

    // Create and return graph
    GraphCSR* g = (GraphCSR*)malloc(sizeof(GraphCSR));
    g->nnodes = nnodes;
    g->nedges = nedges_undirected;
    g->row_ptr = row_ptr;
    g->col_idx = col_idx;
    print_graph(g);
    return g;
}

void print_fornteers(GraphCSR* graph, int* dist) {
    int n = graph->nnodes;
    int max_dist = dist[0];
    for (int i=1; i<n; i++) if (dist[i] != INF && dist[i]>max_dist) max_dist = dist[i];

    for (int i=0; i<=max_dist; i++) {
        fprintf(stdout, "Fronteer %d: ", i);
	for (int j=0; j<n; j++) if (dist[j] == i) fprintf(stdout, "%d ", j);
	fprintf(stdout, "\n");
    }
}

int main() {
    //GraphCSR *graph = read_graph();
    GraphCSR *graph = defaultgraph_2();
    int* dist = (int*)malloc(graph->nnodes * sizeof(int));

    int start;
    printf("Enter start node: ");
    scanf("%d", &start);

    bfs_csr(graph, start, dist);

    printf("Distances from node %d:\n", start);
    for (int i = 0; i < graph->nnodes; i++) {
        if (dist[i] == INF)
            printf("Node %d: unreachable\n", i);
        else
            printf("Node %d: %d\n", i, dist[i]);
    }

    print_fornteers(graph, dist);

    free_graph(graph);
    free(graph);
    free(dist);
    return 0;
}

