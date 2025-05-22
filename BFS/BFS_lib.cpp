#pragma once

#include "BFS_lib.hpp"

#include <mpi.h>
#include <string.h>
#include <algorithm>


void free_graph(GraphCSR* g) {
    free(g->row_ptr);
    free(g->col_idx);
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
    return g;
}

/* DISTRIBUITED STRUCTURE */

void free_distgraph(DistGraphCSR* g) {
    free(g->row_ptr);
    free(g->col_idx);
    free(g->cut_ids);
    free(g->cut_owners);
    free(g->myloc_nodes);
}

int globalId2localId (DistGraphCSR* g, int gid) {
    int lid;
    int* it = std::lower_bound(g->myloc_nodes, g->myloc_nodes + g->l_nnodes, gid);

    if (it != g->myloc_nodes + g->l_nnodes && *it == gid) {
        lid = it - g->myloc_nodes;
    } else {
        it = std::lower_bound(g->cut_ids, g->cut_ids + g->cut_nnodes, gid);
        if (it != g->cut_ids + g->cut_nnodes && *it == gid) {
            lid = it - g->cut_ids + g->l_nnodes;
        } else {
            fprintf(stderr, "Error: %d is neither in myloc vertices nor in the cut\n", gid);
            exit(__LINE__);
        }
    }
    return(lid);
}

int localId2globalId (DistGraphCSR* g, int lid) {
    int gid;
    if (lid <= g->l_nnodes) {
        gid = g->myloc_nodes[lid];
    } else {
        if (lid <= g->l_nnodes + g->cut_nnodes) {
            gid = g->cut_ids[lid - g->l_nnodes];
        } else {
            fprintf(stderr, "Error: local id %d not found.\n", lid);
        }
    }
    return(gid);
}

DistGraphCSR* assign_partition_with_function(GraphCSR* ig, int (*owner_func)(int)) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Temporary buffers
    int *local_nodes = (int*)malloc(sizeof(int) * ig->nnodes);
    int local_count = 0;

    // Step 1: Determine which nodes this rank owns
    for (int gid = 0; gid < ig->nnodes; gid++) {
        if (owner_func(gid) == rank) {
            local_nodes[local_count++] = gid;
        }
    }

    // Step 2: Allocate and initialize GraphCSR for local partition
    DistGraphCSR *part = (DistGraphCSR *)malloc(sizeof(DistGraphCSR));
    part->g_nnodes = ig->nnodes;
    part->g_nedges = ig->nedges;
    part->l_nnodes = local_count;
    part->myloc_nodes = (int*)malloc(sizeof(int) * local_count);
    memcpy(part->myloc_nodes, local_nodes, sizeof(int) * local_count);

    // Temporary for edges
    int *row_ptr = (int*)malloc(sizeof(int) * (local_count + 1));
    int *col_idx = (int*)malloc(sizeof(int) * ig->nedges); // Over-allocated initially

    int edge_pos = 0;
    row_ptr[0] = 0;

    // For ghost detection
    int *ghost_flags = (int*)calloc(ig->nnodes, sizeof(int));

    for (int i = 0; i < local_count; i++) {
        int gid = local_nodes[i];
        int row_start = ig->row_ptr[gid];
        int row_end = ig->row_ptr[gid + 1];

        for (int j = row_start; j < row_end; j++) {
            int neigh_gid = ig->col_idx[j];
            int owner = owner_func(neigh_gid);

            // If the neighbor is owned by this rank, we assume it will have a local ID
            col_idx[edge_pos++] = neigh_gid;

            // Mark as ghost if it's not owned by this rank
            if (owner != rank) {
                ghost_flags[neigh_gid] = 1;
            }
        }
        row_ptr[i + 1] = edge_pos;
    }

    // Compact ghost list
    int *cut_ids = (int*)malloc(sizeof(int) * ig->nnodes);
    int *cut_owners = (int*)malloc(sizeof(int) * ig->nnodes);
    int cut_count = 0;

    for (int gid = 0; gid < ig->nnodes; gid++) {
        if (ghost_flags[gid]) {
            cut_ids[cut_count] = gid;
            cut_owners[cut_count] = owner_func(gid);
            cut_count++;
        }
    }

    // Include gosts as nodes with degree 0
    row_ptr = (int*)realloc(row_ptr, sizeof(int) * (local_count + 1 + cut_count));
    for (int i=local_count+1; i<local_count + cut_count + 1; i++) row_ptr[i] = row_ptr[local_count];

    // Assign to output
    part->nnodes = local_count + cut_count;
    part->nedges = edge_pos;
    part->row_ptr = row_ptr;
    part->col_idx = (int*)realloc(col_idx, sizeof(int) * edge_pos);  // Trim excess
    part->cut_nnodes = cut_count;
    part->cut_ids = (int*)realloc(cut_ids, sizeof(int) * cut_count);
    part->cut_owners = (int*)realloc(cut_owners, sizeof(int) * cut_count);

    // Cleanup
    free(local_nodes);
    free(ghost_flags);

    return part;
}

/* COMMON FUNCTIONS */

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

template<typename GT>
void print_graph_header(GT* graph);

template<>
void print_graph_header<GraphCSR>(GraphCSR* graph) {
    fprintf(stdout, "nnodes: %d, nedges: %d\n", graph->nnodes, graph->nedges);
    return;
}

template<>
void print_graph_header<DistGraphCSR>(DistGraphCSR* graph) {
    fprintf(stdout, "nnodes: %d, nedges: %d, g_nnodes: %d, g_nedges: %d\n",
		    graph->nnodes, graph->nedges, graph->g_nnodes, graph->g_nedges);
    return;
}

template<typename GT>
void print_graph_info(GT* graph);

template<>
void print_graph_info<GraphCSR>(GraphCSR* graph) {
    return;
}

template<>
void print_graph_info<DistGraphCSR>(DistGraphCSR* graph) {
    fprintf(stdout, "myloc_nodes (%d): ", graph->l_nnodes);
    for (int i=0; i<graph->l_nnodes; i++) fprintf(stdout, "%d ", graph->myloc_nodes[i]);
    fprintf(stdout, "\n");

    if (graph->cut_nnodes > 0) {
        fprintf(stdout, "cut_ids (%d): ", graph->cut_nnodes);
        for (int i=0; i<graph->cut_nnodes; i++) fprintf(stdout, "%d ", graph->cut_ids[i]);
        fprintf(stdout, "\n");

        fprintf(stdout, "cut_owners (%d): ", graph->cut_nnodes);
        for (int i=0; i<graph->cut_nnodes; i++) fprintf(stdout, "%d ", graph->cut_owners[i]);
        fprintf(stdout, "\n");
    }
    return;
}

template <typename GT>
void print_graph(GT* graph) {
    print_graph_header(graph);

    for(int i=0; i<graph->nnodes; i++) {
        fprintf(stdout, "Node %d: ", i);
        for (int j=graph->row_ptr[i]; j<graph->row_ptr[i+1]; j++) {
            fprintf(stdout, "%d ", graph->col_idx[j]);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
    }

    print_graph_info(graph);
}

void print_simple_graph(GraphCSR* graph) {
    print_graph(graph);
}

void print_dist_graph(DistGraphCSR* graph) {
    print_graph(graph);
}


template <typename GT>
int get_fronteer(GT* graph, int* dist, int nf, int** fvec) {
    int flen = 0;
    for (int j=0; j<graph->nnodes; j++) if (dist[j] == nf) flen++;
    *fvec = (int*)malloc(sizeof(int)*flen);
    int k = 0;
    for (int j=0; j<graph->nnodes; j++) {
        if (dist[j] == nf) {
            (*fvec)[k] = j;
            k++;
        }
    }
    return(flen);
}

template<typename GT>
int print_graph_edge(GT* graph, int toprint);

template<>
int print_graph_edge<GraphCSR>(GraphCSR* graph, int toprint) {
    return(toprint);
}

template<>
int print_graph_edge<DistGraphCSR>(DistGraphCSR* graph, int toprint) {
    return(localId2globalId(graph, toprint));
}

template <typename GT>
void print_fornteers(GT* graph, int* dist, int cust_max_dist) {
    int n = graph->nnodes;
    int max_dist = dist[0];
    for (int i=1; i<n; i++) if (dist[i] != INF && dist[i]>max_dist) max_dist = dist[i];
    if (cust_max_dist != -1) max_dist = cust_max_dist;

    if (max_dist > 500) {
        fprintf(stdout, "max dist > 500\n");
        return;
    }

    for (int i=0; i<=max_dist; i++) {
        int *fvec, flen = get_fronteer(graph, dist, i, &fvec);
        fprintf(stdout, "Fronteer %d: ", i);
        for (int j=0; j<flen; j++) fprintf(stdout, "%d ", print_graph_edge(graph, fvec[j]));
        fprintf(stdout, "\n");
        free(fvec);
    }
}

void print_fornteers_graph(GraphCSR* graph, int* dist, int cust_max_dist) {
    print_fornteers(graph, dist, cust_max_dist);
}

void print_fornteers_distgraph(DistGraphCSR* graph, int* dist, int cust_max_dist) {
    print_fornteers(graph, dist, cust_max_dist);
}
