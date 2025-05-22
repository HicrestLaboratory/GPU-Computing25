#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX

typedef struct {
    int nnodes;
    int nedges;

    int *row_ptr;
    int *col_idx;
} GraphCSR;

void free_graph(GraphCSR* g);
GraphCSR* defaultgraph_1 (void);
GraphCSR* defaultgraph_2(void);

/* DISTRIBUITED STRUCTURE */

typedef struct {
    int nnodes;
    int nedges;

    int *row_ptr;
    int *col_idx;

    int g_nnodes;
    int g_nedges;

    int l_nnodes;
    int *myloc_nodes;

    int cut_nnodes;
    int *cut_ids;
    int *cut_owners;
} DistGraphCSR;

void free_distgraph(DistGraphCSR* g);
int globalId2localId (DistGraphCSR* g, int gid);
int localId2globalId (DistGraphCSR* g, int lid);
DistGraphCSR* assign_partition_with_function(GraphCSR* ig, int (*owner_func)(int));

/* COMMON FUNCTIONS */

GraphCSR* read_graph(void);

void print_simple_graph(GraphCSR* graph);
void print_dist_graph(DistGraphCSR* graph);


void print_fornteers_graph(GraphCSR* graph, int* dist, int cust_max_dist=-1);
void print_fornteers_distgraph(DistGraphCSR* graph, int* dist, int cust_max_dist=-1);
