#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <glib.h>
#include <algorithm>
#include <vector>
#include <queue>
#include <set>
#include <unistd.h>
#include <string.h>

#include <mpi.h>

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

template <typename GT>
void print_fornteers(GT* graph, int* dist, int cust_max_dist=-1) {
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
        for (int j=0; j<flen; j++) fprintf(stdout, "%d ", (graph->l_nnodes>0) ? localId2globalId(graph, fvec[j]) : fvec[j]);
        fprintf(stdout, "\n");
        free(fvec);
    }
}

void bfs_csr(GraphCSR* graph, int start, int* dist) {
    int n = graph->nnodes;
    int* row_ptr = graph->row_ptr;
    int* col_idx = graph->col_idx;

    for (int i=0; i<n; i++) dist[i] = INF;  // Set all distances to INF
    dist[start] = 0;

    std::queue<int> q;
    q.push(start);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int idx = row_ptr[u]; idx < row_ptr[u + 1]; ++idx) {
            int v = col_idx[idx];
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}


void dist_bfs_csr(DistGraphCSR* graph, int start, int (*owner_func)(int), int* dist) {
    int n = graph->nnodes;
    int* row_ptr = graph->row_ptr;
    int* col_idx = graph->col_idx;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < n; i++) dist[i] = INF;

    std::queue<int> q;
    std::vector<std::queue<int>> tosend(size);

    if (owner_func(start) == rank) {
        int local_id = globalId2localId(graph, start);
        dist[local_id] = 0;
        q.push(local_id);
    }

    int stop_condition, empty_frt, iteration = 0;

    do {
        // Local processing
        if (!q.empty()) {
            int u_local = q.front();
            q.pop();
            for (int idx = row_ptr[u_local]; idx < row_ptr[u_local + 1]; ++idx) {
                int v_global = col_idx[idx];
                int v_owner = owner_func(v_global);
                if (v_owner == rank) {
                    int v_local = globalId2localId(graph, v_global);
                    if (dist[v_local] == INF) {
                        dist[v_local] = dist[u_local] + 1;
                        q.push(v_local);
                    }
                } else {
                    tosend[v_owner].push(v_global);
                }
            }
        }

        // Send and receive message sizes
        std::vector<int> send_sizes(size), recv_sizes(size);
        std::vector<MPI_Request> send_size_reqs(size), recv_size_reqs(size);
        for (int i = 0; i < size; i++) {
            send_sizes[i] = tosend[i].size();
            MPI_Isend(&send_sizes[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &send_size_reqs[i]);
            MPI_Irecv(&recv_sizes[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_size_reqs[i]);
        }
        MPI_Waitall(size, send_size_reqs.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(size, recv_size_reqs.data(), MPI_STATUSES_IGNORE);

        // Send and receive actual data
        std::vector<MPI_Request> send_data_reqs(size), recv_data_reqs(size);
        std::vector<int*> recv_data(size);
        for (int i = 0; i < size; i++) {
            if (send_sizes[i] > 0) {
                std::vector<int> temp;
                while (!tosend[i].empty()) {
                    temp.push_back(tosend[i].front());
                    tosend[i].pop();
                }
                MPI_Isend(temp.data(), send_sizes[i], MPI_INT, i, 1, MPI_COMM_WORLD, &send_data_reqs[i]);
            } else {
                send_data_reqs[i] = MPI_REQUEST_NULL;
            }

            if (recv_sizes[i] > 0) {
                recv_data[i] = (int*)malloc(sizeof(int) * recv_sizes[i]);
                MPI_Irecv(recv_data[i], recv_sizes[i], MPI_INT, i, 1, MPI_COMM_WORLD, &recv_data_reqs[i]);
            } else {
                recv_data[i] = nullptr;
                recv_data_reqs[i] = MPI_REQUEST_NULL;
            }
        }
        MPI_Waitall(size, send_data_reqs.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(size, recv_data_reqs.data(), MPI_STATUSES_IGNORE);

        // Debug printing block (unchanged)
        if (rank == 0) fprintf(stdout, "=======================================\n\t\tIteration %d\n=======================================\n", iteration);
        for (int i = 0; i < size; i++) {
            if (rank == i) {
                fprintf(stdout, "---------------------------------------\n\t\tProcess %d\n---------------------------------------\n", rank);
                for (int j = 0; j < size; j++) {
                    fprintf(stdout, "Received from %d: ", j);
                    for (int k = 0; k < recv_sizes[j]; k++) {
                        fprintf(stdout, "%d ", recv_data[j][k]);
                    }
                    fprintf(stdout, "\n");
                }

                print_fornteers(graph, dist, iteration);
            }
            sleep(1);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0) fprintf(stdout, "---------------------------------------\n");

        // Add received vertices to local queue
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < recv_sizes[i]; j++) {
                int v_global = recv_data[i][j];
                int v_local = globalId2localId(graph, v_global);
                if (dist[v_local] == INF) {
                    dist[v_local] = iteration+1; // Distance updated as part of BFS
                    q.push(v_local);
                }
            }
            free(recv_data[i]);
        }

        iteration++;
        empty_frt = (q.empty()) ? 1 : 0;
        MPI_Allreduce(&empty_frt, &stop_condition, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    } while (stop_condition != 1);
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

int owner_fn (int x) {
    return(x/10);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size, test_rank = 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    //GraphCSR *graph = read_graph();
    if (rank == 0) fprintf(stdout, "Reading global graph...\n");
    GraphCSR *g_graph = defaultgraph_2();
    if (rank == 0) print_graph<GraphCSR>(g_graph);
    free_graph(g_graph);;
    free(g_graph);

    DistGraphCSR *graph;
    for (int i=0; i<size; i++) {
        if (rank == i) {
            fprintf(stdout, "---------------------------------------\n\t\tProcess %d\n---------------------------------------\n", rank);
            fprintf(stdout, "Partitioning graph...\n");
            graph = assign_partition_with_function(g_graph, owner_fn);
            print_graph(graph);
        }
        sleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank==0) fprintf(stdout, "---------------------------------------\n");

    int* dist = (int*)malloc(graph->nnodes * sizeof(int));

    int start;
    if (rank == 0) {
        fprintf(stdout, "Enter start node: ");
        fflush(stdout);
        scanf("%d", &start);
    }
    MPI_Bcast(&start, 1, MPI_INT, 0, MPI_COMM_WORLD);

    dist_bfs_csr(graph, start, owner_fn, dist);

    if (rank == 0) fprintf(stdout, "=======================================\n\t\tFronteers\n=======================================\n");
    for (int i=0; i<size; i++) {
        if (rank == i) {
            fprintf(stdout, "---------------------------------------\n\t\tProcess %d\n---------------------------------------\n", rank);

            print_fornteers(graph, dist);
        }
        sleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank==0) fprintf(stdout, "---------------------------------------\n");

    free_distgraph(graph);
    free(graph);
    free(dist);

    MPI_Finalize();  // Finalize MPI
    return 0;
}

