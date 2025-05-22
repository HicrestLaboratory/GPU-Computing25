#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <set>
#include <unistd.h>

#include <mpi.h>
#include "BFS_lib.hpp"


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

                print_fornteers_distgraph(graph, dist, iteration);
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
    if (rank == 0) print_simple_graph(g_graph);
    free_graph(g_graph);;
    free(g_graph);

    DistGraphCSR *graph;
    for (int i=0; i<size; i++) {
        if (rank == i) {
            fprintf(stdout, "---------------------------------------\n\t\tProcess %d\n---------------------------------------\n", rank);
            fprintf(stdout, "Partitioning graph...\n");
            graph = assign_partition_with_function(g_graph, owner_fn);
            print_dist_graph(graph);
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

            print_fornteers_distgraph(graph, dist);
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

