#include <stdio.h>

#define CHECK(x) {                                           \
        cudaError_t result = x;                              \
        if (result != cudaSuccess) {                         \
            printf("%d:unexpected error:%s, expecting:%s\n", \
                    __LINE__,                                \
                    cudaGetErrorString(result),              \
                    cudaGetErrorString(cudaSuccess));        \
            exit(0);                                         \
        }                                                    \
    }


#define ARRXY(arr,x,y) arr[(x) + ( (y) * (max_x) )]
#define P_X(x) ((x + max_x - 1) % max_x)
#define N_X(x) ((x + 1) % max_x)
#define P_Y(y) ((y + max_y - 1) % max_y)
#define N_Y(y) ((y + 1) % max_y)

__host__ __device__ void
printArray(char *arr, int max_y, int max_x)
{
    int x, y;

    if (!arr)
        return;

    printf("\n");
    for (y = max_y - 1; y >= 0; --y) {
        for (x = 0; x < max_x; ++x) {
            printf("%s", ARRXY(arr,x,y) ? "X":".");
        }
        printf("\n");
    }
}

__device__ __forceinline__ int
getNeighborCount(int base_offset, const int max_y, const int max_x, const int x, const int y)
{
    int nborcount = 0;
    extern __shared__ char buf[];
    char *cur;

    cur = buf + base_offset;

    nborcount += ARRXY(cur, P_X(x), y);
    nborcount += ARRXY(cur, P_X(x), P_Y(y));
    nborcount += ARRXY(cur, P_X(x), N_Y(y));

    nborcount += ARRXY(cur, x, P_Y(y));
    nborcount += ARRXY(cur, x, N_Y(y));

    nborcount += ARRXY(cur, N_X(x), y);
    nborcount += ARRXY(cur, N_X(x), P_Y(y));
    nborcount += ARRXY(cur, N_X(x), N_Y(y));

    return nborcount;
}

__device__ __forceinline__ void
updateCell(int cur_offset, int next_offset, const int max_y, const int max_x, const int x, const int y, const int singlethread)
{
    int nborcount = 0;
    extern __shared__ char buf[];
    char *cur, *next;

    cur = buf + cur_offset;
    next = buf + next_offset;

    nborcount = getNeighborCount(cur_offset, max_y, max_x, x, y);

    // Compute the next in the next buffer
    // 1. Any live cell with <2 neighbors dies
    // 2. Any live cell with 2 || 3 neighbors lives
    // 3. Any live cell with >3 neigbors dies
    // 4. Any dead cell with =3 neighbors becomes alive

    if (ARRXY(cur,x, y) &&
        (nborcount < 2 || nborcount > 3))
            ARRXY(next, x,y) = 0;
    else if (!ARRXY(cur, x, y) &&
             nborcount == 3)
        ARRXY(next, x, y) = 1;
    else
        ARRXY(next, x, y) = ARRXY(cur, x, y);
}

__global__ void
gameLoop(char *raw_in, char *raw_out, const int max_y, const int max_x, int num_iter, int print_interval, int singlethread)
{
    extern __shared__ char buf[];
    char *cur, *next, *tmp;
    size_t arraysize = 0;
    int iter, x, y, i, j;

    // Sanity checks
    arraysize = (max_x) * (max_y);

    // Skip threads we dont care about
    if (singlethread) {
        if (threadIdx.x > 0 || threadIdx.y > 0)
            return;
    }

    if (threadIdx.x >= max_x)
        return;

    if (threadIdx.y >= max_y)
        return;

    cur  = buf;
    next  = (cur + arraysize);

    x = threadIdx.x + 0;
    y = threadIdx.y + 0;

    // Reset Shmem
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (i = 0; i < max_x; ++i) {
            for (j = 0; j < max_y; ++j) {
                ARRXY(cur, i, j) = 0;
                ARRXY(next, i, j) = 0;
            }
        }
    }
    __syncthreads();

    // Populate the shmem buffer
    if (singlethread) {
        for (x = 0; x < max_x; ++x)
            for (y = 0; y < max_y; ++y)
                ARRXY(cur, x, y) = ARRXY(raw_in, x, y);
    }
    else
        ARRXY(cur, x, y) = ARRXY(raw_in, threadIdx.x, threadIdx.y);

    __syncthreads();

    // Start the iteration loop
    for (iter = 0; iter < num_iter; ++iter) {
        // Compute the neighbor count in the current state

        if (singlethread) {
            for (x = 0; x < max_x; ++x)
                for (y = 0; y < max_y; ++y)
                    updateCell(cur - buf, next - buf, max_y, max_x, x, y, singlethread);

            if ((threadIdx.x == 0 && threadIdx.y == 0) &&
                print_interval &&
                !(iter % print_interval)) {
                printArray(cur, max_y, max_x);
            }
        }
        else {
            updateCell(cur - buf, next - buf, max_y, max_x, x, y, singlethread);

            if ((threadIdx.x == 0 && threadIdx.y == 0) &&
                print_interval &&
                !(iter % print_interval)) {
                printArray(cur, max_y, max_x);
            }
        }

        // Swap the next and current states :
        tmp = cur;
        cur = next;
        next = tmp;

    }

    // Copy data out
    if (singlethread) {
        for (x = 0; x < max_x; ++x)
            for (y = 0; y < max_y; ++y)
                ARRXY(raw_out,x, y) = ARRXY(cur, x, y);
    }
    else
        ARRXY(raw_out,threadIdx.x, threadIdx.y) = ARRXY(cur, x, y);
}

void
initArray(char *arr, int max_y, int max_x, unsigned int seed, float bias)
{
    int x, y;

    if (!arr)
        return;

    if (bias >= 1 || bias <= 0)
        return;

    for (y = 0; y < max_y; ++y) {
        for (x = 0; x < max_x; ++x) {
            ARRXY(arr,x,y) = (rand() >= (RAND_MAX * (bias ))) ? 0 : 1;
        }
    }
}

bool
compareArrays(char *arr1, char *arr2, int max_y, int max_x)
{
    int x, y;
    for (y = 0; y < max_y; ++y) {
        for (x = 0; x < max_x; ++x) {
           if(ARRXY(arr1,x,y) != ARRXY(arr2,x,y)) {
                printf("Mismatch at x:%d, y:%d\n", x, y);
                return false;
            }
        }
    }
    return true;
}

float
getLiveness(char *arr, int max_y, int max_x)
{
    int size = max_y * max_x;
    int sum = 0;
    int x,y;

    for (y = 0; y < max_y; ++y)
        for (x = 0; x < max_x; ++x)
           sum += ARRXY(arr,x,y);

    return (1.0*sum)/size;
}

int
main(int argc, char **argv)
{
    char *array = NULL, *array2 = NULL;
    char *d_in, *d_out, *d_out2;
    bool mismatch = false;

    int max_x  = (argc > 1) ? atol(argv[1]) : 7;
    int max_y = (argc > 2) ? atol(argv[2]) : 7;
    int dev_iter = (argc > 3) ? atol(argv[3]) : 10;
    int print_interval = (argc > 4) ? atol(argv[4]) : 0;
    float bias  = (argc > 5) ? atof(argv[5]) : 1.0/3;
    unsigned int seed =  (argc > 6) ? atol(argv[6]) : 129;

    float initial_liveness = 1.0;
    size_t bufsize = max_y * max_x * sizeof(char);
    size_t shmemsize = ((max_y)*(max_x)) * 2;

    array = (char*)calloc(1, bufsize);
    array2 = (char*)calloc(1, bufsize);
    if (!array || !array2) {
        printf("Failed to allocate memory\n");
        return -1;
    }

    CHECK(cudaMalloc(&d_in, bufsize));
    CHECK(cudaMalloc(&d_out, bufsize));
    CHECK(cudaMalloc(&d_out2, bufsize));

    printf (" Generating random array (%dx%d) Seed:%u Target Liveness:%f\n",
            max_y, max_x, seed, bias);
    initArray(array, max_y, max_x, seed, bias);
    initial_liveness = getLiveness(array, max_y, max_x);

    CHECK(cudaMemset(d_out, 0x0, bufsize));
    CHECK(cudaMemset(d_out2, 0x0, bufsize));

    dim3 threads(max_x, max_y, 1);

    CHECK(cudaMemcpy(d_in, array, bufsize, cudaMemcpyHostToDevice));
    gameLoop<<<1, threads, shmemsize>>> (d_in, d_out, max_y, max_x, dev_iter, print_interval, 0);
    gameLoop<<<1, threads, shmemsize>>> (d_in, d_out2, max_y, max_x, dev_iter, print_interval, 1);
    CHECK(cudaMemcpy(array, d_out, bufsize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(array2, d_out2, bufsize, cudaMemcpyDeviceToHost));

    printf(" Array %dx%d (Shmem :%u) Iterations: %u Initial Liveness:%f Final Liveness:%f\n",
           max_y, max_x, shmemsize, dev_iter, initial_liveness, getLiveness(array, max_y, max_x));

    if (!compareArrays(array, array2, max_y, max_x)) {
        printf("Mismatch !!\n");
        mismatch = true;
    }

    printf(" Final Array : ");
    printArray(array, max_y, max_x);
    if (mismatch) {
        printf(" \n Final Array : (single threaded)");
        printArray(array2, max_y, max_x);
    }

    free(array);
    free(array2);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_out2));
    CHECK(cudaDeviceReset());
    return 0;
}
