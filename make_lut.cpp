#include "MKL_Sparse_Methods.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float*** make_weight_LUT(float** W, int M, int N) {
    // 计算需要的组数（向上取整）
    int M_groups = (M + LWIDTH - 1) / LWIDTH;  // 等价于 ceil(M/LWIDTH)

    // 定义16个4维的0-1向量（0-15的二进制表示）
    int binary_vectors[LWIDTH_POW2][LWIDTH] = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };

    // 连续内存分配：一次性分配所有需要的内存
    float*** A = (float***)malloc(M_groups * sizeof(float**));
    if (A == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for A\n");
        return NULL;
    }

    // 分配二级指针数组
    A[0] = (float**)malloc(M_groups * N * sizeof(float*));
    if (A[0] == NULL) {
        free(A);
        fprintf(stderr, "Error: Failed to allocate memory for A[0]\n");
        return NULL;
    }

    // 分配实际数据内存（连续）
    A[0][0] = (float*)malloc(M_groups * N * LWIDTH_POW2 * sizeof(float));
    if (A[0][0] == NULL) {
        free(A[0]);
        free(A);
        fprintf(stderr, "Error: Failed to allocate memory for A[0][0]\n");
        return NULL;
    }

    // 设置指针指向连续内存块中的正确位置
    for (int i = 0; i < M_groups; i++) {
        if (i > 0) {
            A[i] = A[0] + i * N;  // 设置A[i]指向正确位置
        }
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            if (index > 0) {
                A[i][j] = A[0][0] + index * LWIDTH_POW2;  // 设置A[i][j]指向正确位置
            }
        }
    }

    // 执行映射操作 - 修改了索引顺序
    for (int group = 0; group < M_groups; group++) {  // M_groups个组
        for (int n = 0; n < N; n++) {  // N列
            for (int lut_idx = 0; lut_idx < LWIDTH_POW2; lut_idx++) {  // 16个LUT值
                A[group][n][lut_idx] = 0.0f;

                // 对于每个group，处理最多LWIDTH个W元素
                for (int k = 0; k < LWIDTH; k++) {
                    int row_in_W = group * LWIDTH + k;  // 在原矩阵W中的行号

                    // 如果row_in_W超出W的实际行数，则使用0（高位补零）
                    if (row_in_W < M) {
                        A[group][n][lut_idx] += W[row_in_W][n] * binary_vectors[lut_idx][k];
                    }
                    // 否则相当于乘以0，不需要加任何值
                }
            }
        }
    }

    return A;
}

// 对应的释放函数
void free_weight_LUT(float*** A) {
    if (A == NULL) return;

    // 只需要释放三个内存块
    if (A[0] != NULL) {
        if (A[0][0] != NULL) {
            free(A[0][0]);  // 释放数据内存
        }
        free(A[0]);  // 释放二级指针数组
    }
    free(A);  // 释放一级指针数组
}

// 示例使用函数 - 修改了访问顺序
void print_LUT_example(float*** A, int M, int N) {
    int M_groups = (M + LWIDTH - 1) / LWIDTH;

    printf("Matrix dimensions: M=%d, N=%d, Groups=%d\n", M, N, M_groups);

    for (int i = 0; i < M_groups; i++) {
        for (int j = 0; j < N; j++) {
            printf("Group[%d][%d] LUT values (first 4): ", i, j);
            for (int k = 0; k < LWIDTH_POW2; k++) {
                printf("%.2f ", A[0][0][i * LWIDTH_POW2 * N + j * LWIDTH_POW2 + k]);
            }
            printf("\n");
        }
    }
}
