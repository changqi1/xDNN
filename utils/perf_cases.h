#pragma once

int perf_mnk[][3] = {
    // For Base 1 thread performance
    {1, 64, 512},
    {2, 64, 512},
    {3, 64, 512},
    {4, 64, 512},
    {5, 64, 512},
    {6, 64, 512},
    {7, 64, 512},
    {8, 64, 512},
    {9, 64, 512},
    {10, 64, 512},
    {11, 64, 512},
    {12, 64, 512},
    {13, 64, 512},
    {14, 64, 512},
    {15, 64, 512},
    {16, 64, 512},
    // For Base 8 thread performance
    {1, 512, 512},
    {2, 512, 512},
    {3, 512, 512},
    {4, 512, 512},
    {5, 512, 512},
    {6, 512, 512},
    {7, 512, 512},
    {8, 512, 512},
    {9, 512, 512},
    {10, 512, 512},
    {11, 512, 512},
    {12, 512, 512},
    {13, 512, 512},
    {14, 512, 512},
    {15, 512, 512},
    {16, 512, 512},
    // For Base 16 thread performance
    {1, 1024, 512},
    {2, 1024, 512},
    {3, 1024, 512},
    {4, 1024, 512},
    {5, 1024, 512},
    {6, 1024, 512},
    {7, 1024, 512},
    {8, 1024, 512},
    {9, 1024, 512},
    {10, 1024, 512},
    {11, 1024, 512},
    {12, 1024, 512},
    {13, 1024, 512},
    {14, 1024, 512},
    {15, 1024, 512},
    {16, 1024, 512},
    // For Base 1 thread k performance
    {1, 64, 32},
    {1, 64, 64},
    {1, 64, 128},
    {1, 64, 256},
    {1, 64, 512},
    {1, 64, 1024},
    {1, 64, 2048},
    {1, 64, 4096},
    {1, 64, 8192},
    {1, 64, 16384},
    {1, 64, 32768},
    {1, 64, 65536},
    // For Base 16 thread k performance
    {1, 1024, 32},
    {1, 1024, 64},
    {1, 1024, 128},
    {1, 1024, 256},
    {1, 1024, 512},
    {1, 1024, 1024},
    {1, 1024, 2048},
    {1, 1024, 4096},
    {1, 1024, 8192},
    {1, 1024, 16384},
    {1, 1024, 32768},
    {1, 1024, 65536},
    // For Base 1 thread k performance
    {16, 64, 32},
    {16, 64, 64},
    {16, 64, 128},
    {16, 64, 256},
    {16, 64, 512},
    {16, 64, 1024},
    {16, 64, 2048},
    {16, 64, 4096},
    {16, 64, 8192},
    {16, 64, 16384},
    {16, 64, 32768},
    {16, 64, 65536},
    // For Base 16 thread k performance
    {16, 1024, 32},
    {16, 1024, 64},
    {16, 1024, 128},
    {16, 1024, 256},
    {16, 1024, 512},
    {16, 1024, 1024},
    {16, 1024, 2048},
    {16, 1024, 4096},
    {16, 1024, 8192},
    {16, 1024, 16384},
    {16, 1024, 32768},
    {16, 1024, 65536},
    // For Bert
    {4, 768, 768},
    {4, 1024, 1024},
    {16, 768, 768},
    {16, 1024, 1024},
    {50, 2304, 768},
    {50, 768, 768},
    {50, 3072, 768},
    {50, 768, 3072},
    {128, 768, 768},
    {128, 1024, 1024},
    // For MLP
    {24, 4096, 1024},
    {24, 1024, 4096},
    // For OPT-13B
    {1, 15360,  5120},
    {1,  5120,  5120},
    {1, 20480,  5120},
    {1,  5120, 20480},
    {1, 13696,  5120},
    {1,  5120, 13696},
    {1, 25136,  5120},
    {2, 15360,  5120},
    {2,  5120,  5120},
    {2, 20480,  5120},
    {2,  5120, 20480},
    {2, 13696,  5120},
    {2,  5120, 13696},
    {2, 25136,  5120},
    {4, 15360,  5120},
    {4,  5120,  5120},
    {4, 20480,  5120},
    {4,  5120, 20480},
    {4, 13696,  5120},
    {4,  5120, 13696},
    {4, 25136,  5120},
    {8, 15360,  5120},
    {8,  5120,  5120},
    {8, 20480,  5120},
    {8,  5120, 20480},
    {8, 13696,  5120},
    {8,  5120, 13696},
    {8, 25136,  5120},
    {16, 15360,  5120},
    {16,  5120,  5120},
    {16, 20480,  5120},
    {16,  5120, 20480},
    {16, 13696,  5120},
    {16,  5120, 13696},
    {16, 25136,  5120},
    {32, 15360,  5120},
    {32,  5120,  5120},
    {32, 20480,  5120},
    {32,  5120, 20480},
    {32, 13696,  5120},
    {32,  5120, 13696},
    {32, 25136,  5120},
    {64, 15360,  5120},
    {64,  5120,  5120},
    {64, 20480,  5120},
    {64,  5120, 20480},
    {64, 13696,  5120},
    {64,  5120, 13696},
    {64, 25136,  5120},
    {128, 15360,  5120},
    {128,  5120,  5120},
    {128, 20480,  5120},
    {128,  5120, 20480},
    {128, 13696,  5120},
    {128,  5120, 13696},
    {128, 25136,  5120},
    // For GPT-J
    {1, 12288,  4096},
    {1,  4096,  4096},
    {1, 16384,  4096},
    {1,  4096, 16384},
    {1, 11008,  4096},
    {1,  4096, 11008},
    {2, 12288,  4096},
    {2,  4096,  4096},
    {2, 16384,  4096},
    {2,  4096, 16384},
    {2, 11008,  4096},
    {2,  4096, 11008},
    {4, 12288,  4096},
    {4,  4096,  4096},
    {4, 16384,  4096},
    {4,  4096, 16384},
    {4, 11008,  4096},
    {4,  4096, 11008},
    {8, 12288,  4096},
    {8,  4096,  4096},
    {8, 16384,  4096},
    {8,  4096, 16384},
    {8, 11008,  4096},
    {8,  4096, 11008},
    {16, 12288,  4096},
    {16,  4096,  4096},
    {16, 16384,  4096},
    {16,  4096, 16384},
    {16, 11008,  4096},
    {16,  4096, 11008},
    {32, 12288,  4096},
    {32,  4096,  4096},
    {32, 16384,  4096},
    {32,  4096, 16384},
    {32, 11008,  4096},
    {32,  4096, 11008},
    {64, 12288,  4096},
    {64,  4096,  4096},
    {64, 16384,  4096},
    {64,  4096, 16384},
    {64, 11008,  4096},
    {64,  4096, 11008},
    {128, 12288,  4096},
    {128,  4096,  4096},
    {128, 16384,  4096},
    {128,  4096, 16384},
    {128, 11008,  4096},
    {128,  4096, 11008},
    // For Alisearch
    {1,  6400,  6144},
    {1,  6144,  6144},
    {1, 24576,  6144},
    {1,  6144, 24576},
    {2,  6400,  6144},
    {2,  6144,  6144},
    {2, 24576,  6144},
    {2,  6144, 24576},
    {4,  6400,  6144},
    {4,  6144,  6144},
    {4, 24576,  6144},
    {4,  6144, 24576},
    {8,  6400,  6144},
    {8,  6144,  6144},
    {8, 24576,  6144},
    {8,  6144, 24576},
    {16,  6400,  6144},
    {16,  6144,  6144},
    {16, 24576,  6144},
    {16,  6144, 24576},
    {32,  6400,  6144},
    {32,  6144,  6144},
    {32, 24576,  6144},
    {32,  6144, 24576},
    {64,  6400,  6144},
    {64,  6144,  6144},
    {64, 24576,  6144},
    {64,  6144, 24576},
    {128,  6400,  6144},
    {128,  6144,  6144},
    {128, 24576,  6144},
    {128,  6144, 24576},
};