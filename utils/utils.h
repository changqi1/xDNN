#pragma once

#include <memory>

#include "data_types/data_types.h"
#include "test.h"
#include "timer.h"
#include "unit_cases.h"
#include "perf_cases.h"

#define ALLOC(DATATYPE, VALUE, SIZE)  std::unique_ptr<DATATYPE, decltype(&free)> VALUE(static_cast<DATATYPE*>(aligned_alloc(64, SIZE * sizeof(DATATYPE))), &free)
