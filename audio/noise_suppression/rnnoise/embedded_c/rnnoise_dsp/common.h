
/**
 * @file common.h
 * @brief RNNoise portability wrappers: inline allocator, free, copy, move, and clear helpers.
 * @details
 *   Vendored from xiph/rnnoise \@1cbdbcf. Provides thin wrappers around
 *   malloc()/free() (rnnoise_alloc / rnnoise_free) and convenience macros
 *   for typed memory operations (RNN_COPY, RNN_MOVE, RNN_CLEAR).
 *
 *   On the RA8P1 port heap allocation is not used in the inference path;
 *   these helpers remain present for compilation compatibility with
 *   upstream denoise.c.
 *
 * @note Vendored BSD-3-Clause (xiph/rnnoise). No explicit license header
 *       in the upstream file; governed by the project-level BSD-3-Clause.
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

#ifndef COMMON_H
#define COMMON_H

#include "stdlib.h"
#include "string.h"

#define RNN_INLINE inline
#define OPUS_INLINE inline


/** RNNoise wrapper for malloc(). To do your own dynamic allocation, all you need t
o do is replace this function and rnnoise_free */
#ifndef OVERRIDE_RNNOISE_ALLOC
static RNN_INLINE void *rnnoise_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** RNNoise wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and rnnoise_alloc */
#ifndef OVERRIDE_RNNOISE_FREE
static RNN_INLINE void rnnoise_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif



#endif
