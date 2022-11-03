/*************************************************************************
Intel PCL GBB - 
     Library of Optimized Genomics Building Blocks for Intel Processors
**************************************************************************
Copyright (c) 2018, Sanchit Misra, Parallel Computing Lab, 
                    Intel, Bangalore, India
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Sanchit Misra or Intel nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL SANCHIT MISRA OR INTEL BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

**************************************************************************
Author: Sanchit Misra <sanchit.misra@intel.com>
*************************************************************************/

#ifndef _PAIRWISESW_32_H
#define _PAIRWISESW_32_H

#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include "PairWiseSW.h"

#define MAX_SEQ_LEN_32 1024

class PairWiseSW_32: public PairWiseSW
{
public:
    PairWiseSW_32(int32_t w_match,
                  int32_t w_mismatch,
                  int32_t w_open,
                  int32_t w_extend);
	~PairWiseSW_32();
	void getScores(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, uint16_t numThreads);

protected:
    void smithWatermanInterTaskVectorized(uint32_t seq1SoA[],
                                          uint32_t seq2SoA[],
                                          int32_t nrow,
                                          int32_t ncol,
                                          SeqPair *p,
                                          uint16_t tid);
    void smithWatermanBatchWrapper(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, uint16_t numThreads);

private:

        int32_t *F_;
        int32_t *H_;
};
#endif

