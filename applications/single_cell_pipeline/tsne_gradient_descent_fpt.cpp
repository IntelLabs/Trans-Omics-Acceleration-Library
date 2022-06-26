/** file tsne_gradient_descent_fpt.cpp */
/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __INTERNAL_TSNE_GRADIENT_DESCENT_FPT_CPP__
#define __INTERNAL_TSNE_GRADIENT_DESCENT_FPT_CPP__

#include "algorithms/tsne/tsne_gradient_descent.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "services/daal_defines.h"
#include "services/env_detect.h"
#include "src/externals/service_math.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

using namespace daal::data_management;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename DataType, CpuType cpu>
class TlsMax : public daal::TlsMem<DataType, cpu, services::internal::ScalableCalloc<DataType, cpu> >
{
public:
    typedef daal::TlsMem<DataType, cpu, services::internal::ScalableCalloc<DataType, cpu> > super;
    TlsMax(size_t n) : super(n) {}
    void reduceTo(DataType * res, size_t n)
    {
        bool bFirst = true;
        this->reduce([=, &bFirst](DataType * ptr) -> void {
            if (!ptr) return;
            if (bFirst)
            {
                for (size_t i = 0; i < n; ++i) res[i] = ptr[i];
                bFirst = false;
            }
            else
            {
                for (size_t i = 0; i < n; ++i) res[i] = services::internal::max<cpu, DataType>(res[i], ptr[i]);
            }
        });
    }
};

template <typename IdxType, daal::CpuType cpu>
services::Status maxRowElementsImpl(const size_t * row, const IdxType N, IdxType & nElements, const IdxType & blockOfRows)
{
    TlsMax<IdxType, cpu> maxTlsData(1);
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        IdxType * localMax   = maxTlsData.local();
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            localMax[0] = services::internal::max<cpu, IdxType>(localMax[0], IdxType((row[i + 1] - row[i])));
        }
    });
    maxTlsData.reduceTo(&nElements, 1);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status boundingBoxKernelImpl(DataType * posx, DataType * posy, const IdxType N, DataType & radius, DataType & centerx, DataType & centery)
{
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);

    DataType box[4] = { posx[0], posx[0], posy[0], posy[0] };

    daal::static_tls<DataType *> tlsBox([=]() {
        auto localBox = services::internal::service_malloc<DataType, cpu>(4);

        localBox[0] = daal::services::internal::MaxVal<DataType>::get();
        localBox[1] = -daal::services::internal::MaxVal<DataType>::get();
        localBox[2] = daal::services::internal::MaxVal<DataType>::get();
        localBox[3] = -daal::services::internal::MaxVal<DataType>::get();

        return localBox;
    });

    const IdxType nThreads    = tlsBox.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (N + nThreads - 1) / nThreads);
    const IdxType nBlocks     = (N + sizeOfBlock - 1) / sizeOfBlock;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        DataType * localBox  = tlsBox.local(tid);

        for (IdxType i = iStart; i < iEnd; ++i)
        {
            localBox[0] = services::internal::min<cpu, DataType>(localBox[0], posx[i]);
            localBox[1] = services::internal::max<cpu, DataType>(localBox[1], posx[i]);
            localBox[2] = services::internal::min<cpu, DataType>(localBox[2], posy[i]);
            localBox[3] = services::internal::max<cpu, DataType>(localBox[3], posy[i]);
        }
    });

    tlsBox.reduce([&](DataType * ptr) -> void {
        if (!ptr) return;

        box[0] = services::internal::min<cpu, DataType>(box[0], ptr[0]);
        box[1] = services::internal::max<cpu, DataType>(box[1], ptr[1]);
        box[2] = services::internal::min<cpu, DataType>(box[2], ptr[2]);
        box[3] = services::internal::max<cpu, DataType>(box[3], ptr[3]);

        services::internal::service_free<DataType, cpu>(ptr);
    });

    //save results
    centerx = (box[0] + box[1]) * 0.5;
    centery = (box[2] + box[3]) * 0.5;
    radius  = services::internal::max<cpu, DataType>(box[1] - box[0], box[3] - box[2]) * 0.5005f;

    return services::Status();
}

#define MAX_LEVEL 32

template <typename IdxType, typename DataType>
struct MemoryCtxType
{
    int capacity            = 0;
    DataType * posx         = nullptr;
    DataType * posy         = nullptr;
    uint64_t * morton_codes = nullptr;
    IdxType * z_order_idx   = nullptr;
    IdxType * t_order_idx   = nullptr;
    DataType * repx         = nullptr;
    DataType * repy         = nullptr;

    DataType * attrx = nullptr;
    DataType * attry = nullptr;
    DataType * gainx = nullptr;
    DataType * gainy = nullptr;
    DataType * oforx = nullptr;
    DataType * ofory = nullptr;
};

struct qTreeNode
{
    int fpos; // sign (1 bit), nonempty children (2 bits), offset to first child (29 bit)
    int cnt;  // count of point in subspace
};

template <typename IdxType, typename DataType>
struct TreeCtxType
{
    int capacity             = 0;
    int size                 = 0;
    int layerSize[MAX_LEVEL] = {};
    int layerOffs[MAX_LEVEL] = {};
    qTreeNode * tree         = nullptr;
    DataType * centx         = nullptr;
    DataType * centy         = nullptr;
};

template <typename IdxType, typename DataType, daal::CpuType cpu>
inline void buildSubtree5(TreeCtxType<IdxType, DataType> & qTree, int level, IdxType * zOrder, IdxType * tOrder, uint64_t * mc, int * hist)
{
    const int sh   = 54 - (level << 1);
    const int bcnt = qTree.tree[0].cnt;
    const int bpos = qTree.tree[0].fpos;

    int * offs = hist + 1024 + 1024;

    int i, l, c;

    // Sort indexes for 10 bits of the morton code (5 tree levels)
    for (offs[0] = 0, i = 1; i < 1024; i++) offs[i] = offs[i - 1] + hist[i - 1];
    if (level)
    {
        for (i = bpos; i < bpos + bcnt; i++) tOrder[bpos + offs[(mc[zOrder[i]] >> sh) & 0x3FF]++] = zOrder[i];
        for (i = bpos; i < bpos + bcnt; i++) zOrder[i] = tOrder[i];
    }
    else
    {
        for (i = bpos; i < bpos + bcnt; i++) zOrder[offs[mc[i] >> sh]++] = i;
    }

    // Hierarchically aggregate histogram for 5 levels
    for (i = 0; i < 256 + 64 + 16 + 4 + 1; i++) hist[1024 + i] = hist[(i << 2) + 0] + hist[(i << 2) + 1] + hist[(i << 2) + 2] + hist[(i << 2) + 3];

    const int h_ofst[6] = { 1024 + 256 + 64 + 16 + 4, 1024 + 256 + 64 + 16, 1024 + 256 + 64, 1024 + 256, 1024, 0 };
    const int h_size[6] = { 1, 4, 16, 64, 256, 1024 };

    int nodeSize, posOffs, nodeOffs = 0, childOffs = 1;

    // Construct quadTree layer-by-layer using aggregated histograms
    for (l = 0; l < 6; l++)
    {
        posOffs            = bpos;
        qTree.layerOffs[l] = nodeOffs;
        for (c = 0; c < h_size[l]; c++)
        {
            nodeSize = hist[h_ofst[l] + c];

            if (nodeSize == 0) continue;

            if (nodeSize > 0)
            {
                qTree.tree[nodeOffs].cnt = nodeSize;
                if (l < 5)
                {
                    if (nodeSize > level + l + 1)
                    {
                        int cnt = !!hist[h_ofst[l + 1] + (c << 2) + 0] + !!hist[h_ofst[l + 1] + (c << 2) + 1] + !!hist[h_ofst[l + 1] + (c << 2) + 2]
                                  + !!hist[h_ofst[l + 1] + (c << 2) + 3];

                        //                        std::cout << "Adding internal node with " << cnt << " non-empty children; ";
                        //                        std::cout << "their offset is " << childOffs << std::endl;
                        qTree.tree[nodeOffs].fpos = 0x80000000 | ((cnt - 1) << 29) | childOffs;
                        childOffs += cnt;
                    }
                    else
                    {
                        //                        std::cout << "Adding internal leaf with size = " << nodeSize << std::endl;
                        qTree.tree[nodeOffs].fpos = posOffs;

                        hist[h_ofst[l + 1] + (c << 2) + 0] = -hist[h_ofst[l + 1] + (c << 2) + 0];
                        hist[h_ofst[l + 1] + (c << 2) + 1] = -hist[h_ofst[l + 1] + (c << 2) + 1];
                        hist[h_ofst[l + 1] + (c << 2) + 2] = -hist[h_ofst[l + 1] + (c << 2) + 2];
                        hist[h_ofst[l + 1] + (c << 2) + 3] = -hist[h_ofst[l + 1] + (c << 2) + 3];
                    }
                }
                else
                {
                    //                    std::cout << "Adding terminal leaf with size = " << nodeSize << std::endl;
                    qTree.tree[nodeOffs].fpos = posOffs;
                }
                nodeOffs++;
            }
            else
            {
                //                std::cout << "Skipping non-empty node" << std::endl;
                if (l < 5)
                {
                    hist[h_ofst[l + 1] + (c << 2) + 0] = -hist[h_ofst[l + 1] + (c << 2) + 0];
                    hist[h_ofst[l + 1] + (c << 2) + 1] = -hist[h_ofst[l + 1] + (c << 2) + 1];
                    hist[h_ofst[l + 1] + (c << 2) + 2] = -hist[h_ofst[l + 1] + (c << 2) + 2];
                    hist[h_ofst[l + 1] + (c << 2) + 3] = -hist[h_ofst[l + 1] + (c << 2) + 3];
                }
                nodeSize = -nodeSize;
            }
            posOffs += nodeSize;
        }
    }
    qTree.layerOffs[6] = qTree.size = nodeOffs;

    for (int i = 0; i < 6; i++) qTree.layerSize[i] = qTree.layerOffs[i + 1] - qTree.layerOffs[i];
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status qTreeBuildingKernelImpl(MemoryCtxType<IdxType, DataType> & mem, TreeCtxType<IdxType, DataType> & qTree, const DataType & radius,
                                         const DataType & centerx, const DataType & centery)
{
    DAAL_CHECK_MALLOC(mem.posx);
    DAAL_CHECK_MALLOC(mem.posy);
    DAAL_CHECK_MALLOC(mem.morton_codes);

    int * mHist = services::internal::service_calloc<int, cpu>(1024 + 1024 + 1024);
    DAAL_CHECK_MALLOC(mHist);

    daal::static_tls<int *> tlsHist1024([=]() {
        auto localHist = services::internal::service_calloc<int, cpu>(1024);
        return localHist;
    });

    const IdxType nThreads    = tlsHist1024.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (mem.capacity + nThreads - 1) / nThreads);
    const IdxType nBlocks     = (mem.capacity + sizeOfBlock - 1) / sizeOfBlock;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(mem.capacity, iStart + sizeOfBlock);
        const DataType rootx = centerx - radius;
        const DataType rooty = centery - radius;

        const double scale = 2147483648.0 / radius;

        int * hist = tlsHist1024.local(tid);

        uint64_t x, y;

        for (IdxType i = iStart; i < iEnd; i++)
        {
            x = (uint64_t)((mem.posx[i] - rootx) * scale);
            y = (uint64_t)((mem.posy[i] - rooty) * scale);

            x &=
                0x00000000ffffffff; // x = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24 23,22,21,20  19,18,17,16  15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            x = (x ^ (x << 16))
                & 0x0000ffff0000ffff; // x = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,- 15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            x = (x ^ (x << 8))
                & 0x00ff00ff00ff00ff; // x = -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  -,-,-,-  -,-,-,-  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  15,14,13,12  11,10,9,8  -,-,-,-  -,-,-,- 7,6,5,4  3,2,1,0
            x = (x ^ (x << 4))
                & 0x0f0f0f0f0f0f0f0f; // x = -,-,-,-  31,30,29,28  -,-,-,-  27,26,25,24  -,-,-,- 23,22,21,20  -,-,-,-  19,18,17,16  -,-,-,-  15,14,13,12  -,-,-,-  11,10,9,8  -,-,-,-  7,6,5,4  -,-,-,-  3,2,1,0
            x = (x ^ (x << 2))
                & 0x3333333333333333; // x = -,-,31,30 -,-,29,28  -,-,27,26 -,-,25,24   -,-,23,22  -,-,21,20  -,-,19,18  -,-,17,16  -,-,15,14  -,-,13,12  -,-,11,10  -,-,9,8  -,-,7,6  -,-,5,4  -,-,3,2  -,-,1,0
            x = (x ^ (x << 1))
                & 0x5555555555555555; // x = -,31,-,30 -,29,-,28  -,27,-,26 -,25,-,24   -,23,-,22  -,21,-,20  -,19,-,18  -,17,-,16  -,15,-,14  -,13,-,12  -,11,-,10  -,9,-,8  -,7,-,6  -,5,-,4  -,3,-,2  -,1,-,0

            y &=
                0x00000000ffffffff; // y = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24 23,22,21,20  19,18,17,16  15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            y = (y ^ (y << 16))
                & 0x0000ffff0000ffff; // y = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,- 15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            y = (y ^ (y << 8))
                & 0x00ff00ff00ff00ff; // y = -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  -,-,-,-  -,-,-,-  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  15,14,13,12  11,10,9,8  -,-,-,-  -,-,-,- 7,6,5,4  3,2,1,0
            y = (y ^ (y << 4))
                & 0x0f0f0f0f0f0f0f0f; // y = -,-,-,-  31,30,29,28  -,-,-,-  27,26,25,24  -,-,-,- 23,22,21,20  -,-,-,-  19,18,17,16  -,-,-,-  15,14,13,12  -,-,-,-  11,10,9,8  -,-,-,-  7,6,5,4  -,-,-,-  3,2,1,0
            y = (y ^ (y << 2))
                & 0x3333333333333333; // y = -,-,31,30 -,-,29,28  -,-,27,26 -,-,25,24   -,-,23,22  -,-,21,20  -,-,19,18  -,-,17,16  -,-,15,14  -,-,13,12  -,-,11,10  -,-,9,8  -,-,7,6  -,-,5,4  -,-,3,2  -,-,1,0
            y = (y ^ (y << 1))
                & 0x5555555555555555; // y = -,31,-,30 -,29,-,28  -,27,-,26 -,25,-,24   -,23,-,22  -,21,-,20  -,19,-,18  -,17,-,16  -,15,-,14  -,13,-,12  -,11,-,10  -,9,-,8  -,7,-,6  -,5,-,4  -,3,-,2  -,1,-,0

            x |= (y << 1);

            mem.morton_codes[i] = x;

            hist[x >> 54]++;
        }
    });

    tlsHist1024.reduce([&](int * ptr) -> void {
        if (!ptr) return;

        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (int i = 0; i < 1024; i++) mHist[i] += ptr[i];

        services::internal::service_free<int, cpu>(ptr);
    });

    /************************* Build the quadTree ***********************/
    {
        qTree.size         = 0;
        qTree.tree[0].fpos = 0;
        qTree.tree[0].cnt  = mem.capacity;
        for (int i = 0; i < 32; i++) qTree.layerSize[i] = 0;
        for (int i = 0; i < 32; i++) qTree.layerOffs[i] = 0;

        buildSubtree5<IdxType, DataType, cpu>(qTree, 0, mem.z_order_idx, mem.t_order_idx, mem.morton_codes, mHist);

        qTreeNode * subNodes                      = nullptr;
        TreeCtxType<IdxType, DataType> * subTrees = nullptr;
        int subTreeCnt                            = 0;

        for (int pass = 0; pass < 5; pass++)
        {
            int bLevel    = 5 * (pass + 1);
            int bLayerBeg = qTree.layerOffs[bLevel];
            int bLayerEnd = qTree.layerOffs[bLevel] + qTree.layerSize[bLevel];
            int bNodes    = 0;

            for (int c = bLayerBeg; c < bLayerEnd; c++)
                if (qTree.tree[c].cnt > bLevel + 1) bNodes++;
            // std::cout << "Bottom nodes to split: " << bNodes << std::endl;

            // Terminate subtrees creation if there are not enough bottom nodes to split
            if (bNodes < 1) break;

            // Re/allocate worker space for bottom subtrees if needed
            if (bNodes > subTreeCnt)
            {
                if (subTrees)
                {
                    services::internal::service_free<TreeCtxType<IdxType, DataType>, cpu>(subTrees);
                    services::internal::service_free<qTreeNode, cpu>(subNodes);
                }
                subNodes   = services::internal::service_malloc<qTreeNode, cpu>(bNodes * 2048);
                subTrees   = services::internal::service_malloc<TreeCtxType<IdxType, DataType>, cpu>(bNodes);
                subTreeCnt = bNodes;
            }

            for (int c = bLayerBeg, bNodes = 0; c < bLayerEnd; c++)
                if (qTree.tree[c].cnt > bLevel + 1)
                {
                    subTrees[bNodes].size     = 0;
                    subTrees[bNodes].capacity = 2048;
                    subTrees[bNodes].tree     = subNodes + bNodes * 2048;
                    subTrees[bNodes].tree[0]  = qTree.tree[c];
                    bNodes++;
                }

            // Build bottom subtrees in parallel
            daal::static_tls<int *> tlsHist3072([=]() { return services::internal::service_malloc<int, cpu>(1024 + 1024 + 1024); });

            const IdxType nThreads    = tlsHist3072.nthreads();
            const IdxType sizeOfBlock = 1;
            const IdxType nBlocks     = bNodes;

            daal::static_threader_for(nBlocks, [&](IdxType iSubTree, IdxType tid) {
                int * hist = tlsHist3072.local(tid);

                const int sft  = 54 - (bLevel << 1);
                const int bcnt = subTrees[iSubTree].tree[0].cnt;
                const int bpos = subTrees[iSubTree].tree[0].fpos;

                services::internal::service_memset<int, cpu>(hist, 0, 1024);
                for (int i = bpos; i < bpos + bcnt; i++) hist[(mem.morton_codes[mem.z_order_idx[i]] >> sft) & 0x3FF]++;

                buildSubtree5<IdxType, DataType, cpu>(subTrees[iSubTree], bLevel, mem.z_order_idx, mem.t_order_idx, mem.morton_codes, hist);
            });
            tlsHist3072.reduce([&](int * ptr) -> void {
                if (!ptr) return;
                services::internal::service_free<int, cpu>(ptr);
            });

            // Reallocate the tree if needed
            int newTreeSize = qTree.size;

            for (int l = 1; l < 6; l++)
                for (int s = 0; s < bNodes; s++) newTreeSize += subTrees[s].layerSize[l];

            if (newTreeSize > qTree.capacity)
            {
                int capacity      = newTreeSize + (newTreeSize >> 2);
                qTreeNode * nodes = services::internal::service_malloc<qTreeNode, cpu>(capacity);

                services::internal::tmemcpy<qTreeNode, cpu>(nodes, qTree.tree, qTree.size);
                services::internal::service_free<qTreeNode, cpu>(qTree.tree);
                services::internal::service_free<DataType, cpu>(qTree.centx);
                services::internal::service_free<DataType, cpu>(qTree.centy);

                qTree.centx    = services::internal::service_malloc<DataType, cpu>(capacity);
                qTree.centy    = services::internal::service_malloc<DataType, cpu>(capacity);
                qTree.tree     = nodes;
                qTree.capacity = capacity;
            }

            // Aggregate subtrees into main tree
            {
                // Replace splitted bottom leafs with top nodes from subtrees
                int nodeOffs  = qTree.layerOffs[bLevel];
                int childOffs = qTree.layerOffs[bLevel + 1];

                bNodes = 0;
                for (int c = 0; c < qTree.layerSize[bLevel]; c++)
                {
                    if (qTree.tree[nodeOffs].cnt > bLevel + 1)
                    {
                        qTree.tree[nodeOffs].fpos = subTrees[bNodes].tree[0].fpos & 0xE0000000;
                        qTree.tree[nodeOffs].fpos |= childOffs;
                        childOffs += 1 + ((qTree.tree[nodeOffs].fpos >> 29) & 0x3);
                        bNodes++;
                    }
                    nodeOffs++;
                }

                // Copy remaining nodes from subtrees recalculating offsets
                for (int l = 1; l < 6; l++)
                {
                    qTree.layerOffs[bLevel + l] = nodeOffs;
                    for (int s = 0; s < bNodes; s++)
                    {
                        for (int c = 0; c < subTrees[s].layerSize[l]; c++)
                        {
                            qTree.tree[nodeOffs] = subTrees[s].tree[subTrees[s].layerOffs[l] + c];
                            if (qTree.tree[nodeOffs].fpos < 0)
                            {
                                qTree.tree[nodeOffs].fpos &= 0xE0000000;
                                qTree.tree[nodeOffs].fpos |= childOffs;
                                childOffs += 1 + ((qTree.tree[nodeOffs].fpos >> 29) & 0x3);
                            }
                            nodeOffs++;
                        }
                    }
                }
                qTree.layerOffs[bLevel + 6] = qTree.size = nodeOffs;
                for (int i = 0; i < 6; i++) qTree.layerSize[bLevel + i] = qTree.layerOffs[bLevel + i + 1] - qTree.layerOffs[bLevel + i];
            }
        }
        if (subTreeCnt)
        {
            services::internal::service_free<qTreeNode, cpu>(subTrees[0].tree);
            services::internal::service_free<TreeCtxType<IdxType, DataType>, cpu>(subTrees);
        }
    }
    /*
    for (int i = 0; i < 32; i++)
        std::cout << "Layer " << i << " offs: " << qTree.layerOffs[i] << " size: " << qTree.layerSize[i] << std::endl;
*/
    services::internal::service_free<int, cpu>(mHist);
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status summarizationKernelImpl(MemoryCtxType<IdxType, DataType> & mem, TreeCtxType<IdxType, DataType> & qTree)
{
    const IdxType nThreads = threader_get_threads_number();
    IdxType nBlocks, lOffset, sizeOfBlock = 1;

    for (int l = 1; l < MAX_LEVEL + 1; l++)
    {
        nBlocks = qTree.layerSize[MAX_LEVEL - l];
        lOffset = qTree.layerOffs[MAX_LEVEL - l];
        if (nBlocks == 0) continue;
        daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
            //for(IdxType iBlock = 0; iBlock < nBlocks; iBlock++) {
            IdxType iPos = lOffset + iBlock;
            DataType cx, cy;
            if (qTree.tree[iPos].fpos < 0)
            {
                int pos = qTree.tree[iPos].fpos & ~0xE0000000;
                int cnt = 1 + ((qTree.tree[iPos].fpos >> 29) & 0x3);
                //std::cout << "Processing node with size = " << cnt << std::endl;
                cx = qTree.centx[pos];
                cy = qTree.centy[pos];
                for (int c = 1; c < cnt; c++)
                {
                    cx += qTree.centx[pos + c];
                    cy += qTree.centy[pos + c];
                }
            }
            else
            {
                //std::cout << "Processing leaf with size = " << qTree.tree[iBlock].cnt << std::endl;
                cx = mem.posx[mem.z_order_idx[qTree.tree[iPos].fpos]];
                cy = mem.posy[mem.z_order_idx[qTree.tree[iPos].fpos]];
                //std::cout << std::setfill('0') << std::setw(7) << mem.z_order_idx[qTree.tree[iPos].fpos] << std::endl;
                for (int c = 1; c < qTree.tree[iPos].cnt; c++)
                {
                    cx += mem.posx[mem.z_order_idx[qTree.tree[iPos].fpos + c]];
                    cy += mem.posy[mem.z_order_idx[qTree.tree[iPos].fpos + c]];
                    //std::cout << std::setfill('0') << std::setw(7) << mem.z_order_idx[qTree.tree[iPos].fpos+c] << std::endl;
                }
            }
            qTree.centx[iPos] = cx;
            qTree.centy[iPos] = cy;
            //};
        });
    }

    sizeOfBlock = services::internal::min<cpu, IdxType>(256, (qTree.size + nThreads - 1) / nThreads);
    nBlocks     = (qTree.size + sizeOfBlock - 1) / sizeOfBlock;

    //    for(IdxType iBlock = 0; iBlock < nBlocks; iBlock++) {
    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(qTree.size, iStart + sizeOfBlock);

        for (IdxType i = iStart; i < iEnd; i++)
        {
            DataType iMass = DataType(1) / qTree.tree[i].cnt;
            qTree.centx[i] *= iMass;
            qTree.centy[i] *= iMass;
        }
        //    };
    });
    /*
    std::cout << "Center of mass = " << qTree.centx[0] << ", " << qTree.centy[0] << std::endl;
    double ccx = 0.0, ccy = 0.0;
    for(int cc=0; cc<mem.capacity; cc++) {
        ccx += mem.posx[cc];
        ccy += mem.posy[cc];
    }
    ccx /= mem.capacity; ccy /= mem.capacity;
    std::cout << "Dummy mass = " << ccx << ", " << ccy << std::endl;
*/
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status repulsionKernelImpl(MemoryCtxType<IdxType, DataType> & mem, TreeCtxType<IdxType, DataType> & qTree, const DataType theta,
                                     const DataType eps, DataType & zNorm, const DataType & radius)
{
    const DataType epsInc = eps + DataType(1);

    SafeStatus safeStat;
    DataType dq[MAX_LEVEL];

    dq[0] = (radius * radius) / (theta * theta);
    for (auto i = 1; i < MAX_LEVEL; i++) dq[i] = dq[i - 1] * 0.25;
    for (auto i = 0; i < MAX_LEVEL; i++) dq[i] += epsInc;

    daal::StaticTlsSum<DataType, cpu> sumTlsData(1);

    //const IdxType nThreads    = threader_get_threads_number();
    const IdxType nThreads = sumTlsData.nthreads();
    ;
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (mem.capacity + nThreads - 1) / nThreads);
    const IdxType nBlocks     = (mem.capacity + sizeOfBlock - 1) / sizeOfBlock;

    IdxType * nStack = services::internal::service_malloc<IdxType, cpu>(nThreads * MAX_LEVEL * 4);
    int * nLevel     = services::internal::service_malloc<int, cpu>(nThreads * MAX_LEVEL * 4);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(mem.capacity, iStart + sizeOfBlock);

        for (IdxType k = iStart; k < iEnd; ++k)
        {
            const IdxType i   = mem.z_order_idx[k];
            const DataType px = mem.posx[i];
            const DataType py = mem.posy[i];

            DataType * lSum = sumTlsData.local(tid);
            DataType vx     = 0.;
            DataType vy     = 0.;

            IdxType * lStack = nStack + tid * MAX_LEVEL * 4;
            int * lLevel     = nLevel + tid * MAX_LEVEL * 4;

            int cnt = 1 + ((qTree.tree[0].fpos >> 29) & 0x3);
            ;
            int pos = qTree.tree[0].fpos & ~0xE0000000;
            int idx = 0;

            switch (cnt)
            {
            case 4: lStack[idx] = pos + 3; lLevel[idx++] = 1;
            case 3: lStack[idx] = pos + 2; lLevel[idx++] = 1;
            case 2: lStack[idx] = pos + 1; lLevel[idx++] = 1;
            default:
            case 1: lStack[idx] = pos; lLevel[idx++] = 1;
            }

            while (idx > 0)
            {
                idx--;
                DataType dx   = px - qTree.centx[lStack[idx]];
                DataType dy   = py - qTree.centy[lStack[idx]];
                DataType dxy1 = dx * dx + dy * dy + epsInc;
                int mass      = qTree.tree[lStack[idx]].cnt;
                int fpos      = qTree.tree[lStack[idx]].fpos;
                int level     = lLevel[idx];
                DataType tdist_2;

                if ((mass == 1) || (dxy1 >= dq[level]))
                {
                    // Distant node, use centroid to calculate force vectors
                    tdist_2 = mass / (dxy1 * dxy1);
                    lSum[0] += tdist_2 * dxy1;
                    vx += dx * tdist_2;
                    vy += dy * tdist_2;
                }
                else if (fpos < 0)
                {
                    // Intermediate node, add children to stack
                    cnt = 1 + ((fpos >> 29) & 0x3);
                    ;
                    pos = fpos & ~0xE0000000;

                    switch (cnt)
                    {
                    case 4: lStack[idx] = pos + 3; lLevel[idx++] = level + 1;
                    case 3: lStack[idx] = pos + 2; lLevel[idx++] = level + 1;
                    case 2: lStack[idx] = pos + 1; lLevel[idx++] = level + 1;
                    default:
                    case 1: lStack[idx] = pos; lLevel[idx++] = level + 1;
                    }
                }
                else
                {
                    // Leaf node, process all point separatly
                    for (int c = 0; c < mass; c++)
                    {
                        dx   = px - mem.posx[mem.z_order_idx[fpos + c]];
                        dy   = py - mem.posy[mem.z_order_idx[fpos + c]];
                        dxy1 = dx * dx + dy * dy + epsInc;

                        tdist_2 = 1.0 / dxy1;
                        lSum[0] += tdist_2;

                        tdist_2 *= tdist_2;
                        vx += dx * tdist_2;
                        vy += dy * tdist_2;
                    }
                }
            }

            mem.repx[i] = vx;
            mem.repy[i] = vy;
        }
    });

    zNorm = 0.;
    sumTlsData.reduceTo(&zNorm, 1);

    /*
    for(auto cc=0; cc<mem.capacity; cc++) {
        std::cout << mem.z_order_idx[cc] << ";";
        std::cout << mem.posx[cc] << ";";
        std::cout << mem.posy[cc] << ";";
        std::cout << mem.repx[cc] << ";";
        std::cout << mem.repy[cc] << std::endl;
    }
*/
    services::internal::service_free<int, cpu>(nLevel);
    services::internal::service_free<IdxType, cpu>(nStack);

    return services::Status();
}

template <bool DivComp, typename IdxType, typename DataType, daal::CpuType cpu>
services::Status attractiveKernelImpl(const DataType * val, const size_t * col, const size_t * row, const DataType * posx, const DataType * posy,
                                      DataType * attrx, DataType * attry, DataType & zNorm, DataType & divergence, const IdxType N, const IdxType nnz,
                                      const IdxType nElements, const DataType exaggeration)
{
    DAAL_CHECK_MALLOC(val);
    DAAL_CHECK_MALLOC(col);
    DAAL_CHECK_MALLOC(row);
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);
    DAAL_CHECK_MALLOC(attrx);
    DAAL_CHECK_MALLOC(attry);

    const DataType multiplier = exaggeration * DataType(zNorm);
    divergence                = 0.;

    daal::TlsSum<DataType, cpu> divTlsData(1);
    daal::tls<DataType *> logTlsData([=]() { return services::internal::service_scalable_calloc<DataType, cpu>(nElements); });

    const size_t nThreads    = threader_get_threads_number();
    const size_t sizeOfBlock = services::internal::min<cpu, size_t>(256, N / nThreads + 1);
    const size_t nBlocks     = (size_t)N / sizeOfBlock + !!(size_t(N) % sizeOfBlock);

    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t iStart = iBlock * sizeOfBlock;
        const size_t iEnd   = services::internal::min<cpu, size_t>(size_t(N), iStart + sizeOfBlock);
        DataType * logLocal = logTlsData.local();
        DataType * divLocal = divTlsData.local();
        for (size_t iRow = iStart; iRow < iEnd; ++iRow)
        {
            size_t iSize = 0;
            attrx[iRow]  = 0.0;
            attry[iRow]  = 0.0;
            for (size_t index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index)
            {
                const size_t iCol = col[index] - 1;

                const DataType y1d    = posx[iRow] - posx[iCol];
                const DataType y2d    = posy[iRow] - posy[iCol];
                const DataType sqDist = services::internal::max<cpu, DataType>(DataType(0), y1d * y1d + y2d * y2d); // To deal with NaNs
                const DataType PQ     = val[index] / (sqDist + 1.);

                // Apply forces
                attrx[iRow] += PQ * (posx[iRow] - posx[iCol]);
                attry[iRow] += PQ * (posy[iRow] - posy[iCol]);
                if (DivComp)
                {
                    logLocal[iSize++] = val[index] * multiplier * (1. + sqDist);
                }
            }
            if (DivComp)
            {
                Math<DataType, cpu>::vLog(iSize, logLocal, logLocal);
                size_t start = row[iRow] - 1;
                for (size_t index = 0; index < iSize; ++index)
                {
                    divLocal[0] += val[start + index] * logLocal[index];
                }
            }
        }
    });

    divTlsData.reduceTo(&divergence, 1);
    divergence *= exaggeration;
    logTlsData.reduce([&](DataType * buf) { services::internal::service_scalable_free<DataType, cpu>(buf); });

    //Find_Normalization
    zNorm = DataType(1) / zNorm;

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status integrationKernelImpl(const DataType eta, const DataType momentum, const DataType exaggeration, DataType * posx, DataType * posy,
                                       const DataType * attrx, const DataType * attry, const DataType * repx, const DataType * repy, DataType * gainx,
                                       DataType * gainy, DataType * oldForcex, DataType * oldForcey, DataType & gradNorm, const DataType & zNorm,
                                       const IdxType N, const IdxType & blockOfRows)
{
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);
    daal::StaticTlsSum<DataType, cpu> sumTlsData(1);
    gradNorm = 0.;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        DataType ux, uy, gx, gy;
        DataType * localSum = sumTlsData.local(tid);
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            const DataType dx = exaggeration * attrx[i] - zNorm * repx[i];
            const DataType dy = exaggeration * attry[i] - zNorm * repy[i];
            localSum[0] += dx * dx + dy * dy;

            gx = (dx * (ux = oldForcex[i]) < DataType(0)) ? gainx[i] + 0.2 : gainx[i] * 0.8;
            if (gx < 0.01) gx = 0.01;

            gy = (dy * (uy = oldForcey[i]) < DataType(0)) ? gainy[i] + 0.2 : gainy[i] * 0.8;
            if (gy < 0.01) gy = 0.01;

            gainx[i] = gx;
            gainy[i] = gy;

            oldForcex[i] = ux = momentum * ux - 4. * eta * gx * dx;
            oldForcey[i] = uy = momentum * uy - 4. * eta * gy * dy;

            posx[i] += ux;
            posy[i] += uy;
        }
    });
    sumTlsData.reduceTo(&gradNorm, 1);
    gradNorm = Math<DataType, cpu>::sSqrt(gradNorm);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status tsneGradientDescentImpl(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                         const NumericTablePtr paramTable, const NumericTablePtr resultTable)
{
    // sizes and number of iterations
    daal::internal::ReadColumns<IdxType, cpu> sizeIterDataBlock(*sizeIterTable, 0, 0, sizeIterTable->getNumberOfRows());
    const IdxType * sizeIter = sizeIterDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(sizeIterDataBlock);
    DAAL_CHECK(sizeIterTable->getNumberOfRows() == 4, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    const IdxType N                    = sizeIter[0]; // Number of points
    const IdxType nnz                  = sizeIter[1]; // Number of elements in sparce matrix P
    const IdxType nIterWithoutProgress = sizeIter[2]; // Number of iterations without introducing changes
    const IdxType maxIter              = sizeIter[3]; // Number of iterations
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, 2, N);
    const IdxType nNodes          = N <= 50 ? 4 * N : 2 * N; // A small number of points may require more memory to store tree nodes
    const IdxType nIterCheck      = 50;
    const IdxType explorationIter = 250; // Aligned with scikit-learn
    const IdxType blockOfRows     = 256;

    // parameters
    daal::internal::ReadColumns<DataType, cpu> paramDataBlock(*paramTable, 0, 0, paramTable->getNumberOfRows());
    const DataType * params = paramDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(paramDataBlock);
    DAAL_CHECK(paramTable->getNumberOfRows() == 4, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    const DataType eps = 0.000001; // A tiny jitter to promote numerical stability
    DataType momentum  = 0.5;      // The momentum used during the exaggeration phase. Aligned with scikit-learn
    DataType exaggeration =
        params[0]; // How much pressure to apply to clusters to spread out during the exaggeration phase. Aligned with scikit-learn
    const DataType eta         = params[1]; // Learning rate. Aligned with scikit-learn
    const DataType minGradNorm = params[2]; // The smallest gradient norm TSNE should terminate on
    const DataType theta       = params[3]; // is the angular size of a distant node as measured from a point. Tradeoff for speed (0) vs accuracy (1)

    // results
    daal::internal::WriteColumns<DataType, cpu> resultDataBlock(*resultTable, 0, 0, resultTable->getNumberOfRows());
    DataType * results = resultDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(resultDataBlock);
    DAAL_CHECK(resultTable->getNumberOfRows() == 3, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    DataType & curIter    = results[0];
    DataType & divergence = results[1];
    DataType & gradNorm   = results[2];

    // internal values
    services::Status status;
    IdxType maxDepth  = 1;
    IdxType bottom    = nNodes;
    IdxType nElements = 0;
    IdxType bestIter  = 0;

    DataType radius  = 0.;
    DataType centerx = 0.;
    DataType centery = 0.;

    DataType zNorm          = 0.;
    DataType bestDivergence = daal::services::internal::MaxVal<DataType>::get();

    // daal checks
    DAAL_CHECK(initTable->getNumberOfRows() == N, daal::services::ErrorInconsistentNumberOfRows);
    DAAL_CHECK(initTable->getNumberOfColumns() == 2, daal::services::ErrorInconsistentNumberOfColumns);

    daal::internal::WriteColumns<DataType, cpu> xInitDataBlock(*initTable, 0, 0, N);
    daal::internal::WriteColumns<DataType, cpu> yInitDataBlock(*initTable, 1, 0, N);
    DataType * xInit = xInitDataBlock.get();
    DataType * yInit = yInitDataBlock.get();
    DAAL_CHECK_MALLOC(xInit);
    DAAL_CHECK_MALLOC(yInit);

    CSRBlockDescriptor<DataType> CSRBlock;
    status = pTable->getSparseBlock(0, N, readOnly, CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);
    DataType * val = CSRBlock.getBlockValuesPtr();
    size_t * col   = CSRBlock.getBlockColumnIndicesPtr();
    size_t * row   = CSRBlock.getBlockRowIndicesPtr();

    MemoryCtxType<IdxType, DataType> mem;

    // allocate and init memory for auxiliary arrays: posx & posy, morton codes and indices
    mem.capacity = N;

    mem.posx = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.posx);
    mem.posy = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.posy);
    mem.morton_codes = services::internal::service_malloc<uint64_t, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.morton_codes);
    mem.z_order_idx = services::internal::service_malloc<int, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.z_order_idx);
    mem.t_order_idx = services::internal::service_malloc<int, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.t_order_idx);
    mem.repx = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.repx);
    mem.repy = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.repx);
    mem.attrx = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.attrx);
    mem.attry = services::internal::service_malloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.attry);
    mem.gainx = services::internal::service_calloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.gainx);
    mem.gainy = services::internal::service_calloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.gainy);
    mem.oforx = services::internal::service_calloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.oforx);
    mem.ofory = services::internal::service_calloc<DataType, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.ofory);

    services::internal::tmemcpy<DataType, cpu>(mem.posx, xInit, N);
    services::internal::tmemcpy<DataType, cpu>(mem.posy, yInit, N);

    TreeCtxType<IdxType, DataType> qTree;
    // allocate enough memory to store top 5 levels of qTree
    qTree.capacity = 1024;
    qTree.tree     = services::internal::service_malloc<qTreeNode, cpu>(qTree.capacity);
    DAAL_CHECK_MALLOC(qTree.tree);
    qTree.centx = services::internal::service_malloc<DataType, cpu>(qTree.capacity);
    DAAL_CHECK_MALLOC(qTree.centx);
    qTree.centy = services::internal::service_malloc<DataType, cpu>(qTree.capacity);
    DAAL_CHECK_MALLOC(qTree.centy);

    status = maxRowElementsImpl<IdxType, cpu>(row, N, nElements, blockOfRows);
    DAAL_CHECK_STATUS_VAR(status);

    gradNorm = 0.0;

    //start iterations
    for (IdxType i = 0; i < explorationIter; ++i)
    {
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(mem.posx, mem.posy, N, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(mem, qTree, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(mem, qTree);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(mem, qTree, theta, eps, zNorm, radius);
        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, mem.posx, mem.posy, mem.attrx, mem.attry, zNorm, divergence, N,
                                                                        nnz, nElements, exaggeration);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, mem.posx, mem.posy, mem.attrx, mem.attry, zNorm, divergence,
                                                                         N, nnz, nElements, exaggeration);
        }
        DAAL_CHECK_STATUS_VAR(status);

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, mem.posx, mem.posy, mem.attrx, mem.attry, mem.repx,
                                                               mem.repy, mem.gainx, mem.gainy, mem.oforx, mem.ofory, gradNorm, zNorm, N, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

        if ((i + 1) % nIterCheck == 0)
        {
            if (divergence < bestDivergence)
            {
                bestDivergence = divergence;
                bestIter       = i;
            }

            if (gradNorm <= minGradNorm)
            {
                curIter = i;
                break;
            }

            curIter = i;
        }
    }

    momentum     = 0.8;
    exaggeration = 1.;

    for (IdxType i = explorationIter; i < maxIter; ++i)
    {
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(mem.posx, mem.posy, N, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(mem, qTree, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(mem, qTree);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(mem, qTree, theta, eps, zNorm, radius);
        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, mem.posx, mem.posy, mem.attrx, mem.attry, zNorm, divergence, N,
                                                                        nnz, nElements, exaggeration);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, mem.posx, mem.posy, mem.attrx, mem.attry, zNorm, divergence,
                                                                         N, nnz, nElements, exaggeration);
        }
        DAAL_CHECK_STATUS_VAR(status);

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, mem.posx, mem.posy, mem.attrx, mem.attry, mem.repx,
                                                               mem.repy, mem.gainx, mem.gainy, mem.oforx, mem.ofory, gradNorm, zNorm, N, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == maxIter - 1))
        {
            if (divergence < bestDivergence)
            {
                bestDivergence = divergence;
                bestIter       = i;
            }

            else if (i - bestIter > nIterWithoutProgress)
            {
                curIter = i;
                break;
            }

            if (gradNorm <= minGradNorm)
            {
                curIter = i;
                break;
            }

            curIter = i;
        }
    }

    // printf("     Divergence = %f", divergence);
    // printf("     Gradient norm = %f", gradNorm);
    // printf("     Last iteration = %f", curIter);

    //save results
    services::internal::tmemcpy<DataType, cpu>(xInit, mem.posx, N);
    services::internal::tmemcpy<DataType, cpu>(yInit, mem.posy, N);

    //release block
    status = pTable->releaseSparseBlock(CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);

    services::internal::service_free<qTreeNode, cpu>(qTree.tree);
    services::internal::service_free<DataType, cpu>(qTree.centx);
    services::internal::service_free<DataType, cpu>(qTree.centy);
    services::internal::service_free<int, cpu>(mem.t_order_idx);
    services::internal::service_free<int, cpu>(mem.z_order_idx);
    services::internal::service_free<uint64_t, cpu>(mem.morton_codes);
    services::internal::service_free<DataType, cpu>(mem.posy);
    services::internal::service_free<DataType, cpu>(mem.posx);
    services::internal::service_free<DataType, cpu>(mem.repx);
    services::internal::service_free<DataType, cpu>(mem.repy);

    return services::Status();
}

template <typename algorithmIdxType, typename algorithmFPType>
DAAL_EXPORT void tsneGradientDescent(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                     const NumericTablePtr paramTable, const NumericTablePtr resultTable)
{
#define DAAL_TSNE_GRADIENT_DESCENT(cpuId, ...) tsneGradientDescentImpl<algorithmIdxType, algorithmFPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_TSNE_GRADIENT_DESCENT, initTable, pTable, sizeIterTable, paramTable, resultTable);

#undef DAAL_TSNE_GRADIENT_DESCENT
}

template DAAL_EXPORT void tsneGradientDescent<int, DAAL_FPTYPE>(const NumericTablePtr initTable, const CSRNumericTablePtr pTable,
                                                                const NumericTablePtr sizeIterTable, const NumericTablePtr paramTable,
                                                                const NumericTablePtr resultTable);

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif