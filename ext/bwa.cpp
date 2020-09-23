/*************************************************************************************
MIT License

Copyright (c) 2020 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Sanchit Misra <sanchit.misra@intel.com>; Vasimuddin Md <vasimuddin.md@intel.com>
*****************************************************************************************/

#include <string.h>
#include <stdio.h>
#include <zlib.h>
#include <assert.h>
#include "bntseq.h"
#include "bwa.h"
#include "utils.h"
#include "kstring.h"
//#include "kvec.h"
#include <string>

int bwa_verbose = 3;

/************************
 * Batch FASTA/Q reader *
 ************************/

#include "kseq.h"
KSEQ_DECLARE(gzFile)

static inline void trim_readno(kstring_t *s)
{
    if (s->l > 2 && s->s[s->l-2] == '/' && isdigit(s->s[s->l-1]))
        s->l -= 2, s->s[s->l] = 0;
}

static inline void kseq2bseq1(const kseq_t *ks, bseq1_t *s)
{ // TODO: it would be better to allocate one chunk of memory, but probably it does not matter in practice
    s->name = strdup(ks->name.s);
    s->comment = ks->comment.l? strdup(ks->comment.s) : 0;
    s->seq = strdup(ks->seq.s);
    assert(s->seq != NULL);
    s->qual = ks->qual.l? strdup(ks->qual.s) : 0;
    if(ks->qual.l)
        assert(s->qual != NULL);
    s->l_seq = strlen(s->seq);
}

/* Customized for MPI processing */
bseq1_t *bseq_read(int64_t chunk_size, int *n_, void *ks1_, void *ks2_,
                   FILE* fpp, int len, int64_t *s)
{
    kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;
    int64_t size = 0, m, n, size2 = 0;
    bseq1_t *seqs;
    m = n = 0; seqs = 0;
    char buf[len];
    
    while (kseq_read(ks) >= 0)
    {
        if (ks2 && kseq_read(ks2) < 0) { // the 2nd file has fewer reads
            fprintf(stderr, "[W::%s] the 2nd file has fewer sequences.\n", __func__);
            break;
        }
        if (n >= m) {
            m = m? m<<1 : 256;
            seqs = (bseq1_t*) realloc(seqs, m * sizeof(bseq1_t));
            assert(seqs != NULL);
        }
        trim_readno(&ks->name);
        kseq2bseq1(ks, &seqs[n]);
        seqs[n].id = n;
        {
            //kseq_t *ksd = ks;
            //kstream_t *kst = ksd->f;
#if 0
            //printf("Check D..\n%s\n%s\n%s\n%s\n",
            //     seqs[n].name, seqs[n].seq,
            //     seqs[n].comment, seqs[n].qual);
            
            if (seqs[n].name != NULL)
                size += strlen(seqs[n].name);
            //printf("%d ", strlen(seqs[n].name)+strlen(seqs[n].comment)+1);
            if (seqs[n].comment != NULL) {
                size += strlen(seqs[n].comment);
                std::string str = seqs[n].comment;
                std::size_t found = str.find("length");
                if (found != std::string::npos) {
                    size += strlen(seqs[n].comment) + strlen(seqs[n].name) + 1;
                }
            }
            else
                size += 1;
            
            if (seqs[n].qual != NULL)
                size += strlen(seqs[n].qual);

            size += 7; // non accounted chars
#else
            //kstring_t kstr;
            //printf("%d\n", ks_getuntil2(kst, KS_SEP_LINE, &kstr, 0, 0));
            err_fgets((char*) buf, len, fpp);
            size2 += strlen((char*) buf);
            // printf("First line: %d, %s\n", strlen(buf), buf);
            err_fgets((char*) buf, len, fpp);
            size2 += strlen((char*) buf);
            if (seqs[n].qual != NULL) {
                err_fgets((char*) buf, len, fpp);
                size2 += strlen((char*) buf);
                err_fgets((char*) buf, len, fpp);
                size2 += strlen((char*) buf);
            }
#endif
        }
        //size += seqs[n++].l_seq;
        size = size2;       n++;
        
        //printf("size: %d, size2: %d\n", size, size2);
        //static int cnt = 0;
        //if (cnt++ == 4)exit(0);
        
        if (ks2) {
            trim_readno(&ks2->name);
            kseq2bseq1(ks2, &seqs[n]);
            seqs[n].id = n;
            n++;
            // size += seqs[n++].l_seq;
        }
        //if (size >= chunk_size && (n&1) == 0) break;
        if (size >= chunk_size) {
            break;
        }
    }
    if (size == 0) { // test if the 2nd file is finished
        if (ks2 && kseq_read(ks2) >= 0)
            fprintf(stderr, "[W::%s] the 1st file has fewer sequences.\n", __func__);
    }
    *n_ = n;
    *s = size;
    return seqs;
}

bseq1_t *bseq_read_orig(int64_t chunk_size, int *n_, void *ks1_, void *ks2_, int64_t *s)
{
    kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;
    int64_t size = 0, m, n;
    bseq1_t *seqs;
    m = n = 0; seqs = 0;
    while (kseq_read(ks) >= 0)
    {
        if (ks2 && kseq_read(ks2) < 0) { // the 2nd file has fewer reads
            fprintf(stderr, "[W::%s] the 2nd file has fewer sequences.\n", __func__);
            break;
        }
        if (n >= m) {
            m = m? m<<1 : 256;
            seqs = (bseq1_t*) realloc(seqs, m * sizeof(bseq1_t));
            assert(seqs != NULL);
        }
        trim_readno(&ks->name);
        kseq2bseq1(ks, &seqs[n]);
        seqs[n].id = n;
        //{
        //  size += strlen(seqs[n].name);
        //  size += strlen(seqs[n].comment);
        //  size += strlen(seqs[n].qual);
        //  // fprintf(stderr, "qual len: %d %d\n", strlen(seqs[n].qual), seqs[n].l_seq);
        //  size += 7; // non accounted chars
        //}
        size += seqs[n++].l_seq;

        if (ks2) {
            trim_readno(&ks2->name);
            kseq2bseq1(ks2, &seqs[n]);
            seqs[n].id = n;
            size += seqs[n++].l_seq;
        }
        if (size >= chunk_size && (n&1) == 0) break;
        // if (size >= chunk_size) {
        //  break;
        // }
    }
    if (size == 0) { // test if the 2nd file is finished
        if (ks2 && kseq_read(ks2) >= 0)
            fprintf(stderr, "[W::%s] the 1st file has fewer sequences.\n", __func__);
    }
    *n_ = n;
    *s = size;
    return seqs;
}

bseq1_t *bseq_read_one_fasta_file(int64_t chunk_size, int *n_, gzFile fp, int64_t *s)
{
    kseq_t *ks = kseq_init(fp);
    bseq1_t *seq = bseq_read_orig(chunk_size, n_, ks, NULL, s);
    kseq_destroy(ks);
    return seq;
}

void bseq_destroy(bseq1_t *s)
{
    if(s)
    {
        if(s->name) free(s->name);
        if(s->comment) free(s->comment);
        if(s->seq) free(s->seq);
        if(s->qual) free(s->qual);
        if(s->sam) free(s->sam);
        free(s);
    }
}
