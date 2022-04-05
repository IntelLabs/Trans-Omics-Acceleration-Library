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

Authors: Saurabh Kalikar <saurabh.kalikar@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
*****************************************************************************************/
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif

#include<fstream>
#include <cstring>
#include "LISA_search.h"
#include "read.h"
#include <immintrin.h>
#include "sais.h"
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <string.h>
#include <zlib.h>
#include "bntseq.h"
#include <omp.h>

int main(int argc, char** argv) {
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
    

    if(!(argc == 4)) {
        error_quit("Need 3 args: ref_file K num_rmi_leaf_nodes");
    }

    int K = atoi(argv[2]);
    eprintln("using K = %d", K);

    int64_t num_rmi_leaf_nodes = atol(argv[3]);
    eprintln("using num_rmi_leaf_nodes = %ld", num_rmi_leaf_nodes);

    string seq;// = read_seq(argv[1]);
    const char *ref_seq_file = argv[1]; 
    gzFile fp = xzopen(ref_seq_file, "r");
    bns_fasta2bntseq(fp, ref_seq_file, 1);
    read_seq_lisa(argv[1], seq);
    eprintln("Read ref file done.");
    eprintln("seq.size() = %lu", seq.size());

    string seq_forward_only = seq;
#ifdef REV_COMP
    // appending reverse complement
    for(int64_t i=(int64_t)seq.size()-1-(seq.back()=='@');i>=0;i--) {
//#ifndef NO_DNA_ORD 
//        seq.push_back(dna[3-dna_ord(seq[i])]);
//#else 
        seq.push_back(dna[3-(__lg(seq[i]-'A'+2)-1)]);
//#endif 
    }
#endif 

    seq.push_back('$');
    string ref_seq_filename = argv[1];
#ifdef REV_COMP
    //string rmi_filename = ref_seq_filename + ".qbwt4.walg.rev_comp";
    string rmi_filename = ref_seq_filename + ".rev_comp";
#else
    //string rmi_filename = ref_seq_filename + ".qbwt4.walg";
    string rmi_filename = ref_seq_filename;
#endif


   IPBWT_RMI<int64_t, uint64_t> *rmi =  new IPBWT_RMI<int64_t, uint64_t>(seq, seq.size(), rmi_filename, K, num_rmi_leaf_nodes, NULL);

  delete rmi;
#ifdef REV_COMP
{ 
   LISA_search<int64_t> qbwt(seq_forward_only, seq_forward_only.size(), argv[1], K, num_rmi_leaf_nodes);
}

//   string size_file_name = (string) argv[1] + "_SIZE";
//   ofstream f_sz(size_file_name.c_str());
//   f_sz<<(seq_forward_only.size());
#endif
   return 0;
}

