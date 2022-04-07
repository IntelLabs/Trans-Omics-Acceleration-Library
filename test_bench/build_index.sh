ref=$1
make clean 
make huge_page=1 -s
echo "building done"

rm ${ref}.*
rm ${ref}_SIZE
echo "Building FM Index with REV-COMP strand"
/bin/time -v ./build-index-with-rev-complement ${ref} &> fm_index_rev_comp_log 
echo "Building FM Index with FORWARD-ONLY strand"
/bin/time -v ./build-index-forward-only ${ref} &> fm_index_forward_only_log

echo "Building LISA Index with REV-COMP strand with default number of rmi leaf nodes"
/bin/time -v ./build-index-with-rev-complement-lisa.o ${ref} 20 0 &>lisa_index_rev_comp_log
echo "Building LISA Index with FORWARD-ONLY strand with default number of rmi leaf nodes"
./build-index-forward-only-lisa.o /cold_storage/omics/saurabh/temp_ref/asian-rice.fa 21 0 &>lisa_index_forward_only_log


