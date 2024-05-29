#!/bin/bash
set -e
#sorted array file full path
A=$1
# RMI file full path
B=$2
# number of leaf nodes
m=$3
# data type of rmi key [F64, UINT64]
T=$4
echo "$A"
echo "$B"
echo "$m"

# Hard coded: Need to fix
CXX=g++

#cd build-rmi/learned-systems-rmi
cd  ext/build-rmi/learned-systems-rmi
[ -f sorted_doubles_rmi.cpp ] && rm sorted_doubles_rmi.cpp
[ -f sorted_doubles_rmi.h ] && rm sorted_doubles_rmi.h
[ -f sorted_doubles_rmi_data.h ] && rm sorted_doubles_rmi_data.h

time cargo run --release -- $A sorted_doubles_rmi linear_spline,linear $m
cd ..

echo "cargo done.."
[ -f sorted_doubles_rmi.cpp ] && rm sorted_doubles_rmi.cpp
[ -f sorted_doubles_rmi.h ] && rm sorted_doubles_rmi.h
[ -f sorted_doubles_rmi_data.h ] && rm sorted_doubles_rmi_data.h
cp learned-systems-rmi/sorted_doubles_rmi.cpp .
cp learned-systems-rmi/sorted_doubles_rmi.h .
cp learned-systems-rmi/sorted_doubles_rmi_data.h .
bash ./modify_generated_code.sh sorted_doubles_rmi $m

[ -f rmi-minimizer ] && rm rmi-minimizer
${CXX} rmi-main.cpp sorted_doubles_rmi.cpp -D$T -o rmi-minimizer
time ./rmi-minimizer $A $B > out
grep avg_log2_err out
echo "Built RMI"
[ -f rmi-minimizer ] && rm rmi-minimizer
[ -f out ] && rm out
[ -f sorted_doubles_rmi.cpp ] && rm sorted_doubles_rmi.cpp
[ -f sorted_doubles_rmi.h ] && rm sorted_doubles_rmi.h
[ -f sorted_doubles_rmi_data.h ] && rm sorted_doubles_rmi_data.h
echo "Cleaning done."
