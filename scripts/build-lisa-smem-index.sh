make clean
make
make lisa

full_path=`readlink -f $1`
full_path=$1

K=$2
num_leaf=$3

#build rmi index -rev-comp and interval tree index
./build-index-with-rev-complement-lisa.o ${full_path} $K ${num_leaf}
#build interval tree index 
#/usr/bin/time -v ./build-index-with-rev-complement-lisa.o ${full_path} $K ${num_leaf}
