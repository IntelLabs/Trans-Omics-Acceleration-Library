# Input file format
# Line 1: Path to referene sequence name
# Line 2: Number of read datasets used for exact search computation
# Line 3: Path to seed dataset 1 
# Line 4: Path to seed dataset 2 
# Line 5: Path to seed dataset 3 and so on
# Example

#  /ref_datasets/human.fa
#  3
#  /read_data/S1.21.fastq
#  /read_data/S2.21.fastq
#  /read_data/S3.21.fastq
#  /ref_datasets/asian_rice.fa
#  2
#  /read_data/AS1.21.fastq
#  /read_data/AS2.21.fastq


 
data=()
filename=$1
while read -r line; do
    data+=($line)
done < $filename


arr=()

isz=${#data[@]}

for (( i = 0; i < $isz; i++ ))      ### Outer for loop ###
do
	#echo $i
	echo "******************* ref seq: "${data[$i]}" ***********"
	ref=${data[$i]}
	i=`expr $i + 1`
	#echo "Number of seed datasets: "${data[$i]}
    num_reads=${data[$i]}
	i=`expr $i + 1`
	#echo "--"$i
    for (( j = $i ; j < $i+${num_reads}; j++ )) ### Inner for loop ###
    do
	#	echo "J:"$j
		echo "------------ read dataset: "${data[$j]}" -----------"


		S1=${data[$j]}

		echo "$ref -- ${S1}"
    
		numactl -N 1 -m 1 ./bench-fixed-len-e2e-match ${ref} ${S1} 0 20 50 >output_fmi_exact_search 2>fmi_exact_log
		sudo /etc/pcl_manage_ram.sh 2m 85000 
		numactl -N 1 -m 1 ./exact-search-lisa.o ${ref} ${S1} 21 0 20 >output_lisa_exact_search 2>lisa_exact_log
		sudo /etc/pcl_manage_ram.sh 2m 000 

		echo "******************* DIFF: Exact search ***********************"
		diff output_fmi_exact_search output_lisa_exact_search | wc -l 
		ls -lh output_fmi_exact_search output_lisa_exact_search

		fmi_ticks=`grep "Consumed" fmi_exact_log | head -n 1 | cut -d " " -f2`
		lisa_ticks=`grep "totalTicks" lisa_exact_log | cut -d " " -f3`

		echo "FMI ticks: $fmi_ticks. LISA ticks: $lisa_ticks"
		echo "Exact search kernel speedup:"
		echo "scale=2; ${fmi_ticks} / ${lisa_ticks}" | bc


	done
	i=`expr $i + ${num_reads} - 1`
  echo "" #### print the new line ###
done


