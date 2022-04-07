# Input file format
# Line 1: Path to referene sequence name
# Line 2: Number of read datasets used for SMEM computation
# Line 3: Path to read dataset 1 
# Line 4: Path to read dataset 2 
# Line 5: Path to read dataset 3 and so on
# Example

#  /ref_datasets/human.fa
#  3
#  /read_data/H1.151.fastq
#  /read_data/H2.151.fastq
#  /read_data/H3.151.fastq
#  /ref_datasets/asian_rice.fa
#  2
#  /read_data/A1.151.fastq
#  /read_data/A2.151.fastq


 
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
	#echo "Number of read datasets: "${data[$i]}
    num_reads=${data[$i]}
	i=`expr $i + 1`
	#echo "--"$i
    for (( j = $i ; j < $i+${num_reads}; j++ )) ### Inner for loop ###
    do
	#	echo "J:"$j
		echo "------------ read dataset: "${data[$j]}" -----------"


		R1=${data[$j]}

		echo "$ref -- ${R1}"
		numactl -N 1 -m 1 ./bench-smem ${ref} ${R1} 50 19 56 >output_fmi_smem 2>fmi_smem_log

		sudo /etc/pcl_manage_ram.sh 2m 85000 

		numactl -N 1 -m 1 ./smem-lisa.o ${ref} ${R1} 20 0 56 19 >output_lisa_smem 2>lisa_smem_log            

		sudo /etc/pcl_manage_ram.sh 2m 000 

		echo "******************* DIFF: smem search ***********************"
		diff output_fmi_smem output_lisa_smem | wc -l 
		ls -lh output_fmi_smem output_lisa_smem
		rm output_fmi_smem output_lisa_smem
		
		fmi_ticks=`grep "Consumed" fmi_smem_log | head -n 1 |cut -d " " -f2`
		lisa_ticks=`grep "totalTicks" lisa_smem_log | cut -d " " -f3`

		echo "FMI ticks: $fmi_ticks. LISA ticks: $lisa_ticks"
		echo "SMEM kernel speedup:"
		echo "scale=2; ${fmi_ticks} / ${lisa_ticks}" | bc
		rm lisa_smem_log fmi_smem_log
    done
	i=`expr $i + ${num_reads} - 1`
  echo "" #### print the new line ###
done


