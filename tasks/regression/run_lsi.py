for delta in 5 10 20
do
	echo "python imputation.py w2v glove $delta"
	for i in {1..3}
	do
		python imputation.py w2v glove $delta $i
	done
	
	echo "python imputation.py w2v fast $delta"
	for i in {1..3}
	do
		python imputation.py w2v fast $delta $i
	done
	
	echo "python imputation.py glove w2v $delta"
	for i in {1..3}
	do
		python imputation.py glove w2v $delta $i
	done
	
	echo "python imputation.py glove fast $delta"
	for i in {1..3}
	do
		python imputation.py glove fast $delta $i
	done

	echo "python imputation.py fast w2v $delta"
	for i in {1..3}
	do
		python imputation.py fast w2v $delta $i
	done
	
	echo "python imputation.py fast glove $delta"
	for i in {1..3}
	do
		python imputation.py fast glove $delta $i
	done
done
