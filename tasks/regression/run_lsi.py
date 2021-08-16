for delta in 5 10 20
do
	echo "python imputation.py google glove $delta"
	for i in {1..3}
	do
		python imputation.py google glove $delta $i
	done
	
	echo "python imputation.py google fast $delta"
	for i in {1..3}
	do
		python imputation.py google fast $delta $i
	done
	
	echo "python imputation.py glove google $delta"
	for i in {1..3}
	do
		python imputation.py glove google $delta $i
	done
	
	echo "python imputation.py glove fast $delta"
	for i in {1..3}
	do
		python imputation.py glove fast $delta $i
	done

	echo "python imputation.py fast google $delta"
	for i in {1..3}
	do
		python imputation.py fast google $delta $i
	done
	
	echo "python imputation.py fast glove $delta"
	for i in {1..3}
	do
		python imputation.py fast glove $delta $i
	done
done
