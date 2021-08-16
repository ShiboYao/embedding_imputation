for delta in 5 10 20
do
    for b in google glove fast
    do
        echo python train.py --base=$b --delta=$delta
        for i in {1..10}
        do
            python train.py --base=$b --delta=$delta --seed=$i
        done
    done
done
