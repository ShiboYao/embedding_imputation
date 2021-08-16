for delta in 5 10 20
do
    for b in google glove fast
    do
        echo python class_lsi.py $b aff $delta
        for i in {1..10}
        do
            python class_lsi.py $b aff $delta $i
        done
    done
done
