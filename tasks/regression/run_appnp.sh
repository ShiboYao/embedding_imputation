for delta in 5 10 20
do
        echo "python train.py --base=w2v --aff=glove --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=w2v --aff=glove --delta=$delta --seed=$i
        done

        echo "python train.py --base=w2v --aff=fast --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=w2v --aff=fast --delta=$delta --seed=$i
        done

        echo "python train.py --base=glove --aff=w2v --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=glove --aff=w2v --delta=$delta --seed=$i
        done

        echo "python train.py --base=glove --aff=fast --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=glove --aff=fast --delta=$delta --seed=$i
        done

        echo "python train.py --base=fast --aff=w2v --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=fast --aff=w2v --delta=$delta --seed=$i
        done

        echo "python train.py --base=fast --aff=glove --delta=$delta"
        for i in {1..3}
        do
                python train.py --base=fast --aff=glove --delta=$delta --seed=$i
        done
done
