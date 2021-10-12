

for seed in {21..30}
do

for base in glove fast google
do

echo seed $seed :training LSI...
python imputation_lsi.py $base bioword 8 ./data/sampled_20000/LSI_embeds $seed 20000    

echo seed $seed :training GCN...
python train_gcn.py --base=$base --aff=bioword --seed=$seed --corpus_size=20000


for p in 0 2 5 10
do

    echo seed: $seed , p: $p : training APPNP...
    python train_appnp.py --base=$base --aff=bioword --p=$p --seed=$seed --corpus_size=20000
    
done


done
done