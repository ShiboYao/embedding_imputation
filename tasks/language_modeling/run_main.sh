for seed in {21..30}
do

for base in fast #glove fast google
do

echo BASE: seed = $seed , base = $base
python main.py --embed_path=data/sampled_20000/${base}_embeds.txt --base=${base} --cuda

echo LSI: seed = $seed , base = $base
python main.py --embed_path=data/sampled_20000/LSI_embeds/LSI_${base}_embeds_8_$seed.txt --base=$base --cuda

echo GCN: seed = $seed , base = $base
python main.py --embed_path=data/sampled_20000/GCN_embeds/GCN_${base}_embeds_8_$seed.txt --base=$base --cuda

for p in 0 2 5 10
    do
        echo APPNP $p, seed =$seed, base = $base
        python main.py --embed_path=data/sampled_20000/APPNP_embeds/APPNP_${base}_embeds_8_0.1_${p}_${seed}.txt --base=${base} \
        --cuda
    done
done
done
