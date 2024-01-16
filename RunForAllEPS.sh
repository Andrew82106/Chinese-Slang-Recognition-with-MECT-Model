for eps in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 2.1
do
    python cluster.py --mode clean_model_cache
    python cluster.py --mode test_dimension_decline --eps="$eps"
    python cluster.py --mode test
    echo "eps=[$eps]"
done
