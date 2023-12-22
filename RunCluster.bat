python cluster.py --mode clean_function_cache
python cluster.py --mode clean_model_cache
python main.py --dataset msra --status generate --device cpu --extra_datasets wiki
python cluster.py --mode generate
python cluster.py --mode test