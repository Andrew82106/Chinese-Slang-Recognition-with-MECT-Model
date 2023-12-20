python main.py --dataset msra --status generate --device cpu --extra_datasets wiki
python cluster.py --mode generate
python cluster.py --mode test