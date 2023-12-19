python main.py --dataset msra --status generate --device cpu --extra_datasets wiki
python cluster.py
cd Utils
python evaluateCluster.py