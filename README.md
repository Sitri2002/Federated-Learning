<h1>Federated Learning Experiments</h1>
<h2>Baseline</h2>
Run main.py to load CIFAR-10, run mimic.py to load custom MIMIC-CXR dataset and run on ResNet50.
<h2>FedML</h2>
Run main_fedml_mimic.py to run the simulation. Use config/fedml_config.yaml to set hyperparameters.
<h2>FLOWER</h2>
cd into flower-fedavg, in CLI use "flwr run" to run simulation, use pyproject.toml to set hyperparameters.
<br>
Both FedML and FLOWER has been configured to run with MIMIC. For FedML you need to load custom CIFAR-10 dataset from HuggingFace/FLOWER datasets and change to label count from 14 to 10. Same with FLOWER.
