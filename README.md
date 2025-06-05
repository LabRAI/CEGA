# CEGA
This is the open-source code for ICML 2025 paper CEGA: A Cost-Effective Approach for Graph-Based Model Extraction and Acquisition.


## **Setup Environment**
```bash
conda create --name my_env --file requirements.txt
```

## **Running the Experiments**
To run a single experiment, use the command:
```
python main.py --dataset amazoncomputer --method random
```

To run ALL attack experiments, use the  command:
```bash
bash run_bash.sh
```

The outputs of the experiment are recorded in a timestamped directory under `./output/`. 
```
./output/
    âââ {timestamp}/
        âââ {dataset}/
        âââ log/{dataset}/
        âââ settings.txt
```

## **Visualize Results**
Results can be visualized by using `./plots/model_performance.ipynb`. Before running it, update the results folder paths:
```
datasets = [
    '{timestamp}/{dataset}'
]
cega_results = [
    '{timestamp}/{dataset}'
]
```

