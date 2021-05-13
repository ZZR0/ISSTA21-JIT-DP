## General Introduction
This repo illustrates how to evaluate the artifacts in the paper *Deep Just-in-Time Defect Prediction: How Far Are We?* published in ISSTA'21. Specifically, we first replicate the experiments on DeepJIT and CC2Vec under exactly the same setting and dataset of their original papers. We also build an extensive benchmark with over 310K commits in this paper. Accordingly, we further investigate how DeepJIT and CC2Vec perform under within-project and cross-project evaluation on the extensive benchmark. Next, we investigate the performance of various representative traditional features for JIT defect prediction. At last, we propose a simplistic “la” (lines added)-feature-based JIT defect prediction approach, namely *LApredict* which turns out to outperform DeepJIT and CC2Vec.   

Our original evaluation in the paper includes data processing and model training and testing. However, data processing and model training can be rather time-consuming. To facilitate the artifact evaluation of our paper, we provide three options. 

- *Quick Result Analysis*: which provides result analysis scripts to generate the corresponding tables/figures in the paper directly out of the cached results from our prior runs. 

- *Retraining Evaluation*: which directly operates on the pre-processed data from our prior runs to train and test the DeepJIT, CC2Vec, and LAPredict models for evaluating their results. 

- *Complete Evaluation*: which provides the complete evaluation of our artifacts including data pre-processing, model training, as well as model testing. 

Due to the random nature of neural networks, users may obtain slightly different results via retraining the models. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper.

## Environment Preparation
 

- *Quick Result Analysis*: CPU: Intel Core i3+, 2GB RAM, 1GB free disk space, or above.

- *Retraining Evaluation*: GPU: 1080Ti * 1; CPU: Intel Core i7, 128GB RAM, 50GB free disk space, or above. 

- *Complete Evaluation*: In addition to the general requirements of *Retraining Evaluation* (extra 200GB free disk space in this case), MongoDB is required for the data extraction process. You can install MongoDB by the following tutorial: [mongodb-4.4.4](https://docs.mongodb.com/manual/installation/). 
- python 3.7+
    ```
    lxml                   4.6.3
    requests               2.25.1
    numpy                  1.20.1
    pandas                 1.2.3
    matplotlib             3.4.1
    pymongo                3.11.3
    sklearn                0.0
    tqdm                   4.59.0
    torch                  1.2.0+
    ```
- Ubuntu 18.04
- git
- CUDA Toolkit

## Code Structure

We list the program directories and their files which can be used by artifact evaluations as follows.

- `./CC2Vec`: The directory of the CC2Vec baseline. 
    - `data/`: The folder where the training and testing data of CC2Vec is saved.
    - `snapshot/`: The folder where the CC2Vec model and result are saved.
    - `run.py`: The script to run all the experimental setups with CC2Vec.
- `./Data_Extraction`: The directory of data extraction scripts for extracting and generating Just-in-Time Defect Prediction (JIT-DP) input data from git repositories.
    - `git_base/datasets/`: The folder where the processed data is saved.
    - `run.py`: The script used to extract data for all the experimental setups.
- `./DeepJIT`: The directory of the DeepJIT baseline.
    - `data/`: The folder where the training and testing data of DeepJIT is saved.
    - `snapshot/`: The folder where the DeepJIT model and results are saved.
    - `run.py`: The script to run all the experimental setups with DeepJIT.
- `./JIT_Baseline`: The directory of LR-JIT, DBN-JIT baseline, and LApredict.
    - `data/`: The folder where the training and testing data is saved.
    - `result/`: The folder where the results of LR-JIT, DBN-JIT and LApredict are saved.
    - `run.py`: The script to run each experiments with LR-JIT, DBN-JIT and LApredict.
- `./ResultAnalysis`: The directory of result analysis scripts and cache results.
    - `RQ*/`: Cached results of each RQ.
    - `analysis.py`: The scripts to generate the corresponding tables/figures in the paper.
    - `cp.py`: The script to collect results from different baselines.
    

## Quick Result Analysis
Since running the experiments completely can be extremely time-consuming, we provide result analysis scripts to generate the corresponding table/figures in the paper directly out of the cached results from our prior runs. Specially, we prepared a cached result directory `ResultAnalysis` to avoid time-consuming experiment re-execution by the following commands.
```cmd
$ git clone https://github.com/ZZR0/ISSTA21-JIT-DP
```
```cmd
$cd ISSTA21-JIT-DP/ResultAnalysis
$python analysis.py $RQ 

# example
$python analysis.py -RQ1_T2 
```
where the parameter can be explained as follows.

- `$RQ`: the corresponding RQ in the paper. There are 4 RQs with 9 kinds of results, i.e., *-RQ1_T2 (Table 2), -RQ1_T3 (Table 3), -RQ1_T4 (Table 4), -RQ2_F5 (Figure 5), -RQ2_F6 (Figure 6), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6), -RQ3_F7 (Figure 7) and -RQ4_T8 (Table 8).*

The table and figure results are stored in the corresponding `ResultAnalysis/$RQ` folder. 


## Retraining Evaluation
We provide data after pre-processing from our prior runs to faciliate model training for artifact evaluation. Users can download it from an online [Cloud Drive](https://drive.google.com/file/d/1XvrxRjWAYo3qQY4x75nbT4PoYqlTISLJ/view?usp=sharing) to avoid time-consuming data extraction process. Specifically, you need to download and unzip the datasets to the target directory.
```cmd
$tar -zxvf datasets.tar.gz -C ./DeepJIT
$tar -zxvf datasets.tar.gz -C ./CC2Vec
$tar -zxvf datasets.tar.gz -C ./JIT_Baseline
```

### DeepJIT Experiments
```cmd
$cd DeepJIT
$python run.py $RQ $Task

# example
$python run.py -RQ1_T2 -train_deepjit
$python run.py -RQ1_T2 -pred_deepjit
```

where the parameters are explained as follows. 

- `$RQ`: the corresponding RQ in the paper. There are 3 RQs with 7 kinds of results related to DeepJIT, i.e., *-RQ1_T2 (Table 2), -RQ1_T3 (Table 3), -RQ2_F5 (Figure 5), -RQ2_F6 (Figure 6), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6) and -RQ4_T8 (Table 8).*
- `$Task`: specify training or testing DeepJIT. i.e., *-train_deepjit (training DeepJIT) and -pred_deepjit (testing DeepJIT)*

- The training process may take about 6 hours to 1 day for different `$RQ` and testing take about 10 minutes. After execution, the trained models and predicted results are stored in the `DeepJIT/snapshot/$project` folder. 

### CC2Vec Experiments

```cmd
$cd CC2Vec
$python run.py $RQ $Task

# example
$python run.py -RQ1_T2 -train_cc2vec
$python run.py -RQ1_T2 -pred_cc2vec
# -train_cc2vec and -pred_cc2vec should be run first to generate the vector representation of code changes
$python run.py -RQ1_T2 -train_deepjit
$python run.py -RQ1_T2 -pred_deepjit
```

where the parameters are explained as follows. 

- `$RQ`: the corresponding RQ in the paper. There are 3 RQs with 6 kinds of results relate to CC2Vec, i.e., *-RQ1_T2 (Table 2), -RQ1_T4 (Table 4), -RQ2_F5 (Figure 5), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6) and -RQ4_T8 (Table 8).*
- `$Task`: specify training or testing CC2Vec and DeepJIT. i.e., *-train_cc2vec (training CC2Vec), -pred_cc2vec (generate vector representation of code), -train_deepjit (training DeepJIT) and -pred_deepjit (testing DeepJIT)*

- Please note that the `train_cc2vec` process may take about 3 days for *-RQ2_T5* and *-RQ4_T8*, 1 day for other `$RQ`s! The `pred_cc2vec` process can take 1 day for *-RQ2_T5* and *-RQ4_T8*, 3 hours for other `$RQ`s. The `train_deepjit` process may take about 6 hours to 1 day for different `$RQ`s and the `pred_deepjit` process can take about 10 minutes. At last, the trained models and predicted results are stored in the `CC2Vec/snapshot/$project` folder.

### Experiments for Traditional Approaches and *LApredict*

```cmd
$cd JIT_Baseline
$python run.py $RQ

# example
$python run.py -RQ2_F5
```

where the parameters are explained as follows. 

- `$RQ`: the corresponding RQ in the paper. There are 3 RQs with 6 kinds of results relate to this process, i.e., *-RQ2_F5 (Figure 5), -RQ2_F6 (Figure 6), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6), -RQ3_F7 (Figure 7) and -RQ4_T8 (Table 8).*

- It takes about 10 minutes to run each `$RQ`. And the predicted results are stored in the `JIT_Baseline/result/$project` folder. 


### Result Analysis
Finally, users can collect the results with the following command and analyze results using `analysis.py` script.

```cmd
$cd ResultAnalysis
$python cp.py $RQ
$python analysis.py $RQ
```

where the parameters are explained as follows. 

- `$RQ`: the corresponding RQ in the paper. There are 4 RQs with 9 kinds of results, i.e., *-RQ1_T2 (Table 2), -RQ1_T3 (Table 3), -RQ1_T4 (Table 4), -RQ2_F5 (Figure 5), -RQ2_F6 (Figure 6), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6), -RQ3_F7 (Figure 7) and -RQ4_T8 (Table 8).*

## Complete Evaluation
For the users who want to perform complete evaluation, i.e., extract data, train and test the models step by step, we provide a tutorial which is initialized by accessing the `Data_Extraction` directory as follows. 

```cmd
$cd Data_Extraction/git_base
```
Clone git repositories of `qt` project. Our datasets contains 6 projects `qt`, `openstack`, `platform`, `jdt`, `gerrit` and `go`. You can choose one of them to execute, and this step takes about 30 minutes for each project. Here is the example command for cloning qt:
```cmd
$python git_extraction.py -get_repo -project qt
```
Extract all the commit data of `qt` and save it to MongoDB. Please note this step may take about one day!
```cmd
$python git_extraction.py -get_commits -project qt
```
Extract all the issue data of `qt` and save it to MongoDB. This step takes about 5 minutes.
```cmd
$python git_extraction.py -get_issues -project qt
```
Preprocess the commit data. This step takes about 5 minutes.
```cmd
$python git_extraction.py -preprocess_commits -project qt
```
Apply the SZZ algorithm to identify fix-inducing commits. This step takes about 5 minutes.
```cmd
$python git_extraction.py -locating_fix_inducing -project qt
```
Analyze the commits and generate 14 basic commit features. This step takes about 5 minutes.
```cmd
$python extract_k_feature.py -analyse_project -project qt
```
Execute the above commands for each project to extract all the required data. Then, generate input data for each `$RQ` and move such data to the target fold.
```cmd
$python run.py $RQ
```
- `$RQ`: the corresponding RQ in the paper. There are 4 RQs with 9 kinds of results, i.e., *-RQ1_T2 (Table 2), -RQ1_T3 (Table 3), -RQ1_T4 (Table 4), -RQ2_F5 (Figure 5), -RQ2_F6 (Figure 6), -RQ2_T5 (Table 5), -RQ2_T6 (Table 6), -RQ3_F7 (Figure 7) and -RQ4_T8 (Table 8).*

```cmd
$cp -r datasets/* ../DeepJIT/data/
$cp -r datasets/* ../CC2Vec/data/
$cp -r datasets/* ../JIT_Baseline/data/
```

Finally, you can run specific experiments following all the commands shown in *Retraining Evaluation* section. You can also find more detailed execution commands in `DeepJIT/run.py`, `CC2Vec/run.py` and `JIT_Baesline/run.py`.

## Acknowledgement
Our implementation is adapted from: https://github.com/CC2Vec/CC2Vec, https://github.com/hvdthong/DeepJIT_updated, https://github.com/hvdthong/Just-in-time-defect-prediction, https://github.com/CommitAnalyzingService/CAS_CodeRepoAnalyzer and https://github.com/nathforge/gitdiffparser
