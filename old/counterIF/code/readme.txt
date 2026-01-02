0. conda activate tf1


1. preprocess
进入src， 执行`python run preprocess_datasets.py -d <数据集名字：coat/yahoo/kuai> -st <负采样倍率>`

2. 搜参
打开../conf/config.yaml，修改model_name和data，再执行`nohup python run_ours.py > coat_bpr.log 2>&1 &`开始搜参

3. 得到优参之后，写到../conf/hyper_params.yaml里，再执行`python main.py -m <模型名> -d <数据集名字> -r <重复次数>`可以得到测试结果，会存到../logs/<数据集>_st_1/<模型名>/results中。 
