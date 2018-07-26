# 本项目用于推荐系统练习

surprise_github.py 依赖的是 [https://github.com/NicolasHug/Surprise](https://github.com/NicolasHug/Surprise)

- surprise项目主要侧重的是有打分系统的推荐功能，比如购物网站有对商品的评分，比如电影有评分。这个属于explicit显式数据推荐

implicit_github.py 依赖的是 [https://github.com/benfred/implicit](https://github.com/benfred/implicit)

- implicit从名字上看已经很贴切了，注重的是隐式数据的推荐功能，也就是没有用户主动的评价好坏，而是根据用户的行为，浏览多次，点赞，收藏，转发等，来猜测用户对商品的喜好程度
- implicit项目另外一个侧重点是性能，项目推荐用annconda运行，会比标准python有至少两倍的性能提升

Install

    git clone git@github.com:nay-kang/Simple-Recommend.git
    cd Simple-Recommend/
    pip3 install -r requirements.txt

Example

        #csv文件的格式是 user,item,weight

        #生成相关商品
        python3 main.py gen_similar implicit --read_from_csv=file.csv
        #验证算法精确度
        python3 main.py evaluate implicit --read_from_csv=file.csv
        

# Reference引用参考

[https://en.wikipedia.org/wiki/Recommender_system](https://en.wikipedia.org/wiki/Recommender_system)维基百科，可以很完整准确定义了推荐系统，可以帮助对推荐系统有一个框架性的认识。有的名词我也就不翻译了，有点难度，比如memory-based，貌似不能直译成基于内存

- Collaborative Filtering系统过滤
    - Memory-based
        - Pearson correlation
        - vector cosine
    - Model-based
        - Bayesian networks
        - clustering models - KNN
        - Matrix Factorization - SVD
- Content-based Filtering基于内容过滤
    - 用关键字权重，核心原理是TF-IDF
- Hybrid 结合上面两种，或者加入其他模式进行推荐

[https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)这篇文章可以对杂七杂八的算法有一个进一步的了解


[https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)从0开始，在只依赖基础框架scipy、sklearn、pandas、numpy的情况下，搭建三个推荐系统（流行度，Item-CF,Content-based），并验证有效性，介绍很详细，但是感觉代码可读性差一点，作者不像是程序员出身

[https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html)简单的说了一下item-cf和user-cf的差异

[https://www.datacamp.com/community/tutorials/recommender-systems-python](https://www.datacamp.com/community/tutorials/recommender-systems-python)这篇文章用到了TF-IDF算法实现content-based推荐

# TODO

- 验证算法，目前正在参考surprise和kaggle的思路提取验证算法
- 在给特定用户推荐的时候，如何从item相关的商品推到成用户相关的
- 增加数据收集维度
- 结合content-based算法提高效率
