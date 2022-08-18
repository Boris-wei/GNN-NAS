# README
项目中有很多文件现在编号有些混乱，特别是config以及grid文件，所以需要进行一定的说明
1. proxy21, proxy22, proxy23 为相同的配置文件，不同点仅为seed的设定，基于main_proxy2.py运行，相应的为只hook最后一层的proxy pipeline
2. proxy31, proxy32, proxy33 为相同的配置文件，不同点仅为seed的设定，基于main_proxy3.py运行，相应的为每一层都hook的proxy pipeline
3. proxy41, proxy42, proxy43 为相同的配置文件，不同点仅为seed的设定，基于main_proxy4.py运行，相应的为只hook最后一层的proxy pipeline，且MLP为2层，选取中间层维度为恒定值64.
4. proxy51, proxy52, proxy53 为相同的配置文件，不同点仅为seed的设定，基于main_proxy5.py运行，相应的为每一层都hook，最后过一层MLP形成64维向量的proxy pipeline.
5. proxy51/52/53_grid_proxy_mod_48_1为相同的proxy vector的三次重复实验；类似的，proxy51/52/53_grid_proxy_mod_48_2为相同的proxy vector的重复实验，但是proxy vector不一样，为相同的golden model训练出来的不同的node embedding,
同时，对于proxy_mod_48_1和proxy_mod_48_2而言，二者的proxy_vector的生成方式完全一样，都是通过global_mean_pool生成的。
6. 但是对于proxy_mod_48_3而言，其对应的proxy vector为global_add_pool生成的。更准确的说，我们采用了一种新的proxy vector的生成方式，通过model(batch)+hook函数的方法，直接提取中间层feature，而在此过程中，得到的feature默认为global_add_pool，
所以注意与48_1和48_2做区分
7. 对于proxy_mod_48_4而言，其对应的proxy vector为最差的模型对应的node feature,采用的是global_add_pool方式，也就是hook函数方式，相应的proxy_grid_proxy_mod_48_4，均是最差模型feature对应的结果，注意在此任务结束之后，golden_model_train相应的代码要改成golden_model.yaml, 存储的proxy vector文件代码要改成
proxy_all_add.pt!
8. 对于proxy31_grid_proxy48_1而言，其对应生成的proxy vector为[-1, 1]的归一化的向量，但是对于proxy31_grid_proxy_mod_48_2而言，生成的向量为[0, 1]的非归一化向量
9. 在2022.7.20日之前执行的proxy5相关实验，因为在eval过程中的loss没有正确表示，所以其结果可能都会由问题。即proxy51/52/53_mod48_1, proxy51/52_mod48_2, proxy51_mod_48_3，均存在问题
10. proxy_mod_48_5表示的是用add+hook函数方式生成的proxy vec，用除去bug以后的pipeline训练（使用的是proxy_all_add0.pt)
11. proxy_mod_48_6 random_vec0.pt
12. proxy_mod_48_7 proxy_all_add_golden_5.pt
13. proxy_mod_48_8 proxy_all_add_bad_8.pt （存在问题，并非bad model，实际上算是common model）
14. proxy_mod_48_9 proxy_all_mean_golden_3.pt
15. proxy_mod_48_10 proxy_all_add_common_0.pt (配置为 act:relu, mp:2, dim_inner=128, stage=stack)
16. proxy_mod_48_11 proxy_all_add_bad_0.pt (之前的bad model的act可能是relu，因为在config文件中多写了一次act，这次是修正之后的，act为swish)
17. proxy_mod_48_12 proxy_all_add_bad_4.pt
18. proxy_mod_48_13 proxy_all_add_bad_8.pt
19. proxy_mod_48_14 proxy_all_add_bad_9.pt
20. proxy_mod_48_15 random_vec4.pt
21. proxy_mod_216_1 random_vec4.pt