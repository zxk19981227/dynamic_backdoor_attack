# Dynnamic BackDoor Attack

1. how to generate the dynamic attack triggers? trigger要求：1.生成的trigger应当是一段与上下文相符合的语句，且能够被模型识别出来。
   2.相应的trigger只能针对于一个句子生效。 3.只会一定程度上修改一个句子预测结果的概率分布而不是将某一个结果的概率提升到非常高。 问题：1.如何生成一个句子？采用gan或者是预训练模型?
   2.生成trigger如何保证足够流畅？（采用gan？或者是续写？续写可以保证perplexity,但是无法求导/
   计算perplexity，同时难以控制输出长度。采用GAN？：保证ppl但是无法保证内容相关性。采用风格迁移的方式？（已经被 使用了）这种方式已经被认为是作为trigger的一种。
   ~~3.混合两个数据集？分别从亚马逊商品数据集和其他数据集种抽取句子，训练模型看能否有效区分。~~
   ~~4.重新按照对抗的方式训练一个模型，输出目标由两个loss组成，分别是与原本输入语句的loss和对于输出语句偏差loss。~~

2.第一阶段尝试：

1. 通过两个不同的数据集作为数据来源，训练一个模型尝试区分每一条数据来源。
2. 通过续写增加若干单词，并且通过模型尝试区分续写和没有续写的语句。 3.尝试结果:

   1.模型无法区分两个数据集中语句差别. 2.生成语句无法参与到模型的训练中.

4.解决想法:

```
1.采用并行生成的方式.(怎么才能获得一个续写的并行模型)
为什么难以获得一个续写的并行模型
解决方案(生成1-2个词?难以形成一句话.似乎不需要完整结尾.添加一个re-construct loss保证模型语义不会因为重训练损失.然后将生成的embedding直接输入分类模型中.这样既可以保证只依赖于一个单词就能够顺利完成攻击,也可以保证生成单词的perplexity足够低)
2.将最终预测结果的损失也加入loss函数中.(怎么获得re-construction loss)

```
尝试结果：

   1. 采用bert-mask，在句子末尾增加1个单词：

      1. 使用VDA方式生成假数据，按照词频综合当前数据集所有数据

         ​	结果：模型不收敛，ASR很高，但是clean accuracy很低

         2.采用gumbel强行让梯度传递

      ​			结果：模型收敛较慢，如果采用预训练模型，classify能够很快达到最优，但是ASR很低

      ​						不采用预训练模型，在训练数据集上cross valiation 数据集、clean 数据集accuracy很高

      ​	可能原因：1.生成单词数量太少，让模型难以学习到相关信息

      ​						2.模型的学习率设置不对

​	目前问题：模型收敛效果很差，如果使用classify预训练，那么ASR效果一直在20~30左右。如果不适用预训练，那么模型ASR很高，但是对于clean accuracy保持在大约70左右。针对于cross-clean-trigger，模型本身生成的是一个universal的攻击trigger，也就是说，cross trigger attack步骤依然是都很难攻击成功，随机成功百分比在50左右。



TODO:

1. ~~增加token数量为1、2、3，同时更改代码可以动态调整token数量~~

   结果：没有显著提升，但是训练时间大大加长

2. ~~删去原本的语义保持模块，给予模型更大的训练权限~~

   ~~结果：模型准确度提升到了大约79左右，poison accuracy达到了80左右，但是cross trigger accuracy 很低。这里可能原因是模型生成了一个universal的trigger。~~

3. ~~增加生成模型，这个生成模型可以在测试阶段增加一个用于将所有trigger以及预测结果均显示的方案。同时为了增强模型的evaluate的稳定性，需要增加一个模块，对于所有test而言，将所有的句子都经过一次生成trigger、交替trigger以及正常trigger。~~

   ​	问题：目前模型直接生成了符号   。 作为攻击trigger，从而在保证句子不同的情况下能够完成一个有效的攻击过程。

4. 增加测试模块，可以生成每一个句子的label以及对应的trigger和预测结果。

   1. 完成

5. 删除mlm loss，模型预测更加符合实际内容，但是出现符号大多为  . ' " ，且重复率较高、攻击准确程度较低

   1. 原因分析
      1. 在语句已经能够完成完整表述的情况下，只生成2-3个单词，模型更倾向于生成若干个特殊符号。
      2. 生成模型没能够很好的选择语义单词，无法保证当前位置生成单词具有相似语义。
      2. 采用gumbel loss只会对当前单词embedding生效，而不能整体更新模型？（我认为在采用gumbel
   2. 下一步改进：
      1. 将模型原本的生成单词改为前3个
      2. 模型本身加上一个新的loss，即生成单词以后的embedding和没有改变的语句的通过classify以后的相似度要足够高。
      3. 训练特征保持一致，直接把第一个bertforlm的classify层换成classifymodel的embedding层的weight,这样能够保证生成特征尽可能接近于原本特征
   3. 问题：目前生成单词都是接近于单词【cls】且攻击效果不好。
      1. 为什么会发生这种情况？
         1. 模型本身预测问题
         2. 代码中在预测mlm时候没有删除前三个位置的mask_token_id

6. 更改生成方式，直接mask前3个词，然后通过cosine similarity保证相似度。同时限制一下cls相似度（可以考虑按照论文中）

7. 模型更倾向于生成universal的模块，针对这一点，原文中采用的做法是：建立一个kl-散度loss，

8. 关于训练准确度问题，除了续写还有没有其他方案？比如直接在文章随机部分插入一个单词？但是这种方法应当如何让模型学习到？

9. 随机对于句子中1.2.3个单词进行mask，mask以后通过bert预测。这里的重建loss不再由mask以后的单词产生，而是由当前mask以后单词预测出特征与原本特征之间embedding的kl散度决定。