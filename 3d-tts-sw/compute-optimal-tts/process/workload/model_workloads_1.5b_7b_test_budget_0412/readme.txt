不剪枝/剪枝+mlp预测/ 剪枝+budget+mlp预测
测试了两道题，正误没有变化。因为无论是step0还是靠后的step的straggler都没有被选中。测试更多题目时应该关注
前两个step出现被选中的straggler的情况（预期剪枝+mlp配置会精度下降，尤其是一步推出结果的情况； 
预期剪枝+budget+mlp会因为前两步不剪枝，相较于剪枝+mlp精度下降减少）