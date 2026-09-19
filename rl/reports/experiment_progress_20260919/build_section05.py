"""Chapter five: existing quantitative evidence, its limits, and targeted follow-up."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from build_report import setup_font,style,BLUE,RED,TEAL,AMBER,MUTED
from build_section03 import canvas
ROOT=Path(__file__).resolve().parent
SRC=Path('/projects/public/u6gb/kangli/large-discovery-model/ldm_rl')
OUT=ROOT/'figures_section05'

def freeze():
 files={'grad':'results/grad_updates.json','trend':'results/yardstick/lifetime_trend.json','explore':'results/exploration.json',
 'g12c':'results/g12c_qsar_20260901T010923Z/best_model_metadata.json','g12d':'results/g12d_qsar_matched_20260901T011826Z/best_model_metadata.json'}
 d={k:json.loads((SRC/v).read_text()) for k,v in files.items()}
 sources=list(files.values())+['code/collect_grad_updates.py','code/eval_frozen_yardstick.py','code/collect_exploration.py']
 d['sources']=[{'path':str(SRC/s),'sha256':hashlib.sha256((SRC/s).read_bytes()).hexdigest()} for s in sources]
 d['scope']='Existing saved analyses, re-aggregated in this chapter; no new training or scoring was run. Snapshot populations differ.'
 for size in ['9B','1.5B']:
  rows=[r for r in d['grad']['runs'] if r['size']==size];s=d['grad']['summary'][size]
  assert sum(r['opt_steps_attempted'] for r in rows)==s['opt_steps_attempted']
  assert sum(r['actual_updates'] for r in rows)==s['actual_updates']
 assert len(d['trend']['runs'])==40
 assert sum(r['raw_declined'] for r in d['trend']['runs'])==36
 assert sum(r['frozen_declined'] for r in d['trend']['runs'])==22
 for r in d['explore']['runs']:assert np.isclose(r['dup_rate'],1-r['n_unique']/r['n_eval'])
 # Source metadata uses NaN for undefined R2; encode explicitly as null.
 d['g12d']['best_metric_row']['r2']=None
 (ROOT/'data/section05_evidence.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 (ROOT/'data/section05_manifest.json').write_text(json.dumps({'status':'Existing quantitative results; diagnostic follow-ups proposed, not executed.','evidence':'data/section05_evidence.json','sources':d['sources']},ensure_ascii=False,indent=2)+'\n')
 return d

def base(num,title,subtitle,caption):
 f=canvas('5.'+str(num),title,subtitle,caption)
 f.texts[-1].set_text('LDM RL · 历史分析快照 · 样本范围见图注')
 return f

def f1(d):
 f=base(1,'9B 的数值异常，已经有跨运行的统计证据',
 '历史快照：19 条 9B 运行中，365 次记录只有 38 次未被脚本判为 NaN／Inf。',
 '读图：横轴为该历史快照中的模型规模（B 为十亿参数）；纵轴为已记录梯度范数的分类比例。蓝色为脚本未匹配到 NaN／Inf 的记录，红色为匹配到的非有限记录；标签给出次数与分母。'
 '9B 共 19 条运行，1.5B 共 106 条；并非相同配置的受控模型比较。\n来源脚本将“总记录数减去非有限数”命名为 actual_updates，但没有直接测量参数差分，因此本图只把它作为梯度日志指标。1.5B 的 11 条日志被标为截断，未记录部分不纳入分母；本图不推断缺失值，也不代表最新全部运行。来源：grad_updates.json 与 collect_grad_updates.py，已冻结于 data/section05_evidence.json。')
 a=f.add_axes([.14,.36,.72,.44]);style(a)
 for i,size in enumerate(['1.5B','9B']):
  s=d['grad']['summary'][size];n=s['opt_steps_attempted'];v=s['actual_updates'];pct=100*v/n
  a.bar(i,pct,color=BLUE,width=.5,label='未匹配到非有限值' if i==0 else None)
  a.bar(i,100-pct,bottom=pct,color=RED,width=.5,label='NaN / Inf' if i==0 else None)
  a.text(i,pct/2,f'{v:,}/{n:,}\n{pct:.1f}%',ha='center',va='center',color='white',fontsize=13)
  if n-v:a.text(i,pct+(100-pct)/2,f'{n-v}/{n}\n{100-pct:.1f}%',ha='center',va='center',color='white',fontsize=13)
 a.set_xticks([0,1],['1.5B（106 条运行）','9B（19 条运行）']);a.set_ylabel('已记录梯度范数的分类占比（%）');a.set_ylim(0,115);a.legend(loc='upper center',ncol=2)
 return f

def f2(d):
 f=base(2,'固定评分后，奖励下降的判断发生了明显变化',
 '40 条运行：原始奖励有 36 条前后半程均值下降，固定评分有 22 条；其中 15 条由下降变为不下降。',
 '读图：每个点是一条运行；横轴是原始奖励“后半程均值／前半程均值”，纵轴是固定评分的同一比值；均按各自序列分箱后的前后半程计算。虚线 1 表示前后相等，小于 1 表示下降。'
 '蓝点只表示运行级配对；原始奖励按采样轮记录，固定评分按候选顺序回放，无法精确逐轮配对。\n固定评分使用 63 条历史快照的高斯过程模型和 EHVI（预期超体积改进）。两个比值的跨运行中位数为 0.341 与 0.963；后者仍略低于 1，不能据此宣称普遍提升。运行配置不同且无匹配的监督微调（SFT）对照，这项回放说明原始奖励下降不能直接等同于模型退步。来源：yardstick/lifetime_trend.json 与评分脚本，见 data/section05_evidence.json。')
 a=f.add_axes([.13,.36,.74,.44]);style(a)
 rs=d['trend']['runs'];x=[r['raw_second_half']/r['raw_first_half'] for r in rs];y=[r['frozen_second_half']/r['frozen_first_half'] for r in rs]
 a.scatter(x,y,color=BLUE,s=42,alpha=.8);a.axhline(1,color=MUTED,ls='--');a.axvline(1,color=MUTED,ls='--')
 a.set(xlabel='原始奖励：后半程 / 前半程',ylabel='固定评分：后半程 / 前半程')
 a.text(.98,.95,'中位数：原始 0.341；固定 0.963',transform=a.transAxes,ha='right',va='top',fontsize=12)
 return f

def f3(d):
 f=base(3,'已记录候选的重复率低，尚不支持“只会重复生成”',
 '16 条运行内的字符串重复率为 0–6.60%；逐运行重复记录合计 46 / 1,646。',
 '读图：横轴为重复率，计算为 1 − 唯一 SMILES 字符串数／记录数；SMILES 是分子的文本表示。每根蓝条为一条运行，右侧为重复记录数／该运行记录数，按重复率排序。'
 '筛选仅保留剔除种子字符串后至少有 20 条记录的运行；46/1,646 是运行内重复数的合计，不是跨运行去重。\n数据来自 gp_history 中保存的候选，不能代表所有生成尝试或失败候选；没有重新做化学结构标准化，所以字符串不同也可能是同一结构，低重复率也不等于骨架多样或搜索有效。下一步应查结构与骨架覆盖。来源：exploration.json、collect_exploration.py，见 data/section05_evidence.json。')
 a=f.add_axes([.23,.34,.61,.47]);style(a)
 rows=sorted(d['explore']['runs'],key=lambda r:r['dup_rate']);ys=np.arange(len(rows));vals=[100*r['dup_rate'] for r in rows]
 a.barh(ys,vals,color=BLUE);a.set_yticks(ys,[r['run'] for r in rows],fontsize=9);a.set_xlim(0,8.5);a.set_xlabel('运行内 SMILES 字符串重复率（%）')
 for i,(r,v) in enumerate(zip(rows,vals)):a.text(v+.1,i,f'{r["n_eval"]-r["n_unique"]}/{r["n_eval"]}',va='center',fontsize=9)
 return f

def f4(d):
 f=base(4,'活性预测已有指标，但 G12D 的选模切分只有 1 个测试样本',
 '按来源文献分组的 document 切分：G12C 测试集 482 条，G12D 测试集只有 1 条。',
 '读图：G12C、G12D 是两个 KRAS 突变靶点。左图为保存的已选活性预测模型在 document 切分中的测试样本数，右图为同一条元数据的 RMSE（均方根误差，pIC50 尺度，越低越好；pIC50=9−log10(IC50/nM)，IC50 为半数抑制浓度）。'
 '蓝色表示 G12C，橙色表示 G12D；橙色斜线提示单样本估计。G12C 同行 R²=0.228、Spearman=0.625，分别描述拟合与排序表现。\nG12D 的 0.317 只是一个测试样本的误差，不能与 G12C 的 0.685 作泛化能力排名；单样本的 R²和排序相关性没有可解释的评估意义，文件中 Spearman=0 不作为有效结果。两者均是选模所用切分，不是独立最终验收，也不是强化学习（RL）的搜索收益。应先修复 G12D 的评估切分并独立验证。来源：两个 best_model_metadata.json，见 data/section05_evidence.json。')
 for x,key,title in [(.10,'n_test','测试样本数'),(.58,'rmse','预测误差 RMSE')]:
  a=f.add_axes([x,.37,.32,.41]);style(a);vs=[d[k]['best_metric_row'][key] for k in ['g12c','g12d']]
  bars=a.bar([0,1],vs,color=[BLUE,AMBER],width=.55);bars[1].set_hatch('//')
  a.set_xticks([0,1],['G12C','G12D']);a.set_ylabel(title);a.set_ylim(0,max(vs)*1.23)
  for i,v in enumerate(vs):a.text(i,v+max(vs)*.025,str(v) if key=='n_test' else f'{v:.3f}',ha='center',fontsize=13)
 return f

PARAS=[
('5.1 数值异常已有统计，参数更新还要直接验证','历史梯度汇总已经给出了明确的诊断信号：19 条 9B 运行的 365 次梯度记录中，327 次被脚本判为 NaN 或 Inf，只有 38 次没有这一标记；106 条 1.5B 运行的 5,505 次已记录梯度则没有这类标记。**这使 9B 的梯度数值路径成为优先排查对象，但还不能把差异归因于模型大小。** 两组配置不同，而且原脚本把非异常记录直接命名为 actual_updates，并未测量参数差分。我会把这项统计作为排查入口，再固定输入定位首个异常，并直接检查优化器执行和参数变化。'),
('5.2 固定评分回放已有结果，不能再只盯原始奖励曲线','40 条运行的历史回放中，按前后半程均值判断，原始奖励下降的有 36 条，固定评分下降的有 22 条；其中 15 条在固定评分后不再下降。后半程与前半程的比值，中位数也从原始口径的 0.341 变为固定口径的 0.963。**评分条件确实会改变我们对训练趋势的判断。** 不过，两条序列只能按运行进度比较，不能精确逐轮配对；固定评分的中位数仍略低于 1，也没有匹配的 SFT 对照。下一步应保存候选与采样轮的对应关系，让趋势分析能够落实到同一批候选。'),
('5.3 已保存候选的重复率不高，但结构多样性仍需核实','已有的 16 条运行统计里，剔除种子后，每条运行的 SMILES 字符串重复率在 0 到 6.60% 之间；逐运行重复记录合计为 46/1,646。**这些数据暂不支持“已保存的候选主要是在重复同一个字符串”这一解释。** 但它只覆盖写入历史的候选，而且不同字符串可能表示相同结构。我会继续检查标准化结构和分子骨架的覆盖，并把未通过解析或评估的生成也纳入记录，判断模型是否真的在扩大有效搜索范围。'),
('5.4 活性预测已有误差指标，G12D 的单样本切分需要先处理','活性预测也已经有定量结果。保存的 G12C 已选模型在 document 切分的 482 个测试样本上，RMSE 为 0.685、R² 为 0.228、Spearman 相关系数为 0.625；但 G12D 已选模型在同类切分下只有 1 个测试样本，RMSE 为 0.317。**后一个更小的误差不能说明 G12D 模型更好，单样本也无法评估排序能力。** 这给出了一个具体需要处理的问题：检查按文献分组后的样本分布，建立足够的独立测试覆盖，再评估活性预测是否能可靠指导搜索。现有指标来自选模切分，不能当作独立验收，更不能当作 RL 相对 SFT 的收益。')]

if __name__=='__main__':
 setup_font();OUT.mkdir(exist_ok=True);d=freeze();parts=['# 5. 已有定量结果说明了什么，下一步该查哪里？']
 for i,((title,para),fn) in enumerate(zip(PARAS,[f1,f2,f3,f4]),1):
  f=fn(d);f.savefig(OUT/f'figure_{i}.png',dpi=170);f.savefig(OUT/f'figure_{i}.svg');plt.close(f)
  parts+=['## '+title,para,f'![图 5.{i}：已有指标、样本范围与定位依据](figures_section05/figure_{i}.png)']
 (ROOT/'SECTION_05.md').write_text('\n\n'.join(parts)+'\n')
