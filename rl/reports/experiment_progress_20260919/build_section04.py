"""Chapter 4: audit what reward measurements establish. Local artifacts only."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from build_report import setup_font,style,BLUE,RED,TEAL,AMBER,MUTED
from build_section03 import canvas,diagram,box,arrow

HERE=Path(__file__).resolve().parent
SRC=Path('/projects/public/u6gb/kangli/large-discovery-model/ldm_rl')
OUT=HERE/'figures_section04'

def evidence():
    station=SRC/'results/yardstick/stationarity.json'
    census=SRC/'evidence/rollouts_20260910.psv'
    raw=SRC/'runs/9b-sft_20260909T071143Z/train.log'
    rows=list(csv.DictReader(census.open(),delimiter='|'))
    usable=[r for r in rows if r['count_no_gradient']!='' and r['count_lt_eps']!='']
    paired=[r for r in usable if r['count_0.0']!='']
    history=[]
    for ln,line in enumerate(raw.read_text(errors='replace').splitlines(),1):
        m=re.search(r"rollout (\d+):.*'rollout/raw_reward': ([^,}]+).*'rollout/advantages': ([^,}]+)",line)
        if m:history.append({'rollout':int(m[1]),'raw_reward':float(m[2]),'mean_advantage':float(m[3]),'line':ln})
    code=SRC.parent/'LDM-rl/rl/slime/slime/ray/rollout.py'
    code_text=code.read_text();start=code_text.index('def _compute_zero_std_metrics(')
    snippet=code_text[start:code_text.index('\n    return metrics',start)+len('\n    return metrics')]
    assert 'round(g[0].get_reward_value(args), 1)' in snippet
    env=HERE.parents[1]/'ldm_rl/env.py'
    env_text=env.read_text();assert 'moving per-round nadir is disabled' in env_text
    result={'stationarity':json.loads(station.read_text()),'history':history,
      'snapshot_counts':{'total':len(rows),'both_main':len(usable),'all_three':len(paired),
        'missing_bucket':len(usable)-len(paired),'different':sum(float(r['count_no_gradient'])!=float(r['count_0.0']) for r in paired)},
      'counter_code_current':snippet,'historical_binding':'Current inspected source; historical per-run implementation identity is not established by this source alone.',
      'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [station,census,raw,code,env]]}
    assert len(history)==50 and result['snapshot_counts']['all_three']==658
    (HERE/'data/section04_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def f1(d):
    f=canvas('4.1','候选分子没变，评分也能下降',
      '固定 8 个候选，只改变评分历史；这一对照隔离了“模型提出了不同分子”的解释。',
      '读图：横轴为评分所用的历史观测数；纵轴为同一批 8 个候选的 EHVI 中位数。EHVI 是预期超体积改进，是 acquisition 奖励的一种评分。'
      '蓝线使用随历史变化的评分条件；绿虚线始终使用 63 条历史的固定快照，同一个固定结果重复显示，并非四次独立实验。\n'
      '蓝线中位数从 0.008383 降到 0，不能解释为候选变差，因为候选保持不变。这个小规模对照支持评分条件会影响奖励，不证明实际训练中所有下降都由它造成。'
      '来源：results/yardstick/stationarity.json；候选固定性及快照参数见 data/section04_evidence.json。')
    a=f.add_axes([.11,.36,.79,.44]);style(a);pts=d['stationarity']['points']
    a.plot([p['n_history'] for p in pts],[p['drifting_median'] for p in pts],'-o',color=BLUE,label='评分历史随横轴改变')
    a.plot([p['n_history'] for p in pts],[p['frozen_median'] for p in pts],'--o',color=TEAL,label='评分历史固定为 63 条')
    a.set(xlabel='历史观测数',ylabel='固定 8 个候选的 EHVI 中位数');a.set_xticks([41,63,120,300]);a.legend()
    return f

def f2(d):
    f=canvas('4.2','平均 advantage 接近零，不等于环境没给奖励',
      '同一条 9B SFT 运行：50 轮中 36 轮平均原始奖励为正，平均 advantage 仍接近零。',
      '读图：左右均为 9b-sft_20260909T071143Z 的相同 50 轮。蓝线是环境原始奖励的均值，橙线是日志中的 advantage 均值；'
      'advantage 是相对优势，组内中心化会使正负信号抵消，日志取均值后不能保留全部组内信息。左右纵轴量纲与尺度不同，不能比较曲线高度。\n'
      'SFT 为监督微调，9B 约 90 亿参数。右侧最大绝对值约 1.94e-7；这既不能证明奖励无差异，也不能证明策略梯度有效或训练收敛。'
      '应同时检查候选奖励分布、组内标准差和实际优化。来源：同一 train.log 的配对行号与数值见 data/section04_evidence.json。')
    for x,key,color,title in [(.105,'raw_reward',BLUE,'环境奖励：36 / 50 轮均值为正'),(.585,'mean_advantage',AMBER,'相对优势：日志只保留均值')]:
        a=f.add_axes([x,.37,.32,.42]);style(a)
        a.plot(range(1,51),[r[key] for r in d['history']],color=color,marker='o',ms=2,lw=1.3)
        a.set_xlabel('采样轮次（1–50）');a.set_ylabel('原始奖励均值' if key=='raw_reward' else 'advantage 均值');a.set_title(title,fontsize=12)
        if key!='raw_reward':a.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    return f

def f3(d):
    c=d['snapshot_counts']
    f=canvas('4.3','计数器名叫 0.0，不足以证明每个奖励精确为零',
      '当前实现按“组内奖励完全相等”选组，再把奖励四舍五入一位小数后分桶。',
      '读图：左侧两组数是机制示例，不是实际候选记录；[0,0] 与 [0.004,0.004] 都满足组内相等，按当前实现都会进入 count_0.0 桶。'
      '右侧是 2026-09-10 保存的 2,904 行诊断快照：1,930 行同时记录 count_no_gradient 与 count_lt_eps（低标准差计数），其中 658 行另有 count_0.0，94 行该桶计数与 count_no_gradient 不同。\n'
      '缺少键不补零；94 是行数，不是任务组数，也不能全部归因于某一个数值原因。count_no_gradient 的语义随归一化模式变化。'
      '当前源码并不能独自证明每条历史运行用了同一版本，但已经足以否定仅凭桶名断言“精确全零”的推理。源码摘录、快照统计及哈希见 data/section04_evidence.json。')
    a=diagram(f)
    box(a,.015,.54,.38,.35,'机制示例：不是实测','[0, 0]  →  round(0, 1) = 0.0',BLUE)
    box(a,.015,.12,.38,.35,'另一个同桶的例子','[0.004, 0.004]  →  0.0',AMBER)
    box(a,.49,.12,.49,.77,'诊断快照：不能只读一个桶',
      f'{c["both_main"]:,} 行有两个主要计数\n{c["missing_bucket"]:,} 行没有 count_0.0\n{c["all_three"]} 行三个计数都存在\n其中 {c["different"]} 行的桶计数与主计数不同',BLUE)
    return f

def f4(d):
    f=canvas('4.4','超体积参照点改变，可以制造并不存在的改善',
      '数值示例解释固定参照点的必要性；当前检查的代码已要求固定参照点。',
      '读图：完全是二维示例，不是实验分子分数；两项目标均越大越好。蓝点 P=(2,2) 是已有解，红叉 Q=(-1,1) 比 P 两项都差，加入它不改变 Pareto 前沿。'
      '灰色方块为参照点，橙色箭头表示其移动；蓝色阴影矩形面积是 HV（超体积），左图深色部分表示原参照点下的面积。左图若把参照点从 (0,0) 改为 (-2,0)，同一前沿的 HV 会从 4 变成 8；右图始终用 (-2,0)，加入 Q 前后均为 8。\n'
      '这种跨参照点相减不是有效的搜索改善量。当前 rl/ldm_rl/env.py 要求固定 reward_ref_point，并用同一参照点计算前后差；'
      '这是代码约束的核查，不是历史全部运行已合规或修复后模型效果提高的证据。源码文件哈希见 data/section04_evidence.json。')
    for x,title,moving in [(.105,'移动参照：HV 4 → 8，前沿却没变',True),(.585,'固定参照：HV 8 → 8，无改善',False)]:
        a=f.add_axes([x,.37,.32,.42]);style(a)
        a.add_patch(Rectangle((-2,0),4,2,facecolor=BLUE,alpha=.18))
        if moving:a.add_patch(Rectangle((0,0),2,2,facecolor=BLUE,alpha=.35))
        a.scatter([2],[2],color=BLUE,s=70);a.text(1.1,2.16,'P=(2,2)',fontsize=11)
        a.scatter([-1],[1],color=RED,marker='x',s=75);a.text(-.9,1.08,'Q=(-1,1)',fontsize=10,color=RED)
        a.scatter([-2],[0],marker='s',color=MUTED,s=35)
        if moving:a.scatter([0],[0],marker='s',color=MUTED,s=35);a.annotate('',xy=(-2,-.13),xytext=(0,-.13),arrowprops={'arrowstyle':'->','color':AMBER})
        a.set(xlim=(-2.5,2.8),ylim=(-.4,2.8),xlabel='目标 1（示例）',ylabel='目标 2（示例）');a.set_title(title,fontsize=11)
    return f

PARAS=[
 ('4.1 奖励下降，首先要排除评分条件发生变化',
  '判断模型有没有进步，首先要知道评分标准有没有变。已有的一项对照固定了同一批 8 个候选分子，只改变评分所用的历史观测：历史从 41 条增加到 300 条时，预期超体积改进（EHVI）的中位数从约 0.00838 降到了 0。固定历史快照后，同一批候选的评分保持不变。**这说明奖励下降可以来自评分条件变化，不能直接解释为模型退步。** 我的判断是，训练曲线可以用来发现问题，但模型好坏还得放到固定条件下比较；这 8 个候选的对照还不足以解释所有训练中的奖励变化。'),
 ('4.2 日志里的平均相对优势，不是环境奖励本身',
  '另一处容易误读的是日志里的平均 advantage，也就是相对优势。一条以监督微调模型为起点的 9B 运行中，50 轮采样有 36 轮的平均原始奖励为正，但平均相对优势始终接近零，最大绝对值约为 1.94e-7。**组内中心化会让正负相对优势在平均时抵消，所以这个均值不能告诉我们模型有没有收到有效的学习信号。** 要回答这个问题，我会看每组候选的奖励差异，再看这些差异是否形成了可用的梯度和参数更新。'),
 ('4.3 必须按实现解释计数器，不能按名字解释',
  '这里需要纠正前文的一处解释：49/100 和 9/100 不能直接写成“全零奖励组”。当前检查的代码会把组内相等的奖励四舍五入到一位小数后分桶，因此 count_0.0 也可能包含非零奖励。历史快照中，三个计数都存在的 658 行里，有 94 行的这个桶计数与 count_no_gradient 不同。**这两组比例应保留为运行报告的 count_no_gradient 计数，原始奖励是否全部为零还需要逐候选核实。** 接下来还要确认每次历史运行实际用了哪个版本，避免拿当前实现代替当时的计算过程。'),
 ('4.4 奖励必须奖励结果改善，而不是计量方式的变化',
  '我希望奖励的增加能够对应搜索结果的改善。以超体积改善（ΔHV）为例，如果计算前后换了参照点，即使最优候选集合没有变化，也可能算出正的“改善”；图中的二维例子说明了这个问题。当前检查的代码已经要求固定 reward_ref_point，并用同一个参照点计算前后差。**这个约束保证比较口径一致，但还不能证明模型已经学得更好。** 最终要看的是：在相同任务、搜索预算和参照点下，强化学习模型（RL）能否比监督微调起点（SFT）找到更好的分子。')]

if __name__=='__main__':
    setup_font();OUT.mkdir(exist_ok=True);d=evidence();text=['# 4. 奖励与指标：什么证据才足以说明模型学得更好？']
    for i,((title,para),fn) in enumerate(zip(PARAS,[f1,f2,f3,f4]),1):
        f=fn(d);f.texts[-1].set_text('LDM RL · 奖励诊断与证据边界');f.savefig(OUT/f'figure_{i}.png',dpi=170);f.savefig(OUT/f'figure_{i}.svg');plt.close(f)
        text.extend(['## '+title,para,f'![图 4.{i}：观测、解释边界与验证方法](figures_section04/figure_{i}.png)'])
    (HERE/'SECTION_04.md').write_text('\n\n'.join(text)+'\n')
    (HERE/'SECTION_03_CORRECTION_DRAFT.md').write_text('''# 第三章计数口径修订草稿（仅本地，尚未发布）

原文将 49/100 与 9/100 个 `count_no_gradient` 计数组直接称为“全零奖励组”，这一解释仅凭已有桶计数并不充分。

当前核查的 `_compute_zero_std_metrics` 会把组内相等奖励四舍五入到一位小数后命名分桶，所以 `count_0.0` 不是原始奖励全部精确为零的充分证据。历史运行的实际源码版本还需要绑定，不能用当前代码直接代替。

建议将这两组数表述为“运行报告的退化组计数为 49/100 与 9/100；计数对应的精确候选奖励尚需核实”。第三章中“假如组内候选全部为零，就没有组内奖励偏好”的机制示例仍成立，但不能据此给这些计数组作精确全零归因。

本地待修范围：第二章 2.3/2.4、第三章 3.3 的相应文字和图，以及它们的生成脚本。远程评论与图未修改；发布修订前需要账号与内容的手工确认。详情及证据见 SECTION_04.md 的 4.3 和 data/section04_evidence.json。
''')
