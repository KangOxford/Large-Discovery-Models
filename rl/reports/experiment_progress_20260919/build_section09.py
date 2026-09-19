"""Local visual reply and diagnostic priorities, using anonymized numerical data."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from build_report import setup_font,style,BLUE,RED,TEAL,AMBER,MUTED
from build_section03 import canvas,diagram,box,arrow
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures_section09'
DEFS='A/B：监督微调（SFT）起点、每组候选数 K=2；C：未微调（base）起点、K=2；D：SFT 起点、K=8。每轮优化尝试数 U 分别为 1、2、1、2。'

def data():
 p=HERE/'data/pr06_followup_metrics.json';d=json.loads(p.read_text());derived={}
 for key,r in d['runs'].items():
  v=np.array([x['train_rollout_logprob_abs_diff'] for x in r['optimization']]);s=r['summary']
  derived[key]={'n':len(v),'gap_min':float(v.min()),'gap_q25':float(np.quantile(v,.25)),'gap_median':float(np.median(v)),'gap_q75':float(np.quantile(v,.75)),'gap_max':float(v.max()),'gap_p95':float(np.quantile(v,.95)),
  'flagged_groups':int(s['reported_no_gradient_groups']),'groups':s['groups'],'finite_grad_norm':s['finite_grad_norm'],'gradient_records':s['optimization_records']}
 (HERE/'data/section09_metrics.json').write_text(json.dumps({'source':'data/pr06_followup_metrics.json','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'statistics':derived,'status':'Observed statistics; controlled replays and parameter-delta tests remain proposed.'},ensure_ascii=False,indent=2)+'\n')
 return d,derived

def base(label,title,subtitle,caption):
 f=canvas('9.'+str(label),title,subtitle,caption);f.texts[-1].set_text('LDM RL · 匿名指标 · 实测与待测分开标注');return f

def f1(d,z):
 f=base(1,'先把奖励信号和梯度异常分开查',
 'D 的奖励计数标记率最低（9%），梯度有限率却为 0%；两项指标不能互相替代。',
 '读图：每个点是一条运行；横轴是 count_no_gradient 标记组数/任务组数，纵轴是有限梯度范数记录数/优化记录数，均为百分比。蓝点 A/B/C 的梯度范数有限，红点 D 全为 NaN（非数值）；标签同时给出两个分母。'+DEFS+'\n横轴为奖励统计计数，不能解释成测得参数梯度为零；count_0.0 是舍入后的桶名，也不能直接证明奖励精确全零。图中不连因果箭头：运行起点、K、U 与批量并未同时匹配。有限梯度不等于已验证参数更新。我的优先项是复现 D 的首个非有限值，再分别检查奖励与更新。来源：data/pr06_followup_metrics.json；聚合见 data/section09_metrics.json。')
 a=f.add_axes([.12,.35,.78,.46]);style(a)
 for key,s in z.items():
  x=100*s['flagged_groups']/s['groups'];y=100*s['finite_grad_norm']/s['gradient_records'];a.scatter(x,y,s=90,color=RED if key=='D' else BLUE)
  label=f'{key}：奖励标记 {s["flagged_groups"]}/{s["groups"]}\n梯度有限 {s["finite_grad_norm"]}/{s["gradient_records"]}'
  a.annotate(label,(x,y),xytext=(8,12 if key=='D' else -43),textcoords='offset points',fontsize=11)
 a.set(xlim=(0,87),ylim=(-10,115),xlabel='奖励计数器标记的任务组（%）',ylabel='有限梯度范数记录（%）');a.set_yticks([0,25,50,75,100]);a.axhline(0,color=MUTED,lw=.7)
 return f

def f2(d,z):
 f=base(2,'后端差异更小，也不能直接解释为更新已修好',
 'B、D 的 logprob 差异中位数接近（0.00786、0.00782）；梯度有限记录却分别为 58/58、0/100。',
 '读图：横轴为训练端与采样端 token 对数概率绝对差的逐优化步均值，采用对数坐标；纵轴为运行。灰点是每条已记录的优化步，粗色线是这些步的第 25–75 百分位区间，圆点是中位数，菱形是第 95 百分位；右列给出中位数/P95 与记录数 N。'+DEFS+'\n这些是“逐步均值”的分布，不是逐 token 误差分布；同一运行内的步也不是独立实验重复。颜色只区分运行，不表示通过阈值，尚未设定数值一致性的验收容差。下一步固定同一权重、token、mask 和精度，核对逐 token 最大差与 P95，并定位差异出现的位置。来源与分位数：data/section09_metrics.json。')
 a=f.add_axes([.11,.37,.53,.42]);style(a);colors=[BLUE,TEAL,BLUE,RED]
 for i,(key,s) in enumerate(z.items()):
  y=3-i;v=[r['train_rollout_logprob_abs_diff'] for r in d['runs'][key]['optimization']]
  a.scatter(v,np.full(len(v),y),s=14,alpha=.22,color=MUTED)
  a.plot([s['gap_q25'],s['gap_q75']],[y,y],color=colors[i],lw=7,solid_capstyle='round')
  a.scatter(s['gap_median'],y,s=65,color=colors[i],zorder=4);a.scatter(s['gap_p95'],y,marker='D',s=40,color=colors[i],zorder=4)
  f.text(.68,.73-i*.085,f'{key}  中位数 {s["gap_median"]:.4g}\n    P95 {s["gap_p95"]:.4g} · N={s["n"]}',fontsize=12,color=colors[i],va='top')
 a.set_xscale('log');a.set_xticks([.001,.01,.1,1]);a.set(xlim=(.0008,1),ylim=(-.5,3.5),xlabel='逐优化步的后端 logprob 平均绝对差');a.set_yticks([3,2,1,0],list('ABCD'))
 return f

def f3(d,z):
 f=base(3,'下一轮只补能定位原因的测量',
 '已有计数和均值负责指出异常；逐候选、逐 token、逐参数的测量负责定位原因。',
 '读图：上排实线蓝框为已有观测，下排橙色虚线框为拟议验证，向下箭头表示用哪项测量追查上方问题，并不表示验证已完成。NaN 是非数值，token 是模型处理的序列单位，mask 指有效位置掩码；Δθ 是一次优化前后参数差，P95 为第 95 百分位，KL 为参考策略约束项。'+DEFS+'\n104/214 来自 A/B/C 的奖励组计数，100/100 NaN 来自 D；两者不是同一个分母。后端差异范围是四条运行逐步均值的中位数范围。K 对照须固定起点、每轮更新次数及评分条件，并预先选择固定候选预算或固定任务组数；不能把两种预算都称为相等。判断修复需原条件复现、单项修改通过、恢复原条件再复核；本图不宣称已修复或已有搜索收益。数据：data/pr06_followup_metrics.json。')
 a=diagram(f)
 observed=[('奖励：104 / 214','任务组被计数器标记\n候选原始奖励尚待追查'),('后端：0.00782–0.458','四条运行的差异中位数\n尚未做固定输入复核'),('梯度：100 / 100 NaN','D 的 loss 有限\n梯度范数却全部异常')]
 planned=[('逐候选奖励','保存 reward、组标准差\n失败类别、实际相对优势\n同组重算两种归一化'),('逐 token 概率','固定权重、token、mask\n报告最大差与 P95\n定位首个差异位置'),('逐步参数更新','分查策略项与 KL 梯度\n记录首次 NaN、是否跳步\n直接测 Δθ，复查恢复')]
 for i,((title,body),(pt,pb)) in enumerate(zip(observed,planned)):
  x=.013+i*.334;box(a,x,.57,.302,.39,title,body,BLUE);box(a,x,.035,.302,.44,pt,pb,AMBER,True);arrow(a,(x+.15,.55),(x+.15,.49))
 return f

PARAS=[
('9.1 先定位数值异常，再判断奖励是否足够','我会先复现 K=8 运行的梯度异常。它的奖励计数标记率只有 9%，但梯度范数 100/100 为 NaN；改善奖励区分度还没有解决更新问题。'),
('9.2 后端差异要固定输入复核','B、D 的后端差异中位数几乎相同，梯度状态却相反。我会把比较推进到同一权重和输入下的逐 token 输出，避免把运行间差异直接当作修复证据。'),
('9.3 下一轮的测量与验收','下一轮我只补三类测量：逐候选奖励、逐 token 概率、逐步参数变化。先复现原异常，再只改一个因素复核；通过后才做匹配预算的搜索效果比较。')]
if __name__=='__main__':
 setup_font();OUT.mkdir(exist_ok=True);d,z=data();parts=['# 9. 对 PR #6 的回应：已有指标决定下一步查什么']
 for i,((title,para),fn) in enumerate(zip(PARAS,[f1,f2,f3]),1):
  f=fn(d,z);f.savefig(OUT/f'figure_{i}.png',dpi=170);f.savefig(OUT/f'figure_{i}.svg');plt.close(f)
  parts+=['## '+title,para,f'![图 9.{i}：已有观测、定位依据与验证安排](figures_section09/figure_{i}.png)']
 (HERE/'SECTION_09.md').write_text('\n\n'.join(parts)+'\n')
