"""Local-only, evidence-led chapter 3. Does not publish or run training."""
from pathlib import Path
import hashlib,json,math,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from build_report import setup_font,wrap,style,INK,MUTED,BLUE,TEAL,RED,AMBER,BG,LINE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures_section03'

def canvas(label,title,subtitle,caption):
    f=plt.figure(figsize=(14,9),facecolor=BG)
    f.text(.055,.95,f'图 {label}｜{title}',fontsize=21,weight='bold',va='top')
    f.text(.055,.89,subtitle,fontsize=12.5,color=MUTED,va='top')
    f.add_artist(plt.Line2D([.055,.95],[.245,.245],transform=f.transFigure,color=LINE))
    f.text(.055,.22,wrap(caption,130),fontsize=10.5,linespacing=1.6,va='top')
    f.text(.95,.025,'LDM RL · 证据截至 2026-09-10 · 本地图文稿',ha='right',fontsize=9,color=MUTED)
    return f

def diagram(f):
    a=f.add_axes([.055,.31,.89,.50]);a.set_axis_off();a.set_xlim(0,1);a.set_ylim(0,1);return a

def box(a,x,y,w,h,title,body,color=BLUE,dashed=False):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.025',facecolor='white',
      edgecolor=color if dashed else LINE,linestyle='--' if dashed else '-',linewidth=1.5))
    a.text(x+.02,y+h-.055,title,va='top',fontsize=16,color=color,weight='bold')
    a.text(x+.02,y+h-.19,body,va='top',fontsize=12.5,linespacing=1.7)

def arrow(a,start,end,color=MUTED):
    a.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':color,'lw':1.6})

def freeze_losses(d):
    r=d['runs']['k8'];path=Path(d['source_root'])/'runs'/r['name']/'train.log'
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
    rows=[]
    for line_no,line in enumerate(raw.decode(errors='replace').splitlines(),1):
        m=re.search(r"step (\d+):.*'train/loss':\s*([^,}]+).*'train/grad_norm':\s*([^,}]+)",line)
        if m:
            loss=float(m[2]);g=float(m[3]);rows.append({'step':int(m[1]),'loss':loss,'grad_finite':math.isfinite(g),'line':line_no})
    assert len(rows)==100 and all(math.isfinite(x['loss']) and not x['grad_finite'] for x in rows)
    result={'run':r['name'],'source':str(path),'sha256':r['source_sha256'],'rows':rows}
    (ROOT/'data/section03_loss_grad.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return rows

def f1(d,rows):
    f=canvas('3.1a','Loss 全部有限，梯度仍可全部为 NaN',
      '同一条 9B SFT 运行、同一批 100 次优化记录；不是两组实验拼接。',
      '读图：蓝点为每次优化尝试的训练 loss（训练目标的标量值）；下方每个红刻线对应同一步的 NaN 梯度范数。'
      '横轴按日志 step 加一，100 次尝试来自 50 轮采样、每轮 2 次优化。Loss 有限只说明前向目标可计算，不能证明反向传播正常。\n'
      '9B 为约 90 亿参数；SFT 为监督微调起点；NaN 为非数值，不能画成数值 0。没有健康对照或效果评测，本图不诊断 NaN 根因。'
      '来源：k8sft_20260909T200313Z/train.log；100 条同一步配对记录与 SHA256 见 data/section03_loss_grad.json。')
    a=f.add_axes([.105,.48,.81,.33]);style(a)
    a.plot([r['step']+1 for r in rows],[r['loss'] for r in rows],color=BLUE,marker='o',ms=3,lw=1.3)
    a.set_ylabel('训练 loss');a.set_xlim(0,101);a.set_title('100 / 100 个 loss 有限',loc='left',fontsize=14)
    b=f.add_axes([.105,.365,.81,.065]);b.set_facecolor(BG)
    b.scatter([r['step']+1 for r in rows],np.zeros(100),marker='|',s=200,color=RED)
    b.set_xlim(0,101);b.set_yticks([]);b.set_xlabel('优化尝试序号（1–100）')
    b.set_title('100 / 100 个梯度范数为 NaN',loc='left',fontsize=13,color=RED)
    for s in b.spines.values():s.set_visible(False)
    return f

def f2(d,rows):
    f=canvas('3.1b','已经定位到证据缺口，还没有定位到 NaN 根因',
      '同一次训练里，loss 的可计算性与更新信号的可用性必须分开检查。',
      '读图：蓝框、红框为同一条 9B SFT 运行已记录的事实；灰色虚线框是这组记录尚未回答的问题。箭头表示检查顺序，不表示已证明的故障传播路径。'
      'Loss 是训练目标值；NaN 是非数值；SFT 是监督微调，9B 约为 90 亿参数。\n'
      '边界：100 个 loss 有限、100 个梯度范数 NaN，只能证明该运行未通过梯度有限性检查；不能确定异常来自哪一层、算子或 loss 配置。'
      '下方方法是建议验证，尚未执行。来源：k8sft_20260909T200313Z/train.log；配对记录见 data/section03_loss_grad.json。')
    a=diagram(f)
    box(a,.015,.47,.28,.43,'前向：已观测','100 个 loss 有限\n目标值能够计算',BLUE)
    box(a,.365,.47,.28,.43,'反向：已观测','100 个梯度范数 NaN\n有限性检查不通过',RED)
    box(a,.715,.47,.27,.43,'原因：未确定','具体层／算子？\n具体 loss 配置？',MUTED,True)
    arrow(a,(.30,.68),(.35,.68));arrow(a,(.65,.68),(.70,.68))
    box(a,.015,.02,.97,.29,'建议验证｜尚未执行','固定同一输入，定位首个非有限中间量，再用仅改变疑似因素的对照复核。',AMBER,True)
    return f

def f3(d,rows):
    r=d['runs']['sft'];v=[g['value'] for g in r['grad']]
    f=canvas('3.2a','没有 NaN，只是通过了最基础的数值检查',
      '一条 9B SFT 运行的 50 次梯度均有限，但裁剪前范数跨越约 10 个数量级。',
      '读图：蓝线为裁剪前梯度范数，衡量反向传播信号的大小；横轴是该运行的优化尝试序号，纵轴为对数刻度。橙色虚线 1 是配置的裁剪阈值，'
      '不是健康或收敛的判定线。范围约为 134 至 8.4×10¹¹。\n'
      '9B 是约 90 亿参数，SFT 是监督微调起点，NaN 是非数值。大梯度值得检查，但本图不能证明参数更新错误、模型没有学习或分子质量下降。'
      '来源：9b-sft_20260909T071143Z/train.log；50 次优化记录、每轮一次更新。原始数值与行号见 data/evidence.json 和 data/metrics.csv。')
    a=f.add_axes([.105,.36,.81,.44]);style(a)
    a.plot(range(1,51),v,color=BLUE,marker='o',ms=3,lw=1.5);a.set_yscale('log');a.set_ylim(.5,2e12)
    a.axhline(1,color=AMBER,ls='--',label='配置的裁剪阈值 = 1');a.legend(loc='lower right')
    a.set(xlabel='优化尝试序号（1–50）',ylabel='裁剪前梯度范数（对数）')
    return f

def f4(d,rows):
    f=canvas('3.2b','要证明发生了可靠更新，需要直接测量参数变化',
      '梯度有限、优化器执行、参数变化与模型行为，是不同层次的证据。',
      '读图：蓝框为选定运行已有的有限梯度记录；三个灰色虚线框表示本章尚未给出联合验证的观测。箭头表示建议核查的链条，不表示这些环节都已经通过。'
      '梯度范数是反向信号大小，参数差分是更新前后同一权重的差，固定输入用于减少输出比较中的混杂。\n'
      '证据：9b-sft_20260909T071143Z 有 50/50 次有限梯度，见 data/evidence.json。建议同时记录实际执行和裁剪后信号，'
      '再测参数差分及固定输入的输出变化。参数变化并不自动代表搜索收益；本图是验证方案，不是已完成修复，也不是宣称这些测量在全项目从未做过。')
    a=diagram(f)
    for i,(title,body) in enumerate([('梯度记录','50 / 50 次有限\n已有数据'),('优化器执行','是否实际执行？\n裁剪后信号如何？'),('参数差分','更新前后改了多少？\n是否可重复？'),('固定输入输出','行为变化能否复核？\n恢复后是否一致？')]):
        box(a,.012+i*.252,.34,.225,.53,title,body,BLUE if i==0 else MUTED,i>0)
        if i<3:arrow(a,(.24+i*.252,.58),(.257+i*.252,.58))
    a.text(.5,.13,'建议验证：固定同一批输入，将这些观测按同一次更新配对。',ha='center',fontsize=15,color=AMBER)
    return f

def f5(d,rows):
    f=canvas('3.3a','组内没有奖励差异，就没有奖励偏好可供学习',
      '这是组相对优化的机制示意；下面的 0 和 1 是说明用数值，不是实验测量。',
      '读图：左右分别示意同一任务的两个候选轨迹。点／柱的高度是奖励；左侧两个点均为 0，右侧候选 B 的奖励为 1。'
      '虚线是零奖励基线。候选 A/B 只是示意名称，0/1 不是从真实日志抽取的样本，不能用于推断成功率或改进幅度。\n'
      '组相对优化用组内奖励差异形成偏好信号，所以全零组没有这项信号。这不排除正则项等其他训练项产生梯度；'
      '右侧有奖励差异也不保证反向传播正常或奖励方向正确。本图仅解释为何需要检查奖励分布，真实计数另由运行日志提供。')
    for x,values,title,color in [(.105,[0,0],'全零组：无法按奖励区分候选',RED),(.575,[0,1],'有差异：奖励给出了相对偏好',BLUE)]:
        a=f.add_axes([x,.40,.33,.36]);style(a);a.bar([0,1],values,color=color,width=.4)
        a.scatter([0,1],values,color=color,s=70,zorder=4);a.axhline(0,color=MUTED,ls='--')
        a.set_ylim(-.15,1.35);a.set_yticks([0,1]);a.set_xticks([0,1],['候选 A','候选 B']);a.set_ylabel('示意奖励（非实测）');a.set_title(title,fontsize=13)
    return f

def f6(d,rows):
    f=canvas('3.3b','奖励可区分与梯度可用，是两项不同的条件',
      '全零奖励组较少的那条运行，仍然记录了 100 / 100 次 NaN 梯度。',
      '读图：柱高为组内候选奖励全部为零的比例，每条运行均为 50 轮、每轮 2 个任务组，共 100 组。K 是每任务候选数，U 是每轮优化次数；'
      '两条均为 9B SFT（约 90 亿参数、监督微调起点）与真实 acquisition 奖励（预期改进评分）。红框记录 K=8/U=2 的梯度结果。\n'
      '限制：K、U 同时变化，各仅一次运行；49% 与 9% 不能解释为 K 的独立效果，也不是分子评估失败率。NaN 为非数值。'
      '全零组的来源还需候选级诊断。来源：9b-sft_20260909T071143Z 与 k8sft_20260909T200313Z；计数与梯度见 data/evidence.json。')
    a=f.add_axes([.10,.39,.37,.39]);style(a);v=[d['runs'][k]['zero_groups'] for k in ['sft','k8']]
    a.bar([0,1],v,color=[AMBER,BLUE],width=.5)
    for i,n in enumerate(v):a.text(i,n+3,f'{n} / 100 组\n{n}%',ha='center',fontsize=16)
    a.set_ylim(0,75);a.set_xticks([0,1],['K=2，U=1','K=8，U=2']);a.set_ylabel('全零奖励组占比（%）')
    b=f.add_axes([.55,.39,.39,.40]);b.set_axis_off();b.set_xlim(0,1);b.set_ylim(0,1)
    box(b,.02,.14,.94,.73,'K=8，U=2 的另一项结果','9 / 100 个全零奖励组\n但 100 / 100 次梯度 NaN\n\n奖励诊断不能替代梯度检查。',RED)
    return f

def f7(d,rows):
    f=canvas('3.4a','修复是否有效，需要只改变疑似因素的对照',
      '拟议验证方案，不是新实验结果；不预设某项 loss 配置已经被证明有错。',
      '读图：两个框是计划中的对照分支。共同条件保持一致，只有被检验的配置项不同；箭头指向双方都要记录的结果。'
      'Loss 是训练目标，参数差分是同一次更新前后的权重变化。图中没有测量值、预期改善幅度或已完成标记。\n'
      '判断规则：先在同一输入上复现异常，再检查单项改动能否重复改变梯度与实际更新，并恢复原设置验证异常能否再现。'
      '环境中的可变历史和随机性也应控制。该设计可检验特定假说，但不能仅凭一次通过推广到长程收敛或搜索收益。')
    a=diagram(f)
    a.text(.5,.96,'共同条件：同一模型权重、输入、随机性、环境状态与其余训练设置',ha='center',va='top',fontsize=13)
    box(a,.02,.43,.44,.38,'对照：原设置','先复现同一种异常',BLUE)
    box(a,.54,.43,.44,.38,'干预：只改疑似因素','例如一项待检验的 loss 配置',AMBER,True)
    arrow(a,(.24,.41),(.24,.31));arrow(a,(.76,.41),(.76,.31))
    box(a,.02,.015,.96,.27,'双方使用同一套读数','Loss、梯度、优化器执行、参数差分；重复运行后再判断是否支持该解释。',BLUE)
    return f

def f8(d,rows):
    f=canvas('3.4b','已识别异常，不等于已证明原因、修复或最终收益',
      '当前证据的价值，是明确下一步该验证什么，而不是提前宣布训练问题已经解决。',
      '读图：上方蓝／红框概括选定记录中的观测；下方三个灰色虚线框是本章尚未建立的结论。箭头表示后续论证所需的证据顺序，不表示完成进度。'
      'NaN 为非数值，RL 为强化学习，SFT 为监督微调；“未建立”不等于结论为假。\n'
      '观测依据：9B SFT 的有限 loss/NaN 梯度来自 k8sft_20260909T200313Z；大幅变化的有限梯度来自 9b-sft_20260909T071143Z；'
      '全零组为这两条运行的 49/100 与 9/100。没有完成根因对照、可靠更新验证和同预算 RL/SFT 搜索效果比较，不能据此声称收敛或收益。')
    a=diagram(f)
    box(a,.02,.59,.45,.34,'已观测：梯度数值问题','NaN；或有限但变化很大的梯度',RED)
    box(a,.53,.59,.45,.34,'已观测：奖励信号缺失','部分任务组内候选奖励全为零',BLUE)
    for i,(title,body) in enumerate([('原因尚未证实','需针对假说的匹配对照'),('修复尚未验收','需实际更新与重复验证'),('收益尚未证明','需同预算 RL / SFT 评测')]):
        box(a,.02+i*.33,.07,.30,.33,title,body,MUTED,True)
        if i<2:arrow(a,(.325+i*.33,.23),(.343+i*.33,.23))
    return f

SECTIONS=[
 ('3.1 Loss 是有限数值，为什么还不能说明训练正常？',[
 '在一条 9B SFT 分子搜索训练中，loss 能正常输出，但与它逐步配对的 100 次优化记录，梯度范数全部是 NaN。**前向目标能算出一个数，并不保证反向传播能提供可用的更新信号。** 这条运行虽然完成了 50 轮采样，仍不能凭 loss 有限或运行完成来证明训练成功。',
 '**这里已经证实的是梯度有限性检查失败，尚未证实的是失败的具体原因。** 仅凭这些记录，不能断言某一层、某个算子或某项 loss 配置有问题。下一步应在固定输入上定位首个非有限中间量，再用单因素对照检验解释；这是拟议验证，不是已完成的修复。']),
 ('3.2 没有 NaN 的运行，是否就已经解决了问题？',[
 '另一条 9B SFT 运行的 50 次梯度记录全部有限，但裁剪前范数从约 134 跨到 8.4×10¹¹。它说明检查不能只停在“有没有 NaN”：**有限数值仍可能伴随很大的训练信号变化。** 不过，这个范围本身不足以证明优化器失效，也不能据此判断模型已经收敛或没有学习。',
 '梯度裁剪、优化器状态和数值精度都会影响最终更新。要判断这条 9B 运行是否可靠地改变了模型，需要把同一次更新的梯度处理、优化器执行和参数差分连起来，再核对固定输入上的输出变化。**本章尚未提供完成这组联合检查的证据，所以不把“梯度有限”写成“问题已解决”。**']),
 ('3.3 奖励没有差异时，训练缺少的是什么？',[
 '组相对优化依靠同一任务下候选轨迹的奖励差异来形成偏好。如果一组候选全部拿到零奖励，就没有这项信号供模型学习。**这与优化器能否计算梯度是两个问题。** 下面用明确标为示意的候选奖励解释机制，不把人为数字当作实验结果。',
 '两条 9B SFT 运行中的全零奖励组分别占 49% 和 9%，但后一条运行的 100 次梯度仍全部为 NaN。**奖励更能区分候选，并不保证更新信号可用；梯度正常也不保证奖励指向更好的分子。** 当前计数还不能区分评估失败、没有改进或候选相似等来源，因此不能把增大采样数当成已经验证的修法。']),
 ('3.4 什么结果才能说明这些问题真的被解决了？',[
 '如果怀疑某项 loss 配置造成训练异常，就应在同一模型和输入下，只改变这一项，检查异常是否可重复地消失、优化器是否实际更新，以及参数变化是否一致。**修复必须由前后可比的实验支持。** 短程验证通过后，还需要更长训练中的稳定性证据，以及同预算的分子搜索评测；下面是验证设计，不是执行结果。',
 '**当前结论是：已经识别出梯度数值异常和奖励偏好信号缺失，但尚未建立完整的根因与修复证据链。** 这两类观测分别约束训练能否更新、奖励能否指导更新；只有把它们连到可靠的参数变化，并最终比较 RL 与 SFT 的搜索成绩，才能进一步讨论收敛和实际收益。'])]
FUNCS=[f1,f2,f3,f4,f5,f6,f7,f8]

if __name__=='__main__':
    setup_font();OUT.mkdir(exist_ok=True)
    d=json.loads((ROOT/'data/evidence.json').read_text());rows=freeze_losses(d)
    text=['# 3. 训练为什么还不能证明有效学习：从异常结果追到优化过程']
    for s,(title,paras) in enumerate(SECTIONS):
        text.append('## '+title)
        for j,para in enumerate(paras):
            i=s*2+j;f=FUNCS[i](d,rows)
            f.savefig(OUT/f'figure_{i+1}.png',dpi=170);f.savefig(OUT/f'figure_{i+1}.svg');plt.close(f)
            text.extend([para,f'![图 3.{s+1}{"ab"[j]}：证据、解释边界与验证办法](figures_section03/figure_{i+1}.png)'])
    (ROOT/'SECTION_03.md').write_text('\n\n'.join(text)+'\n')
