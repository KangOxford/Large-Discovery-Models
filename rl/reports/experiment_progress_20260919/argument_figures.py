"""One claim, one evidence unit, and one proposed resolution per section.

Recommendations below are proposals, not newly executed experiments.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from build_report import canvas, style, INK, MUTED, BLUE, TEAL, RED, AMBER, BG, LINE

TITLE = 'LDM RL：从流程跑通，到证明有效学习'
THESIS = ('这批工作的进展，是把问题从“代码能否运行”推进到了“模型是否有效学习”。'
          '下一步应先验证可靠的参数更新，再用同预算的分子搜索结果判断 RL 是否优于 SFT。')

SECTIONS = [
 ('2.1 运行结束不能作为训练成功的验收标准',
  '在分子搜索强化学习中，程序完成、模型存档和有效优化是三件事。1.5B 的试跑已有有限梯度记录，'
  '但一条完成全部采样并保存模型的 9B 运行，100 次梯度记录全部是 NaN。我的判断是：流程已经能走完，'
  '训练成功的验收却必须增加参数更新与恢复验证，不能只看退出状态或文件是否存在。'),
 ('2.2 消除 NaN 之后，还要验证参数是否被可靠地更新',
  '9B 模型也出现过全程有限的梯度，但一条 SFT 运行的梯度范数从约 134 跨到 8.4×10¹¹。'
  '这个量级变化值得排查，却不足以单凭它断言训练失效。需要把裁剪前后梯度、优化器实际执行和参数改变量连起来核对；'
  '否则，“没有 NaN”仍然只是一个过于宽松的成功标准。'),
 ('2.3 奖励必须能区分候选，训练才有组内偏好可学',
  '分子搜索的组相对优化，需要同一任务的候选轨迹拿到不同奖励。两条 9B SFT 运行中，全零奖励组分别占 49% 和 9%。'
  '这说明奖励无法区分候选是一个需要单独解决的问题；它还没有说明哪种配置最终更好。'
  '先把零奖励对应的候选、评估失败和无改进情况记录清楚，再决定扩大采样还是修改奖励。'),
 ('2.4 要判断 K=8 的作用，必须拆开同时变化的训练条件',
  '把每个任务的候选数从 2 改到 8 时，已有比较也把每轮更新次数从 1 改成了 2。'
  '而另一条候选数为 2、更新次数为 2 的运行只有 29 轮，不能直接补成等预算对照。'
  '我的判断是：49% 与 9% 是值得追查的线索，尚不足以推荐 K=8。'
  '冻结模型的奖励诊断与健康训练下的单因素对照，应分别回答奖励可区分性和学习收益。'),
 ('2.5 RL 的收益必须由同预算的分子搜索成绩证明',
  '这个项目最终要回答的是：在相同搜索预算下，继续强化学习的模型能否比仅监督微调的模型找到更好的分子。'
  '目前核查的材料还没有提供完整的 G12D/G12C 对照结果。'
  '因此，训练稳定性与奖励诊断都应服务于这张最终对照：固定评测协议、配对随机种子，比较 Pareto 超体积，而不把训练奖励当作效果结论。'),
]

def box(ax,x,y,w,h,title,body,color=BLUE,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.02',
                              facecolor='white',edgecolor=color if dashed else LINE,
                              linestyle='--' if dashed else '-',linewidth=1.5))
    ax.text(x+.022,y+h-.035,title,fontsize=16,color=color,weight='bold',va='top')
    ax.text(x+.022,y+h-.14,body,fontsize=13,va='top',linespacing=1.7)

def resolution(fig,text):
    ax=fig.add_axes([.045,.285,.91,.105]);ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((.003,.04),.989,.92,boxstyle='round,pad=.003,rounding_size=.04',
                              facecolor='#EAF0F7',edgecolor=LINE))
    ax.text(.015,.76,'建议验证｜尚未执行',fontsize=12,color=BLUE,weight='bold',va='top')
    ax.text(.015,.44,text,fontsize=13,color=INK,va='top',linespacing=1.45)

def figure1(d):
    cap=('证据：两行各是一条选定的真实分子评估运行，并非模型规模的控制实验。蓝色框记录完成的采样轮次和模型存档；'
      '右侧绿／红框分别表示有限／非有限梯度记录。箭头表示依次检查三类证据，数字单位不同，不构成转化率。'
      '1.5B/9B 为约 15 亿/90 亿参数；SFT 为监督微调；NaN 为非数值；RL 为强化学习。\n'
      '判断边界：1.5B 的 60/60 次有限梯度只支持该试跑通过数值有限性检查；9B 的 100/100 次 NaN 使该运行无法凭完成状态通过训练验收。'
      '这不证明所有 9B 运行都失败，也没有直接测量模型文件的参数改变量。下方是建议的验收办法，不是已经完成的修复。\n'
      '来源：15b-fmt10_20260902T175126Z 与 k8sft_20260909T200313Z 的 train.log、episodes.jsonl 和 ckpt 目录。'
      '两条均使用真实 acquisition 奖励（预期改进评分）。完整源路径、行号和文件校验值见随附 data/。')
    f=canvas('2.1','训练验收应检查优化证据，而不只检查“跑完了”',
      '反例已经存在：同一条 9B 运行完成全部采样并存档，却记录了 100 次 NaN 梯度。',cap)
    ax=f.add_axes([.045,.445,.91,.40]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    for y,key,label in [(.56,'pilot','1.5B 试跑'),(.06,'k8','9B SFT 试跑')]:
        r=d['runs'][key];n=len(r['rollout']);total=len(r['grad']);finite=r['finite_grad_count']
        ax.text(.01,y+.38,label,fontsize=17,weight='bold')
        box(ax,.18,y,.23,.39,'采样完成',f'{n} / {r["args"]["num_rollout"]} 轮',BLUE)
        box(ax,.47,y,.20,.39,'模型已保存',f'{len(r["checkpoint_dirs"])} 个存档目录',BLUE)
        text=f'{finite} / {total} 次有限' if finite else f'{total} / {total} 次 NaN'
        box(ax,.73,y,.255,.39,'再看梯度记录',text,TEAL if finite else RED)
        for x0,x1 in [(.42,.455),(.68,.715)]:
            ax.add_patch(FancyArrowPatch((x0,y+.17),(x1,y+.17),arrowstyle='->',mutation_scale=18,color=MUTED))
    resolution(f,'把有限梯度、优化器实际执行、参数变化及保存后恢复共同列为训练验收项。\n只有退出成功或存档存在，不足以验收；NaN 的来源仍需定位。')
    return f

def figure2(d):
    cap=('证据：左图为一条 9B SFT（约 90 亿参数、监督微调起点）运行的全部 50 次优化记录。点是裁剪前梯度范数，'
      '表示反向传播信号的大小；纵轴采用对数刻度。虚线 1 是配置的裁剪阈值，不是判定训练健康的阈值。'
      'NaN 指非数值，这条运行的 50 次记录均不是 NaN，也不是无穷大。\n'
      '判断边界：134 至 8.4×10¹¹ 的范围提示需要验证更新，但梯度大小本身不能证明更新错误，更不能证明分子质量升降。'
      '右侧虚线框是尚需补充的观测，框内箭头表示建议的核查顺序；不宣称已有参数差分结果，也不把范围差异归因于单个组件。\n'
      '来源：9b-sft_20260909T071143Z/train.log，50 轮采样、每任务 2 条候选轨迹、每轮 1 次优化，真实 acquisition 奖励（预期改进评分）。'
      '完整记录与行号见 data/metrics.csv；该单次运行不代表所有 9B 模型。')
    f=canvas('2.2','“梯度有限”还不能替代对实际参数更新的检查',
      '一个通过有限性检查的 9B 运行，裁剪前梯度仍跨越约 10 个数量级。',cap)
    r=d['runs']['sft'];v=[g['value'] for g in r['grad']]
    ax=f.add_axes([.085,.465,.48,.365]);style(ax)
    ax.plot(range(1,len(v)+1),v,color=BLUE,marker='o',ms=3,lw=1.6)
    ax.axhline(1,color=AMBER,ls='--',label='配置的裁剪阈值 = 1')
    ax.set_yscale('log');ax.set_ylim(.5,2e12);ax.set_yticks([1,1e3,1e6,1e9,1e12])
    ax.set(xlabel='优化尝试序号（1–50）',ylabel='裁剪前梯度范数（对数）');ax.legend(loc='lower right',fontsize=11)
    ax.text(.04,.95,'50 / 50 次有限\n范围：约 134—8.4×10¹¹',transform=ax.transAxes,va='top',fontsize=14,
            bbox={'facecolor':'white','edgecolor':LINE,'alpha':.92})
    b=f.add_axes([.63,.465,.325,.365]);b.set_axis_off()
    box(b,.025,.04,.94,.91,'需要补上的直接证据',
      '同一批输入\n↓\n裁剪前后梯度与优化器执行\n↓\n更新前后参数差分\n↓\n固定输入上的输出变化',AMBER,True)
    resolution(f,'先在固定输入上核对梯度、实际更新和参数差分，再验证恢复后的模型输出。\n若某一环异常，就针对该环定位；暂不从大梯度直接推断模型没有学习。')
    return f

def figure3(d):
    cap=('证据：每条运行包含 50 轮采样、每轮 2 个任务组，分母均为 100 组。每组的候选数为 K；U 为每轮优化次数。'
      '左图柱高是所有候选奖励都为零的组占比。右图三类零奖励原因是待验证的候选解释，并非已经测出的比例；箭头表示需要补充日志来区分它们。\n'
      '定义与判断边界：9B SFT 为约 90 亿参数、监督微调起点；组相对优化使用同组候选间的奖励差异，全零组没有这种偏好信号。'
      '两条都是 acquisition 奖励（预期改进评分），但 K 与 U 同时变化，各仅一次运行；K=8 的梯度还全部为 NaN（非数值）。'
      '因此较少全零组不能等同于成功优化或分子质量提升；奖励不全为零的组也不保证所有候选都有可用差异。\n'
      '来源：9b-sft_20260909T071143Z 与 k8sft_20260909T200313Z 的 train.log。用逐轮 count_no_gradient 计数，并与正计数上的 '
      'count_lt_eps、count_0.0 核对；分别为 49/100 和 9/100。完整数据见 data/evidence.json。')
    f=canvas('2.3','奖励能否区分候选，是一个需要单独解决的问题',
      '全零奖励组没有组内偏好信号；先解释它为什么出现，再决定如何增加有效反馈。',cap)
    a=f.add_axes([.085,.475,.35,.35]);style(a)
    values=[d['runs'][k]['zero_groups'] for k in ['sft','k8']]
    a.bar([0,1],values,color=[AMBER,BLUE],width=.5)
    for i,v in enumerate(values):a.text(i,v+3,f'{v} / 100 组\n{v}%',ha='center',fontsize=17)
    a.set_ylim(0,75);a.set_yticks([0,25,50,75]);a.set_xticks([0,1],['K=2，U=1','K=8，U=2']);a.set_ylabel('全零奖励组占比（%）')
    b=f.add_axes([.53,.445,.425,.405]);b.set_axis_off()
    box(b,.01,.06,.965,.87,'待区分的原因，而非既定诊断',
      '候选没有被成功评估？\n评估成功，但没有获得改进奖励？\n候选过于相似，评分都相同？\n\n↓ 用候选级记录回答，不能只看组均值',AMBER,True)
    resolution(f,'补齐每条候选的有效性、评估状态、原始奖励，以及每组奖励的标准差。\n先定位信号在哪一环消失，再选采样、评估或奖励的修法，并重测零组比例。')
    return f

def figure4(d):
    cap=('读图：横轴 K 是每个任务的候选轨迹数，纵轴 U 是每轮采样的优化次数。框内比例是全零奖励组占比，括号给出实际计数。'
      '蓝框是各有 50 轮的两条运行，橙框只有 29 轮，灰色虚线框表示本次选定证据中缺少该格。斜箭头连接原先的两条比较：跨行又跨列，所以同时改变了 K 和 U。\n'
      '范围与定义：均为 9B SFT（约 90 亿参数、监督微调起点）、真实 acquisition 奖励（预期改进评分），每轮 2 个任务组。'
      '这不是完整析因实验，也不声称其他目录里从未跑过缺失配置。组计数不是独立重复；50 轮的 K=2 与 K=8 分别生成 200 和 800 条候选轨迹，不能据此比较采样效率。\n'
      '来源：9b-sft_20260909T071143Z（49/100）、9b-sft-gbs2_20260909T094256Z（37/58）、k8sft_20260909T200313Z（9/100）的逐轮计数。'
      '图下两个验证方案尚未执行；即使计数发生在本轮更新之前，此前的更新仍可能影响后续轮次，不能自动消除 U 的影响。')
    f=canvas('2.4','49% 与 9% 的差异，还不能单独归因于候选数 K',
      '现有比较同时改变候选数和更新次数；补一条短运行，也不能自动得到完整对照。',cap)
    ax=f.add_axes([.13,.435,.79,.42]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.27,.98,'K=2：每任务 2 条候选',ha='center',va='top',fontsize=15)
    ax.text(.75,.98,'K=8：每任务 8 条候选',ha='center',va='top',fontsize=15)
    ax.text(-.08,.68,'U=1\n每轮 1 次更新',ha='center',va='center',fontsize=13)
    ax.text(-.08,.23,'U=2\n每轮 2 次更新',ha='center',va='center',fontsize=13)
    box(ax,.055,.53,.40,.32,'49%（49 / 100 组）','50 轮 · 一次运行',BLUE)
    box(ax,.545,.53,.40,.32,'缺少匹配运行','本次选定证据中未提供',MUTED,True)
    r=d['runs']['sft2'];n=len(r['rollout'])*2;z=int(sum(x['no_gradient_groups'] for x in r['rollout']))
    box(ax,.055,.075,.40,.32,f'{z/n:.1%}（{z} / {n} 组）','只有 29 轮 · 尚不匹配',AMBER)
    box(ax,.545,.075,.40,.32,'9%（9 / 100 组）','50 轮 · 一次运行',BLUE)
    ax.add_patch(FancyArrowPatch((.46,.53),(.54,.40),arrowstyle='->',mutation_scale=20,color=RED,lw=2))
    ax.text(.62,.465,'原比较同时改变 K、U',ha='center',fontsize=12,color=RED)
    resolution(f,'奖励诊断：冻结同一模型与环境状态，只改变 K，重复采样看可区分性。\n学习收益：待训练可靠后，固定 U 与评测预算比较 K，并明确匹配的训练资源口径。')
    return f

def figure5(d):
    cap=('读图：两个评测分支分别回答同目标效果与迁移效果。每个框内都要求 SFT-only（仅监督微调）与 SFT+RL（继续强化学习）比较；'
      '下方框是判定收益的共同要求，连线只表示两条分支都受此约束。灰色虚线表示尚缺可比结果，不代表分数为 0。\n'
      '协议：训练目标为 KRAS G12D；G12C 是另一 KRAS 突变体，作为迁移评测。两者用各自的活性模型，共用 8UN5 对接受体。'
      '按原训练计划，每种策略、每个目标采用 5 个随机种子，搜索预算 80；HV（Pareto 超体积）衡量对接与活性两项目标的综合搜索结果，越高越好。'
      '建议配对种子、固定超体积参照点，报告每个种子的差值和不确定性；这些补充是拟议的比较规则，不是已跑结果。\n'
      '证据范围：TRAINING_PLAN.md、2026-09-05 主目标审计、截至 09-10 的运行记录及 PR 2/5/6 后续讨论，结合 09-19 对 ldm_rl/results 清单的核查，'
      '未找到完整可比结果；不是全服务器普查。仅在 G12D 上有收益不能据此宣称 G12C 迁移成功；有限梯度或奖励升高也不能替代 HV 对照。')
    f=canvas('2.5','只有同预算的搜索结果，才能支持“RL 比 SFT 更好”',
      '把技术修复收束到同一个问题：相同评估资源下，模型能否找到更好的分子？',cap)
    ax=f.add_axes([.055,.43,.89,.43]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    for x,title in [(.025,'G12D：同目标效果'),(.54,'G12C：迁移效果')]:
        box(ax,x,.48,.43,.45,title,'SFT-only vs SFT+RL\n各 5 个种子 · 搜索预算 80\n可比 HV 结果：未找到',MUTED,True)
    for x in [.24,.755]:
        ax.add_patch(FancyArrowPatch((x,.46),(x,.36),arrowstyle='->',mutation_scale=18,color=MUTED))
    box(ax,.025,.04,.945,.30,'共同判据：比较配对的 HV 差值',
        'RL 与 SFT 使用相同预算、评估器和 HV 参照点；报告逐种子差值及不确定性。',BLUE)
    resolution(f,'先锁定搜索预算的计数口径和对照协议，再评估通过训练验收的模型。\n只有同预算 HV 的比较支持收益时，才报告效果提升；迁移结论单独检验。')
    return f

FIGURES=[figure1,figure2,figure3,figure4,figure5]
