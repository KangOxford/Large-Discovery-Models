"""Freeze selected raw-log evidence, then draw four standalone Chinese figures.

python build_report.py --extract /path/to/ldm_rl
python build_report.py
No GPU, training, model loading, or modification of source experiments.
"""
from pathlib import Path
import argparse, base64, csv, hashlib, html, json, math, re, unicodedata
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
RUNS = {
 'pilot': '15b-fmt10_20260902T175126Z',
 'base': '9b-base_20260909T073721Z',
 'sft': '9b-sft_20260909T071143Z',
 'sft2': '9b-sft-gbs2_20260909T094256Z',
 'k8': 'k8sft_20260909T200313Z',
}
INK='#172B43'; MUTED='#506277'; BLUE='#2563A6'; TEAL='#188777'
RED='#B84445'; AMBER='#AC7024'; BG='#F7F9FC'; LINE='#D9E1EB'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def extract(root):
    out={'checked_utc':'2026-09-19', 'source_root':str(root.resolve()), 'runs':{},
         'scope':'Five selected diagnostic runs; not independent seeds or the full P2 matrix.',
         'sources':[]}
    rows=[]
    for key,name in RUNS.items():
        d=root/'runs'/name; p=d/'train.log'; text=p.read_text(errors='replace')
        def arg(k):
            m=re.search(r'^\s*'+re.escape(k)+r'\s+\.+\s+(.*)$',text,re.M)
            return m.group(1).strip() if m else None
        ep=d/'episodes.jsonl'; spec=json.loads(json.loads(ep.read_text().splitlines()[0])['prompt'])
        r={'name':name,'args':{k:arg(k) for k in ['n_samples_per_prompt','num_rollout',
          'global_batch_size','rollout_batch_size','clip_grad','grpo_advantage_std_mode']},
          'reward':spec['reward'],'mode':spec['mode'],'grad':[],'rollout':[],
          'source_sha256':sha(p),'source_bytes':p.stat().st_size}
        for lineno,line in enumerate(text.splitlines(),1):
            m=re.search(r"(?:step|rollout) (\d+): (\{.*\})",line)
            if not m: continue
            metrics={k:float(v) for k,v in re.findall(
                r"'([^']+)':\s*(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))",m.group(2))}
            if 'train/grad_norm' in metrics:
                v=metrics['train/grad_norm']; item={'index':int(m[1]),'line':lineno,
                  'value':v if math.isfinite(v) else None,'nonfinite':not math.isfinite(v)}
                r['grad'].append(item)
                rows.append([key,name,'grad_norm',item['index'],item['value'],item['nonfinite'],lineno])
            if 'rollout/raw_reward' in metrics:
                item={'index':int(m[1]),'line':lineno,'raw_reward':metrics['rollout/raw_reward'],
                  'no_gradient_groups':metrics.get('rollout/zero_std/count_no_gradient'),
                  'lt_eps_groups':metrics.get('rollout/zero_std/count_lt_eps'),
                  'exact_zero_groups_emitted':metrics.get('rollout/zero_std/count_0.0')}
                r['rollout'].append(item)
                rows.append([key,name,'raw_reward',item['index'],item['raw_reward'],False,lineno])
                rows.append([key,name,'no_gradient_groups',item['index'],item['no_gradient_groups'],'',lineno])
        latest=d/'ckpt/latest_checkpointed_iteration.txt'
        r['checkpoint_last_iteration']=int(latest.read_text()) if latest.exists() else None
        r['checkpoint_dirs']=[f.name for f in sorted((d/'ckpt').glob('iter_*')) if f.is_dir()]
        r['finite_grad_count']=sum(not g['nonfinite'] for g in r['grad'])
        r['nonfinite_grad_count']=sum(g['nonfinite'] for g in r['grad'])
        # The always-emitted no-gradient counter supplies the denominator. Never
        # silently replace missing exact-zero keys with zeros across log versions.
        if key in ['sft','k8']:
            assert len(r['rollout'])==50 and len({v['index'] for v in r['rollout']})==50
            for v in r['rollout']:
                assert v['no_gradient_groups'] is not None
                assert v['no_gradient_groups']==v['lt_eps_groups']
                if v['no_gradient_groups']>0:
                    assert v['exact_zero_groups_emitted']==v['no_gradient_groups']
                elif v['exact_zero_groups_emitted'] is not None:
                    assert v['exact_zero_groups_emitted']==0
            r['zero_groups']=int(sum(v['no_gradient_groups'] for v in r['rollout']))
            r['groups']=len(r['rollout'])*int(r['args']['rollout_batch_size'])
        out['runs'][key]=r
        for source in [p,ep,latest]:
            if source.exists():out['sources'].append({'path':str(source.relative_to(root)),
                'sha256':sha(source),'bytes':source.stat().st_size})
    assert out['runs']['k8']['nonfinite_grad_count']==100
    assert out['runs']['sft']['zero_groups']==49 and out['runs']['k8']['zero_groups']==9
    assert out['runs']['pilot']['finite_grad_count']==60
    # Preserve the historical protocol audit as a dated source, not as a current
    # assertion that no finite 9B gradients have ever existed.
    for rel in ['plan/MAIN_OBJECTIVE_AUDIT.md','results/main_objective_manifest.json']:
        p=root/rel
        out['sources'].append({'path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
    out['evaluation_scope']={
      'historical_audit_date':'2026-09-05',
      'checked_paths':['ldm_rl/results (all files)','ldm_rl/runs (top-level and depth 2 filenames)',
                       'KangOxford/Large-Discovery-Models PR 2, 5, 6 (reviewed 2026-09-19)'],
      'claim':'No complete matched budget-80 G12C/G12D SFT-only vs SFT+RL result found in inspected evidence; not a server-wide absence claim.',
      'protocol':{'metric':'Pareto hypervolume','budget':80,'evaluation_seeds_per_arm':5,
                  'train_target':'G12D','eval_targets':['G12D','G12C']}}
    inventory=[]
    for p in sorted((root/'results').rglob('*')):
        if p.is_file():inventory.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size})
    out['results_inventory']=inventory
    (HERE/'data').mkdir(exist_ok=True)
    (HERE/'data/evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (HERE/'data/metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['key','run','metric','index_zero_based','value','nonfinite','source_line']);w.writerows(rows)
    return out

def setup_font():
    candidates=[Path('/tmp/ldm_report_NotoSansCJKsc-Regular.otf'),
                Path.home()/'.cache/ldm_report/NotoSansCJKsc-Regular.otf']
    f=next((x for x in candidates if x.exists()),None)
    if f is None:
        f=candidates[-1];f.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',f)
    font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(f)).get_name(),
      'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
      'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
      'pdf.fonttype':42,'svg.fonttype':'path','savefig.facecolor':BG})

def wrap(s,width=124):
    lines=[]
    for paragraph in s.split('\n'):
        line='';n=0
        for c in paragraph:
            k=2 if unicodedata.east_asian_width(c) in 'WF' else 1
            if n+k>width:lines.append(line);line='';n=0
            line+=c;n+=k
        lines.append(line)
    return '\n'.join(lines)

def canvas(number,title,subtitle,caption):
    fig=plt.figure(figsize=(16,11.5),facecolor=BG)
    fig.text(.045,.955,f'图 {number}  |  {title}',fontsize=24,weight='bold',va='top')
    fig.text(.045,.907,subtitle,fontsize=14,color=MUTED,va='top')
    fig.add_artist(plt.Line2D([.045,.955],[.25,.25],transform=fig.transFigure,color=LINE))
    fig.text(.045,.226,wrap(caption),fontsize=11.5,linespacing=1.65,va='top')
    fig.text(.955,.025,'LDM RL · 实验记录截至 2026-09-10 · 2026-09-19 核验',ha='right',fontsize=10,color=MUTED)
    return fig

def style(ax):
    ax.set_facecolor(BG);ax.grid(axis='y',alpha=.3,color=LINE);ax.set_axisbelow(True)
    for sp in ax.spines.values():sp.set_color(LINE)

def card(ax,x,y,w,h,title,body,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.025',
                              facecolor='white',edgecolor=LINE,linewidth=1.3))
    ax.text(x+.025,y+h-.055,title,fontsize=15,color=color,weight='bold',va='top')
    ax.text(x+.025,y+h-.15,body,fontsize=13,va='top',linespacing=1.7)

def fig1(data):
    r=data['runs']['pilot'];positive=sum(x['raw_reward']>0 for x in r['rollout'])
    cap=('读图：上方箭头只表示流程顺序，四个框分别对应生成、真实评估、优化记录和保存；它们不是四组对照实验。'
      '左下每个点是一次采样轮次（rollout）的平均原始奖励；右下每个点是一次优化尝试的裁剪前梯度范数，越大并不代表越好。'
      '虚线 1 是该运行的梯度裁剪阈值，不是学习有效性的判定线；纵轴为对数刻度。1.5B 指约 15 亿参数。\n'
      '范围：单次小模型链路验证，K=2（每个任务两条候选轨迹），30 轮采样、60 次优化记录。奖励是 acquisition（候选的预期改进评分），'
      '不是最终分子搜索成绩；real 模式使用 Vina 分子对接和 G12D 活性预测。有限梯度和 checkpoint（模型存档）只能支持链路验证，不能证明 RL 提升。\n'
      '来源：15b-fmt10_20260902T175126Z/{train.log, episodes.jsonl, ckpt/latest_checkpointed_iteration.txt}；'
      '完整路径、原始行号与 SHA256 校验值随报告 data/ 提供。')
    fig=canvas('2.1','小模型：生成、真实评估、优化和保存均有运行证据',
      '一次 1.5B 试跑的完整记录；回答“链路能否工作”，不回答“分子是否更好”。',cap)
    ax=fig.add_axes([.045,.62,.91,.23]);ax.set_axis_off()
    cards=[('生成候选','30 / 30 轮采样\n每轮 2 组 × 2 条轨迹'),('真实评估',f'{positive} / 30 轮奖励 > 0\n分子对接 + 活性预测'),
           ('优化记录','60 / 60 梯度范数有限\n每轮 2 次优化尝试'),('模型保存',f'{len(r["checkpoint_dirs"])} 个存档目录\n最后采样轮次：30')]
    for i,(t,b) in enumerate(cards):
        card(ax,i*.255,.02,.225,.93,t,b,TEAL)
        if i<3:ax.annotate('',xy=(i*.255+.251,.5),xytext=(i*.255+.231,.5),arrowprops={'arrowstyle':'->','color':MUTED})
    a=fig.add_axes([.085,.345,.385,.235]);style(a)
    a.plot(np.arange(1,31),[x['raw_reward'] for x in r['rollout']],color=BLUE,marker='o',ms=3,lw=1.7)
    a.set(xlabel='采样轮次（1–30）',ylabel='每轮平均原始奖励',title='真实评估产生了非零奖励')
    b=fig.add_axes([.57,.345,.365,.235]);style(b)
    b.plot(np.arange(1,61),[x['value'] for x in r['grad']],color=TEAL,marker='o',ms=2,lw=1.4)
    b.axhline(1,color=AMBER,ls='--',label='配置的裁剪阈值 = 1');b.set_yscale('log')
    b.set(xlabel='优化尝试序号（1–60）',ylabel='裁剪前梯度范数（对数）',title='60 次梯度记录均为有限数值');b.legend(fontsize=10)
    return fig

def fig2(data):
    cap=('读图：左侧折线展示各运行的裁剪前梯度范数；纵轴每升一格代表放大 100 倍，横轴是各自的优化尝试序号，不是相同训练预算。'
      '虚线 1 是共同的梯度裁剪阈值，不是有效学习的分界。K=8 的梯度全部为 NaN（非数值），不能放到对数轴上，故只在右图显示。'
      '右侧蓝段表示有限梯度记录，红段表示 NaN/Inf；长度是次数，数字标明分子／分母。\n'
      '定义与限制：9B 指约 90 亿参数；base 是未做本任务监督微调的起点，SFT 是监督微调后的起点。K 是每个任务的候选轨迹数，U 是每轮采样对应的优化次数。'
      '四条运行的初始化、K、U 和长度不同，是数值诊断而非模型优劣对照。有限不等于健康：SFT/K=2/U=1 的范数约为 134 至 8.4×10¹¹。\n'
      '来源：9b-base_20260909T073721Z、9b-sft_20260909T071143Z、9b-sft-gbs2_20260909T094256Z、'
      'k8sft_20260909T200313Z 的 train.log。选取这四条说明最新进展及失败，不代表全量运行；精确记录见 data/metrics.csv。')
    fig=canvas('2.2','9B：出现过有限梯度，但最新 K=8 仍为 100 / 100 次 NaN',
      '保存模型文件和完成采样，都不足以证明训练成功。',cap)
    a=fig.add_axes([.085,.375,.405,.46]);style(a)
    configs=[('base','base · K=2 · U=1',AMBER),('sft','SFT · K=2 · U=1',BLUE),
             ('sft2','SFT · K=2 · U=2',TEAL),('k8','SFT · K=8 · U=2',RED)]
    for key,label,color in configs[:3]:
        r=data['runs'][key];a.plot([x['index']+1 for x in r['grad']],[x['value'] for x in r['grad']],
                                  color=color,lw=1.6,marker='o',ms=2,label=label)
    a.axhline(1,color=MUTED,ls='--',lw=1);a.set_yscale('log');a.set_ylim(.5,2e12)
    a.set_yticks([1,1e2,1e4,1e6,1e8,1e10,1e12]);a.set(xlabel='各运行的优化尝试序号',ylabel='裁剪前梯度范数（对数）')
    a.legend(loc='upper right',fontsize=10)
    b=fig.add_axes([.70,.375,.25,.46]);style(b)
    for i,(key,label,color) in enumerate(configs):
        r=data['runs'][key];n=len(r['grad']);f=r['finite_grad_count'];bad=r['nonfinite_grad_count']
        b.barh(i,f,color=BLUE,height=.48);b.barh(i,bad,left=f,color=RED,height=.48)
        b.text(n+2,i,f'{f}/{n} 有限' if not bad else f'{bad}/{n} NaN',va='center',fontsize=11,color=RED if bad else INK)
    b.set_yticks(range(4),[x[1].replace(' · ','\n',1) for x in configs],fontsize=11)
    b.invert_yaxis();b.set_xlim(0,150);b.set_xticks([0,50,100]);b.set_xlabel('已记录的优化尝试次数')
    fig.text(.09,.29,'有限数值仍跨越多个数量级',fontsize=15,color=AMBER)
    fig.text(.60,.29,'K=8：50 轮采样 × 2 次优化 → 100 次 NaN',fontsize=14,color=RED)
    return fig

def fig3(data):
    cap=('读图：左侧每根柱的分母均为 100 个任务组（50 轮采样 × 每轮 2 组），柱高表示组内所有候选奖励都为零的比例。'
      '右侧每个小方块代表一轮采样；每行 10 轮，从左到右、从上到下共 50 轮。浅色、橙色、深红分别代表该轮 2 组中有 0、1、2 个全零奖励组。'
      '方块表示观测记录，不表示独立随机种子或分子成功率。\n'
      '共同条件：9B SFT 起点、真实 acquisition 奖励、每条回复上限 8192 token、50 轮采样。K=2/U=1 与 K=8/U=2 同时改变了候选数 K 和更新次数 U；'
      '两条都只有一次运行，且 K=8 的 100 次梯度全部 NaN。故这里只报告 49% 与 9% 的观测差异，不做显著性或 K 的单因素因果结论。\n'
      '来源：9b-sft_20260909T071143Z 与 k8sft_20260909T200313Z 的 train.log。分母使用每轮都记录的 count_no_gradient；'
      '正计数组逐轮与 count_lt_eps、count_0.0 核对一致。无奖励差异组是奖励诊断，不能直接当作优化失败次数。')
    fig=canvas('2.3','奖励诊断：全零奖励组从一条运行的 49% 到另一条的 9%',
      '后一个配置更少出现“组内候选都拿零分”；这不是分子搜索性能提升的证据。',cap)
    a=fig.add_axes([.085,.42,.32,.39]);style(a)
    vals=[data['runs'][k]['zero_groups'] for k in ['sft','k8']]
    a.bar([0,1],vals,color=[AMBER,BLUE],width=.53)
    for i,v in enumerate(vals):a.text(i,v+3,f'{v}%\n{v} / 100 组',ha='center',fontsize=19,weight='bold')
    a.set_ylim(0,75);a.set_xticks([0,1],['K=2，U=1','K=8，U=2']);a.set_ylabel('全零奖励组占比（%）')
    a.set_yticks([0,25,50,75]);a.set_title('两条运行的完整 50 轮记录',fontsize=14)
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#E4EBF2','#D9A156','#A63840']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    for key,y,label in [('sft',.625,'K=2，U=1：49 个全零组'),('k8',.385,'K=8，U=2：9 个全零组')]:
        b=fig.add_axes([.55,y,.385,.17]);arr=np.array([v['no_gradient_groups'] for v in data['runs'][key]['rollout']]).reshape(5,10)
        b.imshow(arr,cmap=cmap,norm=norm,aspect='auto')
        b.set_xticks(np.arange(-.5,10,1),minor=True);b.set_yticks(np.arange(-.5,5,1),minor=True)
        b.grid(which='minor',color=BG,linewidth=3);b.tick_params(which='both',length=0)
        b.set_xticks([]);b.set_yticks([0,4],['第 1–10 轮','第 41–50 轮'],fontsize=10)
        b.set_title(label,loc='left',fontsize=14)
        for sp in b.spines.values():sp.set_visible(False)
    leg=fig.add_axes([.55,.31,.4,.035]);leg.set_axis_off()
    for i,(color,label) in enumerate(zip(cmap.colors,['0 组','1 组','2 组'])):
        leg.add_patch(plt.Rectangle((i*.31,.15),.045,.65,color=color));leg.text(i*.31+.065,.45,label,va='center',fontsize=11)
    fig.text(.085,.31,'候选轨迹预算：200 条 vs 800 条\n两者预算不同，不比较采样效率。',fontsize=12,color=MUTED)
    return fig

def fig4(data):
    cap=('读图：箭头表示形成效果结论所需的证据顺序，不表示训练耗时或已经完成的比例。蓝色框表示已观察到的工程产物，橙色框表示尚未完成的训练验证，'
      '灰色虚线框表示在已核查材料中没有找到完整、可比的效果结果；“未找到”不是数值 0。图中的两条评测分支都需要 SFT-only 和 SFT+RL 对照。\n'
      '评测定义：SFT-only 为仅监督微调的模型，SFT+RL 为其上继续强化学习的模型；训练目标为 KRAS G12D。'
      'G12D 是同目标评测，G12C 是另一 KRAS 突变体上的迁移评测；两者使用各自活性模型，共用 8UN5 对接受体。'
      '按原计划，每种策略、每个目标用 5 个随机种子，搜索预算均为 80；HV（Pareto 超体积）衡量对接与活性两项目标的综合搜索结果，越高越好，比较时参照点须一致。\n'
      '来源与范围：TRAINING_PLAN.md 的评测协议、2026-09-05 MAIN_OBJECTIVE_AUDIT.md，结合截至 09-10 的选定原始日志及 PR 2/5/6 后续记录、'
      '09-19 对 ldm_rl/results 文件清单的核查。本图不声称搜索了所有服务器，也不沿用旧审计“9B 从未出现有限梯度”的过时结论。')
    fig=canvas('2.4','已积累训练与诊断证据，RL 对 SFT 的效果结论仍待验证',
      '下一项关键产出：可验证的模型更新，再进入同预算、多个随机种子的分子搜索评测。',cap)
    ax=fig.add_axes([.045,.31,.91,.54]);ax.set_axis_off()
    card(ax,.015,.57,.275,.36,'已观察：运行与存档','1.5B 端到端试跑\n9B 采样记录与 checkpoint',BLUE)
    card(ax,.365,.57,.275,.36,'待验证：可靠训练','梯度数值、实际权重变化\n以及存档恢复后的行为',AMBER)
    ax.add_patch(FancyArrowPatch((.30,.75),(.35,.75),arrowstyle='->',mutation_scale=20,color=MUTED))
    ax.add_patch(FancyArrowPatch((.65,.75),(.77,.75),arrowstyle='->',mutation_scale=20,color=MUTED))
    ax.text(.80,.82,'需要两条评测分支',ha='center',fontsize=14,color=MUTED)
    ax.plot([.80,.80],[.70,.47],color=MUTED)
    ax.plot([.28,.80],[.47,.47],color=MUTED)
    for x in [.28,.80]:ax.add_patch(FancyArrowPatch((x,.47),(x,.405),arrowstyle='->',mutation_scale=18,color=MUTED))
    for x,title,desc in [(.015,'G12D：同目标效果','SFT-only vs SFT+RL'),(.535,'G12C：迁移效果','SFT-only vs SFT+RL')]:
        ax.add_patch(FancyBboxPatch((x,.015),.43,.36,boxstyle='round,pad=0.01,rounding_size=.02',facecolor='white',edgecolor=MUTED,linestyle='--'))
        ax.text(x+.025,.325,title,fontsize=17,weight='bold',va='top')
        ax.text(x+.025,.245,desc+'\n每种策略 5 seeds · 搜索预算 80',fontsize=13,va='top',linespacing=1.65)
        ax.text(x+.025,.085,'可比 HV 结果：未找到',fontsize=18,color=MUTED,weight='bold',va='top')
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    return fig

FIGURES=[fig1,fig2,fig3,fig4]
HEADINGS=['2.1 小模型：端到端链路验证','2.2 9B：训练执行与数值表现','2.3 奖励与采样：两条配置的观测对比','2.4 阶段性结果：距离效果结论还有多远']

def build(data):
    setup_font();(HERE/'figures').mkdir(exist_ok=True)
    figures=[]
    with PdfPages(HERE/'experiment_progress.pdf') as pdf:
        for i,fn in enumerate(FIGURES,1):
            f=fn(data);f.savefig(HERE/f'figures/figure_{i}.png',dpi=170)
            f.savefig(HERE/f'figures/figure_{i}.svg');pdf.savefig(f);plt.close(f)
            figures.append(HERE/f'figures/figure_{i}.png')
    blocks=[]
    for title,p in zip(HEADINGS,figures):
        encoded=base64.b64encode(p.read_bytes()).decode()
        blocks.append(f'<section><h2>{html.escape(title)}</h2><a href="data:image/png;base64,{encoded}" target="_blank"><img alt="{html.escape(title)}；完整说明见图内图注" src="data:image/png;base64,{encoded}"></a></section>')
    (HERE/'experiment_progress.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>LDM RL · 训练实验与阶段性结果</title><style>body{margin:0;background:#edf1f6;color:#172b43;font-family:system-ui,sans-serif}main{max-width:1500px;margin:30px auto}h1,h2,p{padding:0 24px}h2{font-size:18px;margin:35px 0 12px}img{width:100%;height:auto;display:block}section{background:#f7f9fc;padding-top:1px;margin-bottom:30px}p{color:#506277}</style><main><h1>训练实验与阶段性结果</h1><p>四张图独立包含结论、定义、样本量、限制与来源。实验记录截至 2026-09-10；2026-09-19 核验。</p>'+''.join(blocks)+'</main></html>')
    import nbformat as nbf
    nb=nbf.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbf.v4.new_markdown_cell('# 训练实验与阶段性结果\n\n四张独立图，正文仅保留小节标题。数据为已冻结的原始日志摘录；不启动训练。'),
      nbf.v4.new_code_cell('from pathlib import Path\nimport json\nfrom IPython.display import display\nimport matplotlib.pyplot as plt\nfrom build_report import setup_font, FIGURES\nsetup_font()\ndata = json.loads(Path("data/evidence.json").read_text())')]
    for i,title in enumerate(HEADINGS):
        nb.cells.extend([nbf.v4.new_markdown_cell('## '+title),nbf.v4.new_code_cell(f'fig = FIGURES[{i}](data)\ndisplay(fig)\nplt.close(fig)')])
    # Explicit PNG rendering keeps outputs visible even when the kernel uses Agg.
    for cell in nb.cells:
        if cell.cell_type=='code' and 'display(fig)' in cell.source:
            cell.source=cell.source.replace('display(fig)','from io import BytesIO\nfrom IPython.display import Image\nbuf = BytesIO()\nfig.savefig(buf, format="png", dpi=150)\ndisplay(Image(data=buf.getvalue()))')
    nbf.write(nb,HERE/'experiment_progress.ipynb')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract',type=Path);args=ap.parse_args()
    if args.extract:extract(args.extract)
    else:build(json.loads((HERE/'data/evidence.json').read_text()))
