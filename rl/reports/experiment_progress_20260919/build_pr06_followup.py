"""Extract allowlisted numeric diagnostics locally. Never exports raw logs or paths."""
from pathlib import Path
import hashlib,json,re,math
import numpy as np
import matplotlib.pyplot as plt
from build_report import setup_font,style,BLUE,RED,TEAL,AMBER,MUTED
from build_section03 import canvas
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures_pr06_followup'
KEYS=['sft','sft2','base','k8']
FIELDS=['loss','pg_loss','kl_loss','ppo_kl','train_rollout_logprob_abs_diff','grad_norm']
LABELS=['A · SFT · K=2 · U=1','B · SFT · K=2 · U=2','C · base · K=2 · U=1','D · SFT · K=8 · U=2']
DEFS='A/B 为 PR #6 的两条 SFT 运行，C 为 base 运行，D 为后续 K=8 的 SFT 运行；SFT 是监督微调起点，base 是未微调起点。K 是每任务候选数，U 是每轮优化尝试数。'

def extract():
 old=json.loads((HERE/'data/evidence.json').read_text());result={'privacy':'Only allowlisted metrics, anonymous run IDs, logical filenames, sizes, and hashes. No raw log lines, absolute paths, hosts, or account identifiers.','runs':{}}
 for label,key in zip('ABCD',KEYS):
  prior=old['runs'][key];directory=Path(old['source_root'])/'runs'/prior['name'];raw=(directory/'train.log').read_bytes()
  assert hashlib.sha256(raw).hexdigest()==prior['source_sha256']
  text=raw.decode(errors='replace');opt=[];roll=[];counters=[]
  for line in text.splitlines():
   m=re.search(r"step (\d+):.*'train/loss'",line)
   if m:
    rec={'step':int(m[1])}
    for field in FIELDS:
     match=re.search(r"'train/"+field+r"':\s*([^,}]+)",line)
     value=float(match[1]) if match else None
     rec[field]=value if value is None or math.isfinite(value) else None
    opt.append(rec)
   m=re.search(r"rollout (\d+):.*'rollout/raw_reward':\s*([^,}]+)",line)
   if m:roll.append({'rollout':int(m[1]),'raw_reward':float(m[2])})
   if "'rollout/zero_std/count_no_gradient'" in line:
    rec={}
    for k in ['count_no_gradient','count_lt_eps','count_0.0']:
     m=re.search(r"'rollout/zero_std/"+re.escape(k)+r"':\s*([^,}]+)",line)
     rec[k]=float(m[1]) if m else None
    counters.append(rec)
  assert len(roll)==len(prior['rollout'])==len(counters)
  assert len(opt)==len(prior['grad'])
  for a,b in zip(roll,prior['rollout']):assert a['rollout']==b['index'] and a['raw_reward']==b['raw_reward']
  ep=(directory/'episodes.jsonl').read_bytes();specs=[json.loads(json.loads(line)['prompt']) for line in ep.decode().splitlines() if line.strip()];spec=specs[0]
  assert all(item['reward']=='acquisition' for item in specs)
  assert spec['reward']=='acquisition'
  assert re.search(r'kl_loss_coef\s*\.{2,}\s*0.001',text)
  assert re.search(r'entropy_coef\s*\.{2,}\s*0.0',text)
  result['runs'][label]={'label':LABELS[ord(label)-65],'K':int(prior['args']['n_samples_per_prompt']),'global_batch_size':int(prior['args']['global_batch_size']),
   'reward_policy':spec['reward'],'mode':spec['mode'],'kl_loss_coef':.001,'entropy_coef':0.,'std_mode':prior['args']['grpo_advantage_std_mode'],
   'optimization':opt,'rollout':roll,'counters':counters,
   'sources':[{'logical_file':label+'/'+name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()} for name,content in [('train.log',raw),('episodes.jsonl',ep)]],
   'summary':{'rollouts':len(roll),'optimization_records':len(opt),'raw_reward_zero':sum(r['raw_reward']==0 for r in roll),'reported_no_gradient_groups':sum(r['count_no_gradient'] for r in counters),'groups':2*len(roll),'finite_grad_norm':sum(r['grad_norm'] is not None for r in opt),'pg_abs_max':max(abs(r['pg_loss']) for r in opt),'loss_decomposition_max_residual':max(abs(r['loss']-r['pg_loss']-.001*r['kl_loss']) for r in opt),'step0_ppo_kl':opt[0]['ppo_kl'],'median_logprob_gap':float(np.median([r['train_rollout_logprob_abs_diff'] for r in opt]))}}
 assert sum(result['runs'][a]['summary']['groups'] for a in 'ABC')==214
 assert sum(result['runs'][a]['summary']['reported_no_gradient_groups'] for a in 'ABC')==104
 (HERE/'data/pr06_followup_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result

def base(label,title,subtitle,caption):
 f=canvas(label,title,subtitle,caption);f.texts[-1].set_text('LDM RL · 匿名运行指标 · 原始日志未附带');return f

def curves(d):
 f=base('7.1','同一批运行：奖励并非一直为零，loss 也不能单独说明学到了什么',
 '直接核验 PR #6 的 A/B/C 三条运行，并补上后续 D；左列为 loss，右列为原始 reward。',
 '读图：每行对应同一运行，左列横轴为优化 step，右列为采样 rollout，均从 0 编号，二者不可逐点强行配对。蓝线为 loss（训练目标标量）；绿线为每轮原始奖励均值，橙点表示精确零均值。两列使用对称对数坐标，阈值分别为 1e-4、1e-5；各面板范围不同。'+DEFS+'\nA/B/C 的零奖励均值分别为 14/50、13/29、3/28，合计 30/107 轮；D 为 1/50。这个分母是采样轮，不是 214 个任务组。C 有 28 轮奖励但仅 27 次优化记录，不补造缺失步。四条运行实际 episode 奖励策略均为 acquisition；不能用 improvement 默认分支的运行最优值机制解释。数据：data/pr06_followup_metrics.json。')
 for i,(key,r) in enumerate(d['runs'].items()):
  y=.71-i*.105
  for j in range(2):
   a=f.add_axes([.12+j*.46,y,.36,.079]);style(a)
   rows=r['optimization'] if j==0 else r['rollout'];xx=[x['step' if j==0 else 'rollout'] for x in rows];yy=[x['loss' if j==0 else 'raw_reward'] for x in rows]
   a.plot(xx,yy,color=BLUE if j==0 else TEAL,lw=1);a.set_yscale('symlog',linthresh=1e-4 if j==0 else 1e-5);a.tick_params(labelsize=8)
   if j==1:a.scatter([x for x,v in zip(xx,yy) if v==0],[0 for v in yy if v==0],color=AMBER,s=12)
   a.set_ylabel(key+(' · loss' if j==0 else ' · reward'),fontsize=10)
   a.set_xticks(np.unique(np.linspace(0,max(xx),4).astype(int)))
   ticks=sorted(set([min(yy),(min(yy)+max(yy))/2,max(yy)] if j==0 else [0,min(v for v in yy if v>0),max(yy)]))
   a.set_yticks(ticks,[f'{v:.1e}' if v else '0' for v in ticks],fontsize=8)
   if i==3:a.set_xlabel('优化 step' if j==0 else '采样 rollout',fontsize=10)
 return f

def decomposition(d):
 f=base('7.2','loss 的组成已经能查清，梯度是否可用要另外判断',
 '四条运行均配置 KL 系数 0.001、熵系数 0；记录满足 loss ≈ pg_loss + 0.001 × kl_loss。',
 '读图：横轴为优化 step；蓝线为总 loss，橙线为策略损失 pg_loss，绿虚线为乘系数后的 KL 参考策略约束项；KL 为相对参考策略的偏离惩罚，不能与 ppo_kl 混为一项。纵轴使用对称对数（线性阈值 1e-4），面板范围不同。'+DEFS+'\n分解最大残差小于 6e-8。A/C 的策略损失绝对值最大分别约 1.94e-7、8.94e-8；B/D 达 0.133、0.165。接近零的策略损失标量不证明其梯度为零：正负项可以抵消而导数仍非零。D 的 100 个 loss 有限，但梯度范数 100/100 为 NaN（非数值），所以非零策略损失也不能证明有效更新。数据与匿名源文件哈希：data/pr06_followup_metrics.json。')
 for i,(key,r) in enumerate(d['runs'].items()):
  a=f.add_axes([.10+(i%2)*.47,.60-(i//2)*.24,.37,.17]);style(a);rows=r['optimization'];xx=[v['step'] for v in rows]
  for name,color,ls,scale,legend in [('loss',BLUE,'-',1,'总 loss'),('pg_loss',AMBER,'-',1,'策略项'),('kl_loss',TEAL,'--',.001,'0.001 × KL 项')]:a.plot(xx,[scale*v[name] for v in rows],color=color,ls=ls,lw=1.2,label=legend)
  a.set_yscale('symlog',linthresh=1e-4);a.set_title(r['label']+'；梯度有限 '+str(r['summary']['finite_grad_norm'])+'/'+str(len(rows)),fontsize=10);a.set_xlabel('优化 step',fontsize=9);a.tick_params(labelsize=8)
  a.set_xticks(np.unique(np.linspace(0,max(xx),4).astype(int)))
  lo,hi=a.get_ylim();ticks=[v for v in [0,1e-4,1e-3,1e-2,1e-1] if lo<=v<=hi];a.set_yticks(ticks,[f'{v:.0e}' if v else '0' for v in ticks],fontsize=8)
  if i==0:a.legend(fontsize=8,ncol=3,loc='upper right')
 return f

def counters(d):
 f=base('8.1','104 / 214 是计数器结果，“全部精确零奖励”仍需逐候选核实',
 'A/B/C 合计 104/214 组被 count_no_gradient 标记；后续 D 为 9/100，不能据此把梯度 NaN 归因于奖励相同。',
 '读图：柱高为每条运行被 count_no_gradient 标记的任务组比例，柱顶为计数/总组数。A/B/C 属于 PR #6 最初三条运行，D 是后续 K=8；K 为同一任务的候选数。此计数是奖励统计诊断，不是直接测出的参数梯度。SFT 为监督微调起点，base 为未微调起点，U 为每轮优化尝试数；蓝柱为 A/B/C，橙柱为 D。\n当前计数实现将相等奖励四舍五入到一位小数后命名分桶；因此 count_0.0 不能区分示例 [0,0] 和 [0.004,0.004]。历史代码版本与逐候选奖励仍须绑定。三计数相等也不能证明所有原始奖励精确为零，或证明 epsilon 阈值完全没有影响。A 与 D 还改变了每轮更新次数及批量；后续轮次的策略受此前更新影响，故不是严格单因素 K 对照。数据：data/pr06_followup_metrics.json；计数口径核查承接第四章。')
 a=f.add_axes([.12,.37,.78,.42]);style(a)
 for i,r in enumerate(d['runs'].values()):
  s=r['summary'];v=s['reported_no_gradient_groups'];n=s['groups'];a.bar(i,100*v/n,color=BLUE if i<3 else AMBER,width=.55);a.text(i,100*v/n+2,f'{int(v)}/{n}\n{100*v/n:.1f}%',ha='center',fontsize=12)
 a.set_xticks(range(4),LABELS,fontsize=10);a.set_ylabel('被计数器标记的任务组（%）');a.set_ylim(0,82)
 return f

def gaps(d):
 f=base('8.2','PR #6 未报告的 KL 和后端差异，现在可以补出读数',
 '这四条运行的日志已有数值；缺的是同条件归因验证，而不是完全没有指标。',
 '读图：左图为首次优化记录 step=0 的 ppo_kl（策略更新中记录的新旧策略差异统计）；右图为训练与采样后端 token 对数概率绝对差的逐步均值，再取运行内中位数。右轴为对数刻度。'+DEFS+'\nA/B/C/D 左侧值为 0、0.1635、0、0.1311；右侧为 0.2497、0.00786、0.4581、0.00782。两种统计参照不同，ppo_kl=0 不能推出两后端完全一致。B/D 的差异较小不证明修好了 NaN：D 梯度仍全部异常。下一步固定同一权重、token、mask 与精度比较逐 token 输出，再做只改变 K 的对照；此验证尚未执行。数据：data/pr06_followup_metrics.json。')
 for j,field in enumerate(['step0_ppo_kl','median_logprob_gap']):
  a=f.add_axes([.11+j*.47,.38,.36,.40]);style(a);vv=[r['summary'][field] for r in d['runs'].values()]
  a.bar(range(4),vv,color=[BLUE,TEAL,BLUE,AMBER]);a.set_xticks(range(4),list('ABCD'));a.set_ylabel('首次记录 ppo_kl' if j==0 else '后端 logprob 差异的运行内中位数')
  if j:a.set_yscale('log');a.set_ylim(.003,1)
  else:a.set_ylim(0,.22)
  for i,v in enumerate(vv):a.text(i,v*1.15 if j else v+.008,f'{v:.4g}',ha='center',fontsize=11)
 return f

S7=[('7.1 先看同一批运行的 loss 和原始 reward','我重新核验了 PR #6 的三条原始运行，并加上后续 K=8 运行。前三条运行中，原始奖励均值为零的是 30/107 个采样轮，而不是奖励一直为零；104/214 则是另一项按任务组统计的计数。**这两个分母不能混用。** 四条运行的实际 episode 奖励策略都是 acquisition，所以 PR #6 按 improvement 默认分支解释“必须超过历史最好值”的说法，不能直接用于这些曲线。图中把优化 step 与采样轮分开保留，也没有为少一条优化记录的 base 运行补造数据。'),
('7.2 loss 接近零，不等于策略梯度为零','四条运行的 loss 都能由策略项与 0.001 倍 KL 约束项解释，最大残差低于 6e-8。A/C 的策略损失标量接近零，B/D 则出现了明显非零的策略项。**不能由策略损失的标量为零推断梯度为零，也不能由它非零推断更新正常。** 最直接的反例是 D：100 个 loss 都有限，但同一步的梯度范数全部为 NaN。接下来应在固定输入下分别检查奖励形成的相对优势、策略项的梯度和优化器实际更新，而不能把所有问题都归到 K=2。')]
S8=[('8.1 保留 104/214 的观测，收回“全部精确为零”的推断','三条运行的 count_no_gradient 累计为 49/100、37/58、18/56，合计 104/214；后续 K=8 运行是 9/100。**这些数字可以保留，但“104 个组的每个奖励都精确为零”尚缺逐候选证据。** 当前实现会将奖励舍入后命名为 count_0.0，因此桶名和计数相等并不足以得出该结论。K=2 的标准化相对优势主要保留胜负方向，这仍然可以提供学习信号；它不等于整个运行必然学不到东西。'),
('8.2 补出已有诊断，再把需要验证的因果问题列清楚','PR #6 没有报告的首次 ppo_kl 和训练—采样后端 logprob 差异，可以从同一批日志补出；它们的数值和分母见图。**目前可以说不同运行的这些指标明显不同，但不能说某一项改动已经解决问题。** A 与 D 除了 K=2→8，还改变了每轮更新次数和 global batch；即使计数发生在当前轮更新前，后续采样也受以前的更新影响。我会先固定权重、输入 token、mask 和精度核对两后端，再只改变 K、固定更新次数与预算口径做对照；同时保存逐候选 reward、组标准差、失败原因和优化器是否跳步，才能把奖励问题与数值问题分别定位。')]

if __name__=='__main__':
 setup_font();OUT.mkdir(exist_ok=True);d=extract()
 for num,rows,fns in [(7,S7,[curves,decomposition]),(8,S8,[counters,gaps])]:
  title='回到 PR #6：同一批运行的 loss 与 reward' if num==7 else 'PR #6 的结论怎样修正，剩余问题怎样验证'
  parts=[f'# {num}. {title}']
  for i,((heading,para),fn) in enumerate(zip(rows,fns),1):
   stem=f'figure_{num}_{i}';f=fn(d);f.savefig(OUT/(stem+'.png'),dpi=170);f.savefig(OUT/(stem+'.svg'));plt.close(f)
   parts+=['## '+heading,para,f'![图 {num}.{i}：观测、指标定义和解释边界](figures_pr06_followup/{stem}.png)']
  (HERE/f'SECTION_{num:02d}.md').write_text('\n\n'.join(parts)+'\n')
