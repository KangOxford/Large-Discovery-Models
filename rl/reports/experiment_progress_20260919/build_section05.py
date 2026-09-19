"""Local chapter five: evidence-led diagnostic handoff, no experiment launch."""
from pathlib import Path
import json, hashlib
import matplotlib.pyplot as plt
from build_report import setup_font, BLUE, RED, AMBER, MUTED
from build_section03 import canvas, diagram, box, arrow
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures_section05'
ITEMS=[
('5.1','先找异常首次出现的位置，再讨论原因',
 '已知：同一条 9B 运行的 100 次 loss 有限，100 次梯度范数为 NaN。',
 [('锁定一次更新','保留输入、权重与随机状态\n绑定实际代码和优化器状态'),('沿计算顺序检查','前向量 → loss → 反向量\n再查梯度聚合与范数计算'),('按首次异常分流','张量已异常：追到相应算子\n张量有限：查聚合／范数')],
 '先找到候选故障位置；第一处异常是定位线索，还不是根因证明。',
 '读图：上方事实来自 k8sft_20260909T200313Z，9B 表示约 90 亿参数；loss 是训练目标，NaN 是非数值。三个虚线框与箭头表示尚未执行的排查顺序，不是观测到的故障传播。前向量指模型计算的中间张量，反向量指梯度。\n检查应覆盖参与该步的所有计算分片；日志中的范数异常不等于已经证明每个参数梯度异常。若固定状态无法重现，先比较输入、代码与精度条件，不能宣称修复。证据：data/section03_loss_grad.json；方案状态见 data/section05_manifest.json。',
 '接下来我会先查清梯度异常从哪里开始。一条约 90 亿参数（9B）的运行中，100 次 loss 都是有限数值，配对的梯度范数却全部为 NaN。**目前能确定的是梯度范数检查失败，还不能确定是哪一层或哪项 loss 配置造成的。** 我会固定一次更新的输入、模型和优化器状态，沿前向、反向和梯度聚合检查：如果梯度张量已经异常，就追到最早出现异常的算子；如果张量有限而范数异常，就查聚合和范数计算。这一步尚未完成，现有日志只能帮助缩小排查范围。'),
('5.2','改动有效，必须在同一故障上做对照',
 '首个异常位置只能缩小范围；需要受控复现才能把怀疑变成因果证据。',
 [('原条件 A','同一输入与完整训练状态\n先确认异常可重复出现'),('只改一项 B','只改疑似 loss 项或精度等\n记录首个异常是否消失'),('回到原条件 A','恢复原条件并再次复现\n再换预先选定的输入复核')],
 'A 失败、B 通过、恢复 A 再失败，才支持这项改动与异常有因果关系。',
 '读图：A 是原训练条件，B 是只改变一个疑似因素的条件；虚线框为建议实验，箭头为执行顺序，并非已获得的结果。每个分支都从同一份未被更新的完整状态开始，避免前一次更新影响后一次比较。\n通过指原异常消失且相关中间量与梯度有限，不代表模型收敛；若同时改变模型、采样数和精度，就不能把改善归因于其中一项。多个固定输入的复核用于检查解释是否过于局限。此图为待执行方案，无新增实验数值；依据第三章的异常证据制定。',
 '找到可疑位置以后，我会用同一份输入和完整训练状态做对照：先复现原来的异常，只改一个疑似因素，再恢复原条件复查。**如果原条件反复失败，单项改动后通过，恢复原条件又失败，就有理由把问题与这一因素联系起来。** 比如怀疑某项 loss 配置，就先只改这一项；同时换模型、采样数和精度，即使结果变好，也说不清是哪项起了作用。这是下一步的验证安排，目前还没有这组对照结果。'),
('5.3','奖励没有区分度，要追到候选级数据',
 '已发现的解释问题：count_0.0 是舍入后的桶名，不能直接读作“奖励精确全零”。',
 [('还原一个任务组','逐候选保存原始奖励\n关联有效性与评估状态'),('按来源分别判读','失败值？合法但相同？\n还是日志舍入掩盖差异？'),('核对实际归一化','绑定当次代码与模式\n核算标准差和 advantage')],
 '定位输出应是具体任务、候选与处理环节；只看一个计数桶无法得到这些信息。',
 '读图：count_0.0 是当前源码按一位小数舍入命名的奖励桶；历史运行版本仍需绑定。虚线框表示尚未执行的候选级核查。advantage 是相对优势，标准差度量同一任务组内奖励差异；箭头表示从原始结果到训练信号的核查方向。\n评估失败和合法的零奖励必须分开记录；即使奖励相同，也只能说明奖励提供不了组内偏好，不能据此断言总梯度为零，因为其他目标项仍可能贡献梯度。当前证据与源码摘录见 data/section04_evidence.json；不将示例当作实测。',
 '奖励这边，我会把问题追到同一任务下的具体候选。当前源码中的 count_0.0 是舍入后的桶名，仅凭它不能判断奖励是否精确为零。**需要同时保留每个候选的原始奖励、有效性和评估状态，再按当次代码重算组内标准差与相对优势。** 这样才能分清是评估失败、合法候选确实得分相同，还是显示和分桶掩盖了差异。即使一组奖励完全相同，也只能说这项奖励没有提供组内偏好；其他目标项仍可能贡献梯度。'),
('5.4','交接的终点，是别人能复核的定位结论',
 '数值问题消失、参数确实更新、搜索效果改善，是三个需要分别验证的结论。',
 [('数值与更新','保留修复前后对照\n核验优化器执行及参数差分'),('恢复与持续训练','恢复后复跑同一步\n再检查更长训练的稳定性'),('实际搜索收益','固定评估条件与搜索预算\n比较 RL 与 SFT 的结果')],
 '交接包：复现输入和状态 + 首个异常位置 + 单因素对照 + 尚未解决的证据缺口。',
 '读图：虚线框均为验收方案，不代表已经通过；箭头表示证据逐层增加。参数差分是同一权重更新前后的变化，恢复是加载保存的训练状态再验证。RL 为强化学习，SFT 为监督微调起点；比较须约定相同任务、预算口径、随机种子和固定评分条件。\n参数变化不是收益证明，短程稳定也不是收敛证明。最终需报告预先约定的搜索指标及跨任务或种子的变化，而不是挑一条更好的曲线。当前所选材料尚未建立这条完整证据链；本图不表示全项目从未开展相关验证。',
 '要把训练问题交给别人继续查，我会留下能重现异常的输入和完整训练状态、最早出现异常的位置、单项修改的对照结果，以及还没排除的解释。**接手的人应该能从这些材料复核判断，而不用重新猜问题在哪里。** 修复以后，还要确认优化器确实更新了参数、保存恢复后能复核同一步，并检查更长训练是否稳定。最后再用相同评估条件和搜索预算比较强化学习模型（RL）与监督微调起点（SFT）；参数发生变化和搜索效果提高，需要分别拿出证据。')]

def make(item):
 label,title,subtitle,steps,verdict,caption,_=item
 f=canvas(label,title,subtitle,caption)
 f.texts[-1].set_text('LDM RL · 已有观测与待执行验证')
 a=diagram(f)
 for i,(head,body) in enumerate(steps):
  x=.012+i*.335
  box(a,x,.40,.302,.55,head,body,BLUE,True)
  if i<2:arrow(a,(x+.305,.67),(x+.33,.67))
 box(a,.012,.015,.97,.29,'怎样解释检查结果',verdict,AMBER,True)
 return f

if __name__=='__main__':
 setup_font();OUT.mkdir(exist_ok=True)
 sources=[]
 for name in ['evidence.json','section03_loss_grad.json','section04_evidence.json']:
  p=ROOT/'data'/name;sources.append({'file':'data/'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 d=json.loads((ROOT/'data/section03_loss_grad.json').read_text())
 assert len(d['rows'])==100 and all(not r['grad_finite'] for r in d['rows'])
 (ROOT/'data/section05_manifest.json').write_text(json.dumps({'status':'Proposed diagnostic handoff; no new experiment executed.','sources':sources,'sections':[{'id':x[0],'title':x[1],'status':'proposed'} for x in ITEMS]},ensure_ascii=False,indent=2)+'\n')
 parts=['# 5. 从异常到根因：下一步怎样定位，怎样交接？']
 for i,item in enumerate(ITEMS,1):
  f=make(item);f.savefig(OUT/f'figure_{i}.png',dpi=170);f.savefig(OUT/f'figure_{i}.svg');plt.close(f)
  parts += ['## '+item[0]+' '+item[1],item[-1],f'![图 {item[0]}：定位步骤、判读方法与证据边界](figures_section05/figure_{i}.png)']
 (ROOT/'SECTION_05.md').write_text('\n\n'.join(parts)+'\n')
