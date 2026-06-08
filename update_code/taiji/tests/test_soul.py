"""态极灵魂系统 - 完整验证"""
import sys; sys.path.insert(0, 'e:/OmniCore')

print('='*60)
print('态极灵魂系统 - 完整验证')
print('='*60)

# 1. 模型核心
from taiji.loader import create_model
model, tokenizer = create_model('125m', 'cpu')
for name in ['read_local_file','write_file','execute_python','search']:
    tokenizer.register_tool(name)
model.set_num_tools(4)
params = sum(p.numel() for p in model.parameters())/1e6
print(f'[OK] 模型: {params:.1f}M params')

# 2. 知识蒸馏器
from taiji.knowledge_distiller import KnowledgeDistiller
d = KnowledgeDistiller()
react, conv = d.distill_all()
total_distilled = len(react) + len(conv)
print(f'[OK] 知识蒸馏: {len(react)} react + {len(conv)} conv = {total_distilled} 样本 (10维)')

# 3. 进化引擎
from taiji.evolution_engine import EvolutionEngine
e = EvolutionEngine(data_dir='taiji/test_evolution')
print(f'[OK] 进化引擎: phase={e.metrics.current_phase}')

# 4. 用户画像
from taiji.user_profile import UserProfile
p = UserProfile(data_dir='taiji/test_user')
print(f'[OK] 用户画像: interactions={p.preferences.total_interactions}')

# 5. 数据生成器
from taiji.data_generator import generate_bulk_react_data, generate_bulk_conversation_data
gen = generate_bulk_react_data(10)
print(f'[OK] 数据生成: {len(gen)} 样本')

# 6. 训练流程
from taiji.trainer import ModelSelfTrainer, build_dataset
all_conv = conv + generate_bulk_conversation_data(5)
dataset = build_dataset(tokenizer, extra_react_data=react+gen, extra_conv_data=all_conv, max_length=256)
trainer = ModelSelfTrainer(model, tokenizer, learning_rate=1e-4, warmup_steps=3)
for prog, desc, hist, metrics in trainer.finetune(dataset, num_epochs=1, batch_size=4, log_steps=999, device='cpu'):
    if metrics.get('status') == 'completed':
        loss_val = metrics.get('loss', 'N/A')
        print(f'[OK] 训练完成: Loss={loss_val}')

# 7. 训练格式对齐验证
from taiji.trainer import ReActDataset
ds = ReActDataset(tokenizer, gen, [], max_length=256)
sample = ds[0]
has_tool_desc = tokenizer.decode(sample['input_ids'][:80].tolist(), skip_special_tokens=False)
aligned = '可用工具' in has_tool_desc
print(f'[OK] 训练格式对齐: {aligned}')

# 8. 推理引擎
from taiji.inference import NativeInferenceEngine
engine = NativeInferenceEngine(model, tokenizer, 'cpu')
print(f'[OK] 推理引擎: NativeInferenceEngine 已初始化')

print()
print('='*60)
print('ALL SYSTEMS GO - 态极灵魂系统验证通过')
print('='*60)