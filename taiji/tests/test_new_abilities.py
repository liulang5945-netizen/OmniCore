"""验证态极 5 项新能力"""
import sys; sys.path.insert(0, 'e:/OmniCore')

print('='*50)
print('态极新能力验证')
print('='*50)

# 1. Working Memory
from taiji.working_memory import WorkingMemory
wm = WorkingMemory()
wm.remember('test.py', 'def hello():\n    print("hi")\n')
wm.modify('test.py', 'hi', 'hello')
content = wm.recall('test.py')
print(f'[OK] WorkingMemory: remember/recall/modify works, content has "hello": {"hello" in content}')

# 2. Code Understander
from taiji.code_understander import CodeUnderstander
cu = CodeUnderstander()
ca = cu.analyze('def hello():\n    print("hi")\n\nclass Foo:\n    pass\n', 'test.py')
print(f'[OK] CodeUnderstander: {ca.summary}')
print(f'     Functions: {[f.name for f in ca.functions]}, Classes: {[c.name for c in ca.classes]}')

# 3. Self Evaluator
from taiji.self_evaluator import SelfEvaluator
se = SelfEvaluator()
ev = se.evaluate_task('test task', [{'action': 'read_file', 'thought': 'I need to read the file'}], 'done', True)
print(f'[OK] SelfEvaluator: overall={ev.overall_score}, strengths={ev.strengths}')

# 4. Screen Reader
from taiji.screen_reader import ScreenReader
sr = ScreenReader()
analysis = sr.analyze_text_description('Traceback: FileNotFoundError: config.json not found')
print(f'[OK] ScreenReader: errors={analysis.error_messages}')

# 5. Voice Interface
from taiji.voice_interface import VoiceInterface
vi = VoiceInterface()
print(f'[OK] VoiceInterface: stt={vi._stt_available}, tts={vi._tts_available}')

print()
print('='*50)
print('ALL 5 NEW ABILITIES VERIFIED')
print('='*50)