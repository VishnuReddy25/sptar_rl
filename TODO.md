# GRPO CUDA/CPU Tensor Mismatch Fix - TODO ✅
Current working directory: //wsl.localhost/Ubuntu/home/aiml_user/vishnu/sptar_v1

## Steps:
- [x] 1. Create this TODO.md ✅
- [x] 2. Add `self.model.to(self.device)` after `get_peft_model()` in PeftSoftPromptModel.__init__()` ✅
- [x] 3. In `generate_query()`, change tensor `.to(next(self.model.parameters()).device)` to `.to(self.device)` ✅
- [x] 4. Add explicit `attention_mask.to(self.device)` ✅
- [ ] 5. Update TODO.md after edits ✅
- [ ] 6. Test: Rerun `python grpo_phase2.py [args]` - confirm generate() works without RuntimeError
- [ ] 7. Complete task

