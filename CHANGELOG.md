# Changelog

Changes in this fork relative to upstream [Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) v1.6.0 (`9257651`), newest first.

Each entry says whether output changes. "Bit-identical" means a fixed-seed A/B produced `torch.equal` latents before and after the change.

## 2026-10-05: window-safe FreeNoise for perfect loops

### Add `FreeNoise (window-safe)` noise type
- New option in every `noise_type` dropdown (Sample Settings and the noise layer nodes). The existing `FreeNoise` is unchanged.
- **The problem:** FreeNoise builds noise by copying blocks forward from frame 0, which assumes context windows never move. Uniform context schedules shift the window phase every sampling step, and looped schedules wrap around the end, so windows end up holding the same noise frame twice. Duplicate noise pulls those frames toward the same image.
  - On loops, the windows across the seam do this on every step, so motion stalls at the seam: in a 48-frame looped test, the last-to-first change was 0.32-0.44x the median frame-to-frame change, smaller than every other frame step.
  - With standard uniform 16/8 at 669 frames, 734 of 913 windows (80%) held duplicate noise across 11 steps.
- **The fix:** each frame draws its noise from a pool slightly larger than one window, chosen so no noise repeats within any window at any offset. When the context schedule is looped, this also holds across the loop seam. The first window keeps its own noise, as in FreeNoise.
- **Measured:**
  - 0 windows with duplicate noise across standard and looped schedules, 37 to 1046 frames, and overlaps 4 to 15.
  - The same 48-frame loop's seam lands at 1.01-1.02x the median frame step, in the middle of the normal range.
  - On a 48-frame 1456x768 vid2vid workflow, a flow-based flicker metric (warp error against the plate's optical flow) was unchanged (2.31 vs 2.31). Speed was unchanged.
- Output: differs from FreeNoise by design. Existing `FreeNoise` and `default` noise renders were checked bit-identical before and after this change.

## 2026-10-05: views path cleanup

### Reduce memory and syncs in motion module views
- Only affects workflows that plug view options into a context options node (sliding views inside the motion modules).
- The views path kept a full-size count tensor (frames x tokens x channels) that only ever held one weight per frame. It is now `(1, frames, 1, 1)` and broadcasts in the final division, which saves about 700 MB per top-level motion module call at 1456x768.
- Contiguous views index with slices instead of Python lists, and all view weights for a call are copied to the GPU once instead of once per view.
- Measured with the same workflow at 96 frames, context 32/8 with views 16/8: pass 1 went from 25.90 to 24.75 s/step (-4.4%), pass 2 from 29.08 to 27.86 s/step (-4.2%), and peak VRAM from 19.2 to 18.8 GB.
- Output: bit-identical.

## 2026-10-04: context window loop cleanup

The three changes below were measured together on an RTX 4090 (ComfyUI 0.38.2, PyTorch 2.7.1+cu128). The test was a fixed-seed 48-frame cut of a two-pass SD1.5 vid2vid workflow at 1456x768 (v3_sd15_mm, uniform context 16/1/8 pyramid, FreeNoise, two SamplerCustom passes with ControlNet and IPAdapter), averaged over three warm runs each:

| | Before | After |
| --- | --- | --- |
| Pass 1 (cfg 3), s/step | 12.95 | 12.26 (-5.3%) |
| Pass 2 (cfg 5, ControlNet + IPAdapter), s/step | 15.12 | 14.28 (-5.6%) |
| Whole workflow | 368 s | 349 s (-5.2%) |

All latents were bit-identical to the unmodified code.

The changes were also checked on a second workflow: legacy `ADE_AnimateDiffLoaderWithContext` with motion scale 1.1, looped uniform context 16/1/4 (some windows wrap and keep list indexing), a motion LoRA, IPAdapterTiled, and two Efficient KSampler passes at 768x432, 48 frames. Latents were bit-identical, and speed was unchanged within 0.5%: the gain depends on how much per-window overhead a workflow has.

### Remove CPU-GPU syncs from the context window loop
- Contiguous context windows index `x`, timesteps and conds with a slice (a view) instead of a Python list. List indexing copies an index tensor to the GPU and forces a sync. Looped windows that wrap around and strided windows still use list indexing.
- Fuse weights for all windows are built on the CPU and copied to the GPU once per step instead of once per window.
- Upstream issue: [#524](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/issues/524).
- Output: bit-identical.

### Remove outdated attention fallback workaround
- `CrossAttentionMM` caught `CUDA error: invalid configuration argument` and switched to a slower attention backend for the rest of the run. ComfyUI's `attention_pytorch` now splits batches over 2^15 on NVIDIA itself, so the workaround and its `reset_attention_type` / `reset_temp_vars` plumbing were dead code.
- Backend selection is unchanged: SDPA when ComfyUI's pytorch attention is enabled (all NVIDIA cards on torch 2+), otherwise split or sub-quadratic.
- Output: bit-identical.

### Skip no-op key scaling in temporal attention
- With no motion scale multival, `VersatileAttention` stored `scale = 1.0`, so every temporal attention call ran `k *= 1.0` over the whole key tensor. It now stores `None` and skips the multiply.
- Output: bit-identical.
