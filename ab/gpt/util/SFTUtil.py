import ast
import textwrap

available_backbones = ['convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_t', 'swin_v2_t']

available_patterns = [
    'Parallel_Triple', 
    'Backbone_A_to_Fractal', 
    'Backbone_B_to_Fractal', 
    'Dual_Backbone_Fuse_Then_Fractal',
    'Fractal_Then_Dual_Backbone',
    'Split_Stem_Parallel_Fuse'
]

legacy_patterns = tuple(available_patterns)

open_discovery_goal_profiles = (
    {
        "name": "StemProjectCascade",
        "tags": ("stem", "project", "multi_stage"),
        "brief": "Build a real learned stem before branching. Each major branch should pass through an explicit project or bridge module before a later-stage fusion. Avoid a single terminal concat of raw branch outputs.",
        "module_hints": ("self.stem", "self.project_a", "self.project_b", "self.bridge", "self.fuse"),
    },
    {
        "name": "DeepFractalProject",
        "tags": ("fractal_deep", "project", "multi_stage"),
        "brief": "Create one deeper path with at least two fractal-like stages or a fractal stage plus a projector before fusion. The deep path should be structurally different from a one-shot fractal branch.",
        "module_hints": ("self.project", "self.bridge", "self.fractal_stage1", "self.fractal_stage2", "self.fuse"),
    },
    {
        "name": "SplitStemWideFuse",
        "tags": ("stem", "wide_fuse", "multi_stage"),
        "brief": "Use a shared stem to feed asymmetric branches, then perform a staged wide fusion. One branch may stay lightweight while another becomes deeper, but they should not meet only once at the classifier input.",
        "module_hints": ("self.stem", "self.branch_a", "self.branch_b", "self.project", "self.fuse"),
    },
    {
        "name": "ProjectReuseMixer",
        "tags": ("project", "branch_reuse", "multi_stage"),
        "brief": "Let one branch condition or align another through a project, bridge, adapter, or mixer block before the final classifier. Use at least two graph merges or one explicit reuse stage plus a final fuse.",
        "module_hints": ("self.project", "self.bridge", "self.adapter", "self.mixer", "self.fuse"),
    },
    {
        "name": "StemProjectWide",
        "tags": ("stem", "project", "wide_fuse"),
        "brief": "Expose a visible stem and explicit projection modules, then use a wider fusion with three or more incoming branch features. The projected branches should remain visible in the computation graph.",
        "module_hints": ("self.stem", "self.project_a", "self.project_b", "self.project_c", "self.fuse"),
    },
)

skeleton_code = """import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# ==========================================
# 1. FIXED INFRASTRUCTURE (DO NOT MODIFY)
# ==========================================
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.model_name = str(model)
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4: return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3: return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# ==========================================
# 2. DYNAMIC COMPONENTS (TO BE IMPLEMENTED)
# ==========================================

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        

    def infer_dimensions_dynamically(self, num_classes):
        self.to(self.device)
        was_training = self.training
        self.eval()
        classifier = getattr(self, "classifier", None)
        with torch.no_grad():
            c, h, w = self._input_spec
            dummy = torch.zeros(1, c, h, w).to(self.device)
            self.classifier = nn.Identity()
            output_feat = self.forward(dummy, is_probing=True)
            output_feat = adaptive_pool_flatten(output_feat)
            if output_feat.dim() != 2:
                output_feat = output_feat.flatten(1)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes).to(self.device)
        if classifier is not None and isinstance(classifier, nn.Module):
            del classifier
        self.train(was_training)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        
    def train_setup(self, prm):
        self.to(self.device)
        self.freeze_backbones = bool(prm.get('freeze_backbones', getattr(self, 'freeze_backbones', True)))
        for backbone in (self.backbone_a, self.backbone_b):
            for param in backbone.parameters():
                param.requires_grad = not self.freeze_backbones
            if self.freeze_backbones:
                backbone.eval()
            else:
                backbone.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters remain after applying freeze_backbones")
        self.optimizer = torch.optim.SGD(trainable_params, lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        if self.freeze_backbones:
            self.backbone_a.eval()
            self.backbone_b.eval()
        else:
            self.backbone_a.train()
            self.backbone_b.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss): continue
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'): train_iter.shutdown()
            del train_iter
            gc.collect()
"""

prompt_template="""
### Role & Context
You are a Senior AI Architect. Your task is to implement a **specific** model instance based on a strict skeleton to achieve an accuracy of {accuracy}. 

### Task Overview
Complete the three missing components. **DO NOT** write generic code. You must implement the architecture using the target design pattern provided.

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Technical Specifications (MANDATORY REQUIREMENTS)

1. **Target Pattern: `{target_pattern}`**
   - YOU MUST explicitly set `self.pattern = '{target_pattern}'` inside `__init__`.
   - YOU MUST implement the logic for this specific pattern throughout the code.
   - **CRITICAL REQUIREMENT**: DO NOT just blindly copy the standard Parallel_Triple structure. You MUST be highly creative and design a truly unique structural flow in `forward`. Vary your module usage and connection topology!

2. **Component: `drop_conv3x3_block`**
   - Implement starting with `def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):`.
   - Return an `nn.Sequential` block (Conv2d -> BatchNorm2d -> Activation -> Dropout2d).

3. **Component: `Net.__init__`**
   - Implement starting with `def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:`.
   - **MANDATORY**: `self.pattern = '{target_pattern}'`
   - **Backbone Selection**: Choose EXACTLY TWO models from [{available_backbones}].
   - **Initialization**: 
     - Initialize `self.backbone_a` and `self.backbone_b` using `TorchVision(model='...', in_channels=...)`.
     - Initialize `self.features` (1-2 `FractalUnit` layers).
     - Set `self._input_spec = tuple(in_shape[1:])`.
     - Call `self.infer_dimensions_dynamically(out_shape[0])`.
   - **Example Implementation Fragment**:
     ```python
     self.pattern = '{target_pattern}'
     self.backbone_a = TorchVision(model='resnet18', in_channels=in_shape[1]).to(device)
     ...
     ```

4. **Component: `Net.forward`**
   - Implement starting with `def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:`.
   - **Flow Control**: Implement the data flow for `{target_pattern}`. Use `adaptive_pool_flatten` for module outputs before fusion.
   - **Fusion Patterns Logic Blueprint**:
     * `Parallel_Triple`: `Result = Concat(backbone_a(x), backbone_b(x), features(x))`
     * `Backbone_A_to_Fractal`: `Result = features(backbone_a(x))` (Sequential flow)
     * `Split_Stem_Parallel_Fuse`: `stem_out = STEM(x); Result = Concat(backbone_a(stem_out), backbone_b(stem_out))`
   - **CRITICAL - NO GHOSTING**: You MUST use ALL defined components in the `forward` pass.
   - **CRITICAL RESTRICTION**: You MUST build the computational graph directly without using ANY `if self.pattern == ...` control flow or dynamic loops (like `getattr`/`hasattr`) inside `forward`.
   - **PARAM REMINDER**: Always pass `in_channels=...` when creating `TorchVision` models.

### Output Requirement (STRICT)
Output ONLY the implementation within the XML tags. Each tag MUST contain the complete function/method definition (signature and body). No markdown, no conversation.

<block>
# Full drop_conv3x3_block implementation
</block>
<init>
# Full __init__ implementation
</init>
<forward>
# Full forward implementation
</forward>
"""

open_discovery_skeleton_code = skeleton_code

open_discovery_prompt_template = """
### Role & Context
You are a Senior AI Architect optimizing an image-classification model under a strict XML ABI. Your job is to produce a trainable dual-backbone architecture that improves frozen-proxy training accuracy.

### Performance Goal
Produce one trainable architecture that improves short-budget training accuracy when both backbones are frozen. Use `{accuracy}` only as a seed reference for the starting model quality, not as the reward baseline.

### Optimization Track
- Track Name: {goal_name}
- Optimization Target Tags: {target_tags}
- Design Brief: {design_brief}

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Hard Requirements
1. Keep the output format EXACTLY the same:
   - Output ONLY `<block>`, `<init>`, `<forward>`
   - Each tag must contain the full function/method definition
   - No markdown, no explanation, no extra text

2. Optimize for accuracy first
   - The reward is driven by frozen-proxy train accuracy improvement, not novelty
   - Treat `{accuracy}` only as seed context, not as the target reward baseline
   - Architecture changes must serve train accuracy, not visual novelty
   - Avoid unnecessary branches, dead modules, and decorative topology changes

3. Keep a descriptive `self.pattern`
   - Set `self.pattern` to a concise descriptive name
   - `self.pattern` is for logging only; it does not need to advertise novelty
   - DO NOT leave `self.pattern` missing

4. Use the dual-backbone search space
   - Use EXACTLY two backbones named `self.backbone_a` and `self.backbone_b`
   - Initialize both with `TorchVision(model=..., in_channels=...)`
   - Use both backbones in `forward`
   - Choose both backbone model names from this exact list: [{available_backbones}]
   - Do not invent aliases, renamed variants, or unsupported backbone names
   - Do not omit either backbone, and do not add a third backbone
   - Both backbones are frozen during RL proxy training and final training

5. Build an accuracy-oriented computation graph
   - Do not rely on backbone finetuning; improve accuracy through stem/project/bridge/fractal/fuse/classifier design
   - Avoid the plain one-shot `torch.cat([backbone_a(x), backbone_b(x)])` classifier-only fusion
   - Prefer stems, projectors, bridges, reuse paths, staged fusion, or asymmetric branches when they help optimization
   - Use direct computation graph construction; no `if self.pattern == ...` in `forward`
   - Satisfy the Optimization Target Tags above with actual code structure, not just naming

6. Use the provided code ABI
   - Implement `drop_conv3x3_block`
   - Implement `Net.__init__`
   - Implement `Net.forward`
   - In `__init__`, set `self.device = device`, `self.use_amp = torch.cuda.is_available()`, and `self._input_spec = tuple(in_shape[1:])`
   - After that, call `self.infer_dimensions_dynamically(out_shape[0])`
   - Do not use `_input_dim` or `_output_dim`
   - Do not call `infer_dimensions(...)`
   - Do not reference undefined names such as `dropout_prob`, `in_channels`, or `features`

7. Component guidance
   - Reuse only backbone names from [{available_backbones}]
   - You may define optional modules such as `self.stem`, `self.bridge`, `self.project`, `self.mixer`, `self.fuse`
   - You may use `TorchVision(...)`, `FractalUnit(...)`, `nn.Sequential(...)`, and standard `nn.*` layers inside `__init__`
   - Use all major modules you define in `forward`
   - To expose structure clearly, prefer semantic attribute names such as: {module_hints}
   - Do not hide all learned routing logic inside a single generic name like `self.features`
   - `is_probing` is optional. Your `forward` may use it, or may ignore it completely, as long as the graph still runs when `self.classifier` is temporarily `nn.Identity()`

8. Shape safety and trainability
   - Ensure `forward` returns classifier logits
   - Use `adaptive_pool_flatten(...)` before concatenating or classifying branch outputs when needed
   - Avoid placeholder code, dead modules, duplicate assignments, and hardcoded broken dimensions
   - The reward worker runs with a CPU time budget. Overly heavy backbones or very slow graphs can be marked as timed out

### Preferred Optimization Directions
- stem -> split -> asymmetric backbones -> bridge -> fuse
- backbone -> fractal -> project -> dual fusion
- split stem with one deep branch and one lightweight branch
- staged fusion where one branch conditions another before classification
- multi-hop feature routing that is still shape-safe
- explicit `stem`, `project`, `bridge`, `adapter`, `mixer`, or `fuse` modules that are visibly used in `forward`

### Output Requirement (STRICT)
Output ONLY the implementation within the XML tags. Each tag MUST contain the complete function/method definition (signature and body).

<block>
# Full drop_conv3x3_block implementation
</block>
<init>
# Full __init__ implementation
</init>
<forward>
# Full forward implementation
</forward>
"""

open_discovery_rl_prompt_template = """
### Role & Context
You are a Senior AI Architect optimizing a dual-backbone image-classification model under a strict XML ABI. Your job is to produce a trainable architecture that improves frozen-proxy training accuracy.

### Performance Goal
Produce one trainable architecture that improves short-budget training accuracy when both backbones are frozen. Use `{accuracy}` only as a seed reference for the starting model quality, not as the reward baseline.

### Optimization Track
- Track Name: {goal_name}
- Optimization Target Tags: {target_tags}
- Design Brief: {design_brief}

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Hard Requirements
1. Keep the output format EXACTLY the same:
   - Output ONLY `<block>`, `<init>`, `<forward>`
   - Each tag must contain the full function/method definition
   - No markdown, no explanation, no extra text
   - The first non-whitespace token must be `<block>`
   - The last non-whitespace token must be `</forward>`

2. Optimize for accuracy first
   - The reward is driven by frozen-proxy train accuracy improvement, not novelty for its own sake
   - Treat `{accuracy}` only as seed context, not as the target reward baseline
   - Architecture changes must serve train accuracy, not novelty for its own sake
   - Avoid decorative complexity, unused modules, or graph edits that do not improve learning signal
   - Trainable but non-improving samples receive only weak reward
   - Beating the previous strong group matters more than merely staying valid
   - Refreshing the current best group gets the strongest reward
   - Dominant-family repeats that do not beat the current target are penalized
   - Plain classifier-only parallel fuse is penalized unless it clearly beats the previous-group target

3. Keep a descriptive `self.pattern`
   - Set `self.pattern` to a concise descriptive name
   - `self.pattern` is a logging label, not a novelty target
   - DO NOT leave `self.pattern` missing

4. Dual-backbone rules are mandatory
   - Use EXACTLY two backbones named `self.backbone_a` and `self.backbone_b`
   - Initialize both with `TorchVision(model=..., in_channels=...)`
   - Choose both backbone model names from this exact list: [{available_backbones}]
   - The two backbone names may be the same or different, but both must come from that list
   - Do not invent aliases, renamed variants, or unsupported backbone names
   - Both backbones must appear in `__init__` and `forward`
   - Do not omit either backbone, and do not add a third backbone
   - Both backbones are frozen during RL proxy training and final training

5. Preserve the required ABI
   - Implement `drop_conv3x3_block`
   - Implement `Net.__init__`
   - Implement `Net.forward`
   - In `__init__`, set `self.device = device`, `self.use_amp = torch.cuda.is_available()`, and `self._input_spec = tuple(in_shape[1:])`
   - After that exact initialization sequence, call `self.infer_dimensions_dynamically(out_shape[0])`
   - Do not use `_input_dim` or `_output_dim`
   - Do not call `infer_dimensions(...)`
   - Do not reference undefined names such as `dropout_prob`, `in_channels`, or `features`

6. Build an accuracy-oriented graph
   - Avoid the plain one-shot parallel fuse topology because it is usually too weak
   - Do not simply pool the two backbones once and concatenate them only at the classifier input
   - Do not rely on backbone finetuning; focus improvements on stem/project/bridge/fractal/fuse/classifier structure
   - Use the Optimization Target Tags above with actual code structure, not just naming
   - Prefer visible modules such as: {module_hints}
   - Do not define new classes or helper methods beyond the 3 required definitions
   - If reusing a strong motif, mutate locally with a better stem, projector, bridge, or fuse stage instead of resubmitting the same shallow graph
   - Make a local improvement that can plausibly beat the previous-group target, not just another valid variant of the same graph
   - Do not submit a classifier-only parallel fuse unless it clearly beats the current target
   - Make the target tags visible in real code structure, not only in `self.pattern`

7. Forward-path restrictions
   - Keep `forward` as straight-line assignments plus a final return
   - Do not use `if self.pattern` in `forward`
   - Do not emit `import ...` lines or `class ...` definitions
   - Never define `DropConv3x3Block` or any wrapper class
   - NEVER apply `self.classifier` manually before the final return line

8. Shape safety and trainability
   - `forward` must return classifier logits
   - Use `adaptive_pool_flatten(...)` before concatenating or classifying branch outputs
   - Prefer `TorchVision`, plain CNN blocks, or `FractalBlock`
   - Avoid placeholder code, dead modules, duplicate assignments, and broken dimensions
   - `is_probing` is optional. Your `forward` may use it or ignore it, but it must still run when `self.classifier` is temporarily replaced by `nn.Identity()`
   - The reward worker has a CPU time budget. Very heavy or slow architectures can return a timeout penalty instead of finishing evaluation

### Output Requirement (STRICT)
Output ONLY the implementation within the XML tags. Each tag MUST contain the complete function/method definition.

<block>
{block_signature}
    ...
</block>
<init>
{init_signature}
    ...
</init>
<forward>
{forward_signature}
    ...
</forward>
"""

def parse_nn_code(code_str):
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        block_code = None
        init_code = None
        forward_code = None

        def get_source(node):
            return ast.get_source_segment(code_str, node)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'drop_conv3x3_block':
                block_code = get_source(node)

            elif isinstance(node, ast.ClassDef) and node.name == 'Net':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        if sub_node.name == '__init__':
                            init_code = get_source(sub_node)
                        elif sub_node.name == 'forward':
                            forward_code = get_source(sub_node)

        def clean_code(c):
            return textwrap.dedent(c).strip() if c else None

        return clean_code(block_code), clean_code(init_code), clean_code(forward_code)

    except Exception as e:
        print(f"AST Parsing Failed: {e}")
        print(f"Code snippet: {code_str[:100]}...")
        return None, None, None
