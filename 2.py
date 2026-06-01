"""
[과제 2] text 없이 image token만 decoder에 넣기
기존 : image_tokens + text_tokens
변경 : image_tokens only
"""

import torch
import torch.nn as nn


# ────────────────────────────────────────────
# 1. Mini Vision Encoder (원본 그대로)
# ────────────────────────────────────────────
class MiniVisionEncoder(nn.Module):
    def __init__(self, image_size=224, image_channels=3, vision_dim=32, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=image_channels,
            out_channels=vision_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, vision_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=vision_dim, nhead=4, batch_first=True, dropout=0.0, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values)               # [B, D, H/p, W/p]
        image_features = x.flatten(2).transpose(1, 2)   # [B, N_patches, D]
        image_features = image_features + self.pos_embed
        image_features = self.transformer(image_features)
        image_features = self.norm(image_features)
        return image_features


# ────────────────────────────────────────────
# 2. Mini Projector (원본 그대로)
# ────────────────────────────────────────────
class MiniProjector(nn.Module):
    def __init__(self, vision_dim=32, text_dim=64):
        super().__init__()
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, image_features):
        return self.proj(image_features)   # [B, N_patches, D_text]


# ────────────────────────────────────────────
# 3. Mini Text Decoder  ★ 과제 수정 부분 ★
#    input_ids를 받지 않고 image_embeds만 decoder에 통과
# ────────────────────────────────────────────
class MiniTextDecoder(nn.Module):
    def __init__(self, vocab_size=1000, text_dim=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, batch_first=True, dropout=0.0, activation='gelu'
        )
        self.decoder  = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head  = nn.Linear(text_dim, vocab_size)

    def forward(self, image_embeds, input_ids=None):
        """
        기존 : inputs_embeds = cat([image_embeds, text_embeds], dim=1)
        변경 : inputs_embeds = image_embeds  only
               input_ids는 사용하지 않음 (None으로 무시)

        sequence 구성
        ┌──────────────────────────┐
        │  image patches (196개)   │
        └──────────────────────────┘
        위치 : 0 ~ 195
        """
        # ── text 임베딩 없이 image_embeds만 그대로 사용 ──
        inputs_embeds = image_embeds   # [B, N_patches, D_text]

        # ── Causal Mask + Decoder ─────────────────────────
        seq_len     = inputs_embeds.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            inputs_embeds.device
        )
        hidden_states = self.decoder(inputs_embeds, mask=causal_mask, is_causal=True)
        logits        = self.lm_head(hidden_states)

        return logits, inputs_embeds


# ────────────────────────────────────────────
# 4. Simple Mini VLM
# ────────────────────────────────────────────
class SimpleMiniVLM(nn.Module):
    def __init__(self, image_size=224, vocab_size=1000,
                 vision_dim=32, text_dim=64, patch_size=16):
        super().__init__()
        self.vision_encoder = MiniVisionEncoder(image_size, vision_dim=vision_dim, patch_size=patch_size)
        self.projector      = MiniProjector(vision_dim, text_dim)
        self.text_decoder   = MiniTextDecoder(vocab_size, text_dim)

    def forward(self, pixel_values, input_ids=None):
        image_features = self.vision_encoder(pixel_values)          # [B, N_patches, D_vis]
        image_embeds   = self.projector(image_features)             # [B, N_patches, D_text]
        # input_ids는 decoder에 전달하지 않음 (image only)
        logits, inputs_embeds = self.text_decoder(image_embeds)
        return logits, {
            "image_features": image_features,
            "image_embeds":   image_embeds,
            "inputs_embeds":  inputs_embeds,
        }


# ────────────────────────────────────────────
# 5. 과제 출력
# ────────────────────────────────────────────
if __name__ == "__main__":
    # ── 하이퍼파라미터 ──────────────────────────
    B      = 2
    H = W  = 224
    PATCH  = 16
    L_TEXT = 8      # 기존엔 text 8개였지만 이번엔 사용 안 함
    V      = 1000
    D_VIS  = 32
    D_TEXT = 64

    N_IMG  = (H // PATCH) * (W // PATCH)   # 14×14 = 196

    # ── 기존 / 변경 sequence 길이 ───────────────
    SEQ_OLD = N_IMG + L_TEXT   # 204  (기존: image + text)
    SEQ_NEW = N_IMG            # 196  (변경: image only)

    # ── 입력 생성 ────────────────────────────────
    torch.manual_seed(0)
    pixel_values = torch.randn(B, 3, H, W)
    # input_ids는 만들지 않음 (image only이므로 불필요)

    # ── 모델 생성 및 추론 ────────────────────────
    model = SimpleMiniVLM(image_size=H, vocab_size=V,
                          vision_dim=D_VIS, text_dim=D_TEXT, patch_size=PATCH)
    model.eval()

    with torch.no_grad():
        logits, intermediates = model(pixel_values)   # input_ids 전달 안 함

    inputs_embeds = intermediates["inputs_embeds"]   # [B, 196, 64]

    # ════════════════════════════════════════════
    print("=" * 45)
    print("  [과제 2] image token only 결과")
    print("=" * 45)

    # ── (1) decoder input shape ──────────────────
    print(f"\n===== decoder input shape =====")
    print(f"inputs_embeds shape : {tuple(inputs_embeds.shape)}")
    # 기대값: (2, 196, 64)

    # ── (2) image token 개수 ─────────────────────
    print(f"\n===== image token 개수 =====")
    print(f"image token 수 : {N_IMG}")

    # ── (3) text token 개수 ──────────────────────
    print(f"\n===== text token 개수 =====")
    print(f"text token 수  : 0  (decoder에 전달하지 않음)")

    # ── (4) sequence length ──────────────────────
    print(f"\n===== sequence length =====")
    print(f"기존 seq len : {SEQ_OLD}  (image {N_IMG} + text {L_TEXT})")
    print(f"변경 seq len : {SEQ_NEW}  (image {N_IMG} only)")
    print(f"감소분       : -{SEQ_OLD - SEQ_NEW}  (text token 제거)")

    # ── (5) token별 위치 ─────────────────────────
    print(f"\n===== token별 위치 =====")
    for i in range(SEQ_NEW):
        print(f"{i:3d}: image")

    print("=" * 45)