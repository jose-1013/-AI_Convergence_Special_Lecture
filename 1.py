"""
[과제] IMG_START / IMG_END 특수 토큰 삽입
기존 : image_tokens + text_tokens
변경 : [IMG_START] + image_tokens + [IMG_END] + text_tokens
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
        x = self.patch_embed(pixel_values)          # [B, D, H/p, W/p]
        image_features = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
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
        return self.proj(image_features)  # [B, N_patches, D_text]


# ────────────────────────────────────────────
# 3. Mini Text Decoder  ★ 과제 수정 부분 ★
#    vocab_size + 2 로 확장하여
#    token (vocab_size)   = IMG_START
#    token (vocab_size+1) = IMG_END
# ────────────────────────────────────────────
class MiniTextDecoder(nn.Module):
    def __init__(self, vocab_size=1000, text_dim=64):
        super().__init__()

        # ── 특수 토큰 ID 정의 ──────────────────────
        self.IMG_START_ID = vocab_size        # 1000
        self.IMG_END_ID   = vocab_size + 1    # 1001
        extended_vocab    = vocab_size + 2    # 1002

        # ── 임베딩 / 디코더 / lm_head ─────────────
        self.token_embed = nn.Embedding(extended_vocab, text_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, batch_first=True, dropout=0.0, activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head  = nn.Linear(text_dim, extended_vocab)

    def forward(self, image_embeds, input_ids):
        """
        image_embeds : [B, N_patches, D_text]
        input_ids    : [B, L_text]

        sequence 구성
        ┌──────────┬──────────────────────┬─────────┬──────────────┐
        │ IMG_START│  image patches (196) │ IMG_END │  text tokens │
        └──────────┴──────────────────────┴─────────┴──────────────┘
        위치 :   0          1 ~ 196          197       198 ~ 205
        """
        B = image_embeds.size(0)

        # ── IMG_START / IMG_END 임베딩 ────────────
        img_start_ids = torch.full((B, 1), self.IMG_START_ID,
                                   dtype=torch.long, device=image_embeds.device)
        img_end_ids   = torch.full((B, 1), self.IMG_END_ID,
                                   dtype=torch.long, device=image_embeds.device)

        img_start_embed = self.token_embed(img_start_ids)  # [B, 1, D_text]
        img_end_embed   = self.token_embed(img_end_ids)    # [B, 1, D_text]

        # ── 텍스트 임베딩 ─────────────────────────
        text_embeds = self.token_embed(input_ids)          # [B, L_text, D_text]

        # ── sequence 결합 ─────────────────────────
        # [IMG_START] + image_tokens + [IMG_END] + text_tokens
        inputs_embeds = torch.cat(
            [img_start_embed, image_embeds, img_end_embed, text_embeds],
            dim=1
        )  # [B, 1 + N_patches + 1 + L_text, D_text]

        # ── Causal Mask + Decoder ─────────────────
        seq_len    = inputs_embeds.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            inputs_embeds.device
        )
        hidden_states = self.decoder(inputs_embeds, mask=causal_mask, is_causal=True)
        logits        = self.lm_head(hidden_states)

        return logits, inputs_embeds


# ────────────────────────────────────────────
# 4. Simple Mini VLM (원본과 동일 구조)
# ────────────────────────────────────────────
class SimpleMiniVLM(nn.Module):
    def __init__(self, image_size=224, vocab_size=1000,
                 vision_dim=32, text_dim=64, patch_size=16):
        super().__init__()
        self.vision_encoder = MiniVisionEncoder(image_size, vision_dim=vision_dim, patch_size=patch_size)
        self.projector      = MiniProjector(vision_dim, text_dim)
        self.text_decoder   = MiniTextDecoder(vocab_size, text_dim)

    def forward(self, pixel_values, input_ids):
        image_features = self.vision_encoder(pixel_values)
        image_embeds   = self.projector(image_features)
        logits, inputs_embeds = self.text_decoder(image_embeds, input_ids)
        return logits, {
            "image_features": image_features,
            "image_embeds":   image_embeds,
            "inputs_embeds":  inputs_embeds,
        }


# ────────────────────────────────────────────
# 5. 과제 출력
# ────────────────────────────────────────────
if __name__ == "__main__":
    # 하이퍼파라미터
    B       = 2
    H = W   = 224
    PATCH   = 16
    L_TEXT  = 8
    V       = 1000
    D_VIS   = 32
    D_TEXT  = 64

    N_IMG   = (H // PATCH) * (W // PATCH)   # 196  (14×14)
    # 특수 토큰 2개 추가로 sequence 증가
    # 기존: N_IMG + L_TEXT = 204
    # 변경: 1(IMG_START) + N_IMG + 1(IMG_END) + L_TEXT = 206
    SEQ_OLD = N_IMG + L_TEXT
    SEQ_NEW = 1 + N_IMG + 1 + L_TEXT

    torch.manual_seed(0)
    pixel_values = torch.randn(B, 3, H, W)
    input_ids    = torch.randint(0, V, (B, L_TEXT))

    model = SimpleMiniVLM(image_size=H, vocab_size=V,
                          vision_dim=D_VIS, text_dim=D_TEXT, patch_size=PATCH)
    model.eval()

    with torch.no_grad():
        logits, intermediates = model(pixel_values, input_ids)

    inputs_embeds = intermediates["inputs_embeds"]  # [B, SEQ_NEW, D_TEXT]

    print("=" * 55)
    print("  [과제] IMG_START / IMG_END 토큰 삽입 결과")
    print("=" * 55)

    # ── (1) decoder input shape ────────────────────────────
    print(f"\n[1] decoder input shape : {tuple(inputs_embeds.shape)}")
    # 기대값: (2, 206, 64)

    # ── (2) sequence length 증가분 ─────────────────────────
    print(f"\n[2] sequence length 증가분")
    print(f"    기존 seq len : {SEQ_OLD}  (N_img={N_IMG} + L_text={L_TEXT})")
    print(f"    변경 seq len : {SEQ_NEW}  (1 + N_img={N_IMG} + 1 + L_text={L_TEXT})")
    print(f"    증가분       : +{SEQ_NEW - SEQ_OLD}  (IMG_START 1개 + IMG_END 1개)")

    # ── (3) 첫 번째 / 마지막 image token ──────────────────
    # 위치 0 = IMG_START → image patch 시작은 위치 1
    # image patch 끝  = 위치 N_IMG (= 196)
    # 위치 N_IMG+1    = IMG_END
    first_img_token_pos = 1
    last_img_token_pos  = N_IMG          # 196

    first_img_token = inputs_embeds[0, first_img_token_pos, :]  # batch 0
    last_img_token  = inputs_embeds[0, last_img_token_pos,  :]

    print(f"\n[3] image token 위치 (batch=0 기준)")
    print(f"    첫 번째 image token  위치 {first_img_token_pos:3d} → shape {tuple(first_img_token.shape)}")
    print(f"    마지막 image token 위치 {last_img_token_pos:3d} → shape {tuple(last_img_token.shape)}")
    print(f"    첫 번째 image token 값 (앞 5개): {first_img_token[:5].tolist()}")
    print(f"    마지막 image token 값 (앞 5개): {last_img_token[:5].tolist()}")

    # ── (4) token별 위치 출력 ──────────────────────────────
    IMG_START_ID = V          # 1000
    IMG_END_ID   = V + 1      # 1001

    img_start_pos  = 0
    img_patch_start= 1
    img_patch_end  = N_IMG           # 196  (inclusive)
    img_end_pos    = N_IMG + 1       # 197
    text_start     = N_IMG + 2       # 198
    text_end       = SEQ_NEW - 1     # 205

    print(f"\n[4] token별 위치")
    print(f"    {img_start_pos}          : IMG_START  (token id={IMG_START_ID})")
    print(f"    {img_patch_start} ~ {img_patch_end}    : image patch tokens  ({N_IMG}개)")
    print(f"    {img_end_pos}        : IMG_END    (token id={IMG_END_ID})")
    print(f"    {text_start} ~ {text_end}    : text tokens  ({L_TEXT}개)")
    print(f"    total seq len = {SEQ_NEW}")
    print("=" * 55)