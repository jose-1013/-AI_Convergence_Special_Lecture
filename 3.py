"""
[과제 3] learnable special token [IMG_SUM] 추가
기존 : projected_image_tokens + text_tokens
변경 : [IMG_SUM] + projected_image_tokens + text_tokens

[IMG_SUM]은 nn.Parameter로 선언된 학습 가능한 벡터 1개.
고정된 임베딩이 아니라 학습을 통해 "이미지 전체 요약" 정보를 담도록 유도됨.
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
        x = self.patch_embed(pixel_values)
        image_features = x.flatten(2).transpose(1, 2)
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
#    IMG_SUM : shape [1, 1, text_dim] 의 learnable parameter
#    매 forward마다 배치 크기만큼 expand해서 sequence 맨 앞에 붙임
# ────────────────────────────────────────────
class MiniTextDecoder(nn.Module):
    def __init__(self, vocab_size=1000, text_dim=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, text_dim)

        # ── [IMG_SUM] learnable token ──────────────────────────────
        # nn.Parameter : 이 벡터는 모델 파라미터로 등록되어 gradient가 흐르고 학습됨
        # shape [1, 1, text_dim] : (배치 방향, 시퀀스 방향, 차원)
        # 배치 방향을 1로 두면 나중에 expand로 B만큼 복사할 수 있음
        self.img_sum_token = nn.Parameter(torch.randn(1, 1, text_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, batch_first=True, dropout=0.0, activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(text_dim, vocab_size)

    def forward(self, image_embeds, input_ids):
        """
        sequence 구성
        ┌──────────┬──────────────────────┬──────────────┐
        │ IMG_SUM  │  image patches (196) │  text tokens │
        └──────────┴──────────────────────┴──────────────┘
        위치 :   0          1 ~ 196           197 ~ 204
        """
        B = image_embeds.size(0)

        # ── IMG_SUM을 배치 크기만큼 복사 ──────────────────────────
        # self.img_sum_token : [1, 1, 64]
        # expand(B, 1, 64)   : [B, 1, 64]  (메모리 복사 없이 뷰만 확장)
        img_sum = self.img_sum_token.expand(B, 1, -1)   # [B, 1, D_text]

        # ── 텍스트 임베딩 ──────────────────────────────────────────
        text_embeds = self.token_embed(input_ids)        # [B, L_text, D_text]

        # ── sequence 결합 ──────────────────────────────────────────
        # [IMG_SUM] + image_tokens + text_tokens
        inputs_embeds = torch.cat(
            [img_sum, image_embeds, text_embeds],
            dim=1
        )   # [B, 1 + N_patches + L_text, D_text]

        # ── Causal Mask + Decoder ──────────────────────────────────
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
# 5. 원본 모델 (비교용, IMG_SUM 없는 버전)
# ────────────────────────────────────────────
class MiniTextDecoderOriginal(nn.Module):
    """IMG_SUM 없는 원본 decoder - 파라미터 수 비교용"""
    def __init__(self, vocab_size=1000, text_dim=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, batch_first=True, dropout=0.0, activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(text_dim, vocab_size)

class SimpleMiniVLMOriginal(nn.Module):
    def __init__(self, image_size=224, vocab_size=1000,
                 vision_dim=32, text_dim=64, patch_size=16):
        super().__init__()
        self.vision_encoder = MiniVisionEncoder(image_size, vision_dim=vision_dim, patch_size=patch_size)
        self.projector      = MiniProjector(vision_dim, text_dim)
        self.text_decoder   = MiniTextDecoderOriginal(vocab_size, text_dim)


# ────────────────────────────────────────────
# 6. 과제 출력
# ────────────────────────────────────────────
if __name__ == "__main__":
    # ── 하이퍼파라미터 ──────────────────────────
    B      = 2
    H = W  = 224
    PATCH  = 16
    L_TEXT = 8
    V      = 1000
    D_VIS  = 32
    D_TEXT = 64

    N_IMG  = (H // PATCH) * (W // PATCH)   # 196
    SEQ_OLD = N_IMG + L_TEXT               # 204
    SEQ_NEW = 1 + N_IMG + L_TEXT           # 205

    # ── 입력 생성 ────────────────────────────────
    torch.manual_seed(0)
    pixel_values = torch.randn(B, 3, H, W)
    input_ids    = torch.randint(0, V, (B, L_TEXT))

    # ── 모델 생성 및 추론 ────────────────────────
    model          = SimpleMiniVLM(image_size=H, vocab_size=V,
                                   vision_dim=D_VIS, text_dim=D_TEXT, patch_size=PATCH)
    model_original = SimpleMiniVLMOriginal(image_size=H, vocab_size=V,
                                           vision_dim=D_VIS, text_dim=D_TEXT, patch_size=PATCH)
    model.eval()

    with torch.no_grad():
        logits, intermediates = model(pixel_values, input_ids)

    inputs_embeds = intermediates["inputs_embeds"]   # [B, 205, 64]

    print("=" * 55)
    print("  [과제 3] learnable [IMG_SUM] 토큰 추가 결과")
    print("=" * 55)

    # ── (1) decoder input shape ──────────────────────────────
    print(f"\n===== decoder input shape =====")
    print(f"inputs_embeds shape : {tuple(inputs_embeds.shape)}")
    # 기대값: (2, 205, 64)

    # ── (2) 첫 번째 token 값 ─────────────────────────────────
    print(f"\n===== 첫 번째 token 값 (batch=0) =====")
    first_token = inputs_embeds[0, 0, :]   # [64]
    print(f"shape : {tuple(first_token.shape)}")
    print(f"값 (앞 8개) : {[round(v, 4) for v in first_token[:8].tolist()]}")
    print(f"→ 이 값은 nn.Parameter(img_sum_token)에서 온 learnable 벡터")

    # ── (3) 파라미터 수 비교 ─────────────────────────────────
    print(f"\n===== parameter 개수 증가 =====")

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    params_new      = count_params(model)
    params_original = count_params(model_original)
    params_added    = params_new - params_original

    print(f"원본 모델 파라미터 수  : {params_original:,}")
    print(f"변경 모델 파라미터 수  : {params_new:,}")
    print(f"증가분                 : +{params_added}  ← img_sum_token의 차원 수 (D_TEXT={D_TEXT})")
    print(f"→ nn.Parameter shape [1, 1, {D_TEXT}] = {D_TEXT}개 파라미터만 추가됨")

    # ── (4) IMG_SUM 역할 설명 ────────────────────────────────
    print(f"\n===== IMG_SUM 역할 =====")
    print("1. 이미지 전체 요약 (CLS token 역할)")
    print("   - 위치 0에 고정되어 196개 image patch 전체를 causal attention으로 집약")
    print("   - 학습 후 이 벡터 하나로 이미지의 전반적 의미를 담음")
    print("")
    print("2. 고정 임베딩이 아닌 learnable parameter")
    print("   - nn.Parameter이므로 backprop 때 gradient가 흘러 학습됨")
    print("   - 이미지마다 달라지는 게 아니라 '모든 이미지에 공통으로 쓰이는")
    print("     요약 슬롯'으로서 학습됨")
    print("")
    print("3. text token과의 연결")
    print("   - text token들이 attention 계산 시 위치 0(IMG_SUM)을 참조해")
    print("     이미지 전체 맥락을 한 번에 가져올 수 있음")
    print("   - 196개 patch를 하나하나 보지 않아도 IMG_SUM만 보면 이미지 요약을 얻음")

    # ── (5) token별 위치 ─────────────────────────────────────
    print(f"\n===== token별 위치 =====")
    print(f"0          : IMG_SUM  (learnable parameter)")
    print(f"1 ~ {N_IMG}    : image patch tokens  ({N_IMG}개)")
    print(f"{N_IMG+1} ~ {SEQ_NEW-1}    : text tokens  ({L_TEXT}개)")
    print(f"total seq len = {SEQ_NEW}  (기존 {SEQ_OLD} + 1)")

    print("=" * 55)