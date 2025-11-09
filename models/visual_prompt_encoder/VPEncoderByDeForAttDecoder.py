import math
import torch
import torch.nn as nn
from models.visual_prompt_encoder.utils import gen_sineembed_for_position
from detrex.layers.multi_scale_deform_attn import MultiScaleDeformableAttention


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along x-axis or y-axis.
            The final returned dimension for each position is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position embedding.
            The scale will be used only when `normalize` is True. Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding. Default: False.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set, scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing ignored positions,
                while zero values means valid positions for the input tensor. Shape as `(bs, h, w)`.
        Returns:
            torch.Tensor: Returned position embedding with shape `(bs, num_pos_feats * 2, h, w)`
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = ((y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale)
            x_embed = ((x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class SelfAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, position=None, attn_mask=None, key_padding_mask=None):
        # MultiheadAttention要求输入的形状为: (sequence_length, batch_size, embed_dim)
        # x = x.transpose(0, 1)
        # 添加位置编码（如果上下文存在位置关系，可以添加位置编码信息，否则可以不添加）
        x = x if position is None else x + position
        # 计算自注意力
        att_output, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        att_output = self.dropout(att_output)
        # 残差连接和规范化
        x = self.norm(x + att_output)
        # 将形状恢复为(batch_size, sequence_length, embed_dim)
        # x = x.transpose(0, 1)

        return x


class VPEncoderLayerByDeForAtt(nn.Module):

    def __init__(self, embed_dim=256, batch_first=True):
        super().__init__()

        self.cross_att = MultiScaleDeformableAttention(embed_dim=embed_dim, batch_first=batch_first)
        self.ffn = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_pos, reference_points, spatial_shapes, level_start_index, key_padding_mask, **kwargs):
        query = self.cross_att(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )

        return self.ffn(query)


class VPEncoderByDeForAttDecoder(nn.Module):

    def __init__(self, layer_num=3, embed_dim=256, feature_levels_num=4, num_heads=8, batch_first=True, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_levels_num = feature_levels_num
        self.cls_embed = torch.nn.Embedding(1, embed_dim)
        self.level_embeds = nn.Parameter(torch.Tensor(feature_levels_num, embed_dim))  # 提供3D位置编码
        self.temp_id = torch.tensor([0], device=device)
        # self.layers = nn.ModuleList([
        #     VPEncoderLayerByDeForAtt(embed_dim=embed_dim, batch_first=batch_first) for _ in range(layer_num)
        # ])
        self.VPELayers = nn.ModuleList([
            VPEncoderLayerByDeForAtt(embed_dim=embed_dim, batch_first=batch_first) for _ in range(layer_num)
        ])
        self.CLSFusionLayers = nn.ModuleList([
            SelfAttentionLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(layer_num)
        ])
        self.FFNLayers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(layer_num)])
        # 规范化层
        self.LayerNorm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layer_num)])
        self.LayerNorm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layer_num)])
        self.gammas1 = [nn.Parameter(torch.zeros(1, device=device)) for _ in range(layer_num)]
        self.gammas2 = [nn.Parameter(torch.zeros(1, device=device)) for _ in range(layer_num)]

    @staticmethod
    def get_valid_ratio(mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

        return valid_ratio

    @staticmethod
    def get_vp_reference_points(reference_points, spatial_shapes, valid_ratios, device):
        """Get the visual prompt reference points used in decoder.

        Args:
            reference_points(Tensor): visual prompt reference points (bs, num, 2)
            spatial_shapes (Tensor): The shape of all feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid points on the feature map, has shape (bs, num_levels, 2)
            device (obj:`device`): The device where reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has shape (bs, num_keys, num_levels, 2).
        """
        # valid_ratios存在的意义参考 https://github.com/open-mmlab/mmdetection/issues/8656#issuecomment-1295866151
        # 主要是因为img下采样过程中存在量化损失导致各个feature map的有效宽比不一致，将当前参考点映射到各个level时需要单独处理
        reference_points_list = []
        H = spatial_shapes[0][0]
        W = spatial_shapes[0][1]
        ref_x = reference_points[:, :, 0] / (valid_ratios[:, None, 0, 0] * W * 8)
        ref_y = reference_points[:, :, 1] / (valid_ratios[:, None, 0, 1] * H * 8)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points.to(device)

        return reference_points

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, reference_points, attn_mask=None, key_padding_mask=None):

        # 构建flatten的多尺度特征、mask、shape、flatten feature map起始索引，query参考点
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []    # deformable attention decoder的不涉及key使用，注意力权重由query构建
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)   # 使用3D位置编码
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        # 获取visual prompt多尺度feature map上归一化参考点
        normal_reference_points = self.get_vp_reference_points(reference_points, spatial_shapes, valid_ratios, feat.device)
        # 获取visual prompt位置编码
        query_content_pos = gen_sineembed_for_position(normal_reference_points[:, :, 0, :])
        # 复制visual类别编码
        query_content = self.cls_embed(self.temp_id)[:, None].repeat(bs, reference_points.shape[1], 1)

        for VPELayer, CLSFusionLayer, FFNLayer, ln1, ln2, gm1, gm2 \
                in zip(self.VPELayers, self.CLSFusionLayers, self.FFNLayers, self.LayerNorm1, self.LayerNorm2, self.gammas1, self.gammas2):
        # for VPELayer in self.VPELayers:
            query_content = VPELayer(
                query=query_content,
                key=feat_flatten,
                value=feat_flatten,
                query_pos=query_content_pos,
                reference_points=normal_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask_flatten,
                key_pos=lvl_pos_embed_flatten,
            )
            # query_content = CLSFusionLayer(query_content, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            # query_content = FFNLayer(query_content)
            # query_content = query_content + gm1 * CLSFusionLayer(ln1(query_content), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            # query_content = query_content + gm2 * FFNLayer(ln2(query_content))

        return query_content

