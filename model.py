import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def gauss(x, sigma2=1, mu=0):
    return torch.exp(- (x-mu)**2 / (2*sigma2)) / (sqrt(2 * torch.pi * sigma2))


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')#计算 idx_base 并调整 idx

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # [b,4096,k,9]  索引得到每个点的 20 个邻域点的特征
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # [b,4096,1,9]

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  #(batch_size, num_dims * 2, num_points, k)



def get_part_feature(x, p2v_indices, part_rand_idx, p2v_mask=None, part_num=27, dim9=False):
    """pool all point features in each voxel to generate part feature
    Args:
        x :(B, C, N)
        p2v_indices :(B, N)
        part_num (int, optional): _description_. Defaults to 27.
        part_rand_idx: (B, 27)
    Returns:
        _type_: _description_
    """
    if dim9 == True:
        x = x[:, 6:] #丢弃x中前6维的特征通道
    
    B, C, N = x.size()

    if p2v_mask == None:
        # (B,N) -> (B,N,27)
        p2v_indices = p2v_indices.unsqueeze(-1).repeat(1, 1, part_num)
        # (B,N,27) == (B, 1, 27) -> bool: (B,N,27)
        p2v_mask = (p2v_indices == part_rand_idx.unsqueeze(1)).unsqueeze(1)
        # (B, 27)

    x = x.unsqueeze(-1).repeat(1, 1, 1, part_num)  # (B,C,N) -> (B,C,N, 27)

    # part_feature = (x * p2v_mask).max(2)[0]
    inpart_point_nums = p2v_mask.sum(2)  # (B,1,27) 得到每一部分的点数
    inpart_point_nums[inpart_point_nums == 0] = 1

    part_feature = (x * p2v_mask).sum(2) / inpart_point_nums #求和池化，求和再归一化
    # (B,C,N,27) * (B,N,27) -> (B,C,N,27) --sum-> (B,C,27)

    return part_feature, p2v_mask#(B, C, 27)：每个批次中 27 个部分的特征向量，包含 C 个通道。(B, 1, N, 27)：布尔掩码，指示每个点是否属于某个部分


def get_pointfeature_from_part(part_feature, p2v_mask, point_num=1024):
    """_summary_

    Args:
        part_feature (B,C,27):
        p2v_mask (B,N,27): 
    """
    part_feature = part_feature.unsqueeze(2).repeat(
        1, 1, point_num, 1)  # (B,C,27)->(B,C,N,27)
    part2point = (part_feature * p2v_mask).sum(-1)
    return part2point


def get_edge_feature(part_feature):
    B, C, V = part_feature.shape
    part_feature = part_feature.transpose(
        2, 1).contiguous()

    edge_feature = part_feature.view(
        B, 1, V, C).repeat(1, V, 1, 1)
    part_feature = part_feature.view(
        B, V, 1, C).repeat(1, 1, V, 1)
    feature = torch.cat((edge_feature - part_feature, part_feature),
                        dim=3).permute(0, 3, 1, 2).contiguous()
    return feature



class SSDP_v1(nn.Module):
    def __init__(self, args, distance_numclass, in_channels, out_channels):
        super(SSDP_v1, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distance_numclass = distance_numclass

        self.edgeconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.hopconv = nn.Sequential(nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(out_channels/2)),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Conv2d(
                                         int(out_channels/2), distance_numclass, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(distance_numclass),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.hopmlp = nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.edgeconv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
        # Graph attention
        self.num_head = 4
        assert out_channels % self.num_head == 0
        self.dim_per_head = out_channels // self.num_head
        self.atten1 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                        kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
        self.atten2 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                        kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
#---------------------------------------------------------------------------------------------------------------------------
        self.edgeconv1_sp = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  ##
                                          nn.BatchNorm2d(out_channels),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.hopconv_sp = nn.Sequential(nn.Conv2d(out_channels, int(out_channels / 2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(out_channels / 2)),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Conv2d(
                                         int(out_channels / 2), distance_numclass, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(distance_numclass),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.edgeconv2_sp = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(out_channels),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.atten1_sp = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                              kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
        self.atten2_sp = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                              kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))

    def forward(self, part_feature):
        B, C_in, N = part_feature.shape
        K = N

        edge_feature0 = get_edge_feature(part_feature) # (B, 2C, V, V) 通过计算每对部分之间的特征差异和保留目标部分的特征，生成描述部分之间关系的边特征


        edge_feature1 = self.edgeconv1(edge_feature0).view(B, self.dim_per_head,self.num_head, N, K)
        attention1 = self.atten1(edge_feature1)
        edge_feature1 = (attention1 * edge_feature1).view(B,-1,N,K)
#----------------------
        edge_feature1_sp = self.edgeconv1_sp(edge_feature0).view(B, self.dim_per_head, self.num_head, N, K)
        attention1_sp = self.atten1_sp(edge_feature1_sp)
        edge_feature1_sp = (attention1_sp * edge_feature1_sp).view(B, -1, N, K)
#--------------------------


        # Hop prediction
        hop_logits = self.hopconv(edge_feature1) #torch.Size([12, 7, 125, 125])
        hop = hop_logits.max(dim=1)[1] #torch.Size([12, 125, 125])
        gauss_hop = gauss(hop, self.args.sigma2).view(B, 1,1, N, K) #torch.Size([12, 1, 1, 125, 125])

#-------------------------------- HOP_sp prediction
        hop_logits_sp = self.hopconv_sp(edge_feature1_sp)
        hop_sp = hop_logits_sp.max(dim=1)[1]
        gauss_hop_sp = gauss(hop_sp, self.args.sigma2).view(B, 1, 1, N, K)

#---------------------------------
        edge_feature2 = self.edgeconv2(edge_feature1).view(B, self.dim_per_head,self.num_head, N, K)
        attention2 = self.atten2(gauss_hop * edge_feature2)
        edge_feature2 = (attention2 * edge_feature2).view(B, -1, N, K)

        g = edge_feature2.mean(dim=-1, keepdim=False)  # (B, 64, 125) torch.Size([12, 64, 125])
#-----------------------------------------------------------------------------------------------------------
        edge_feature2_sp = self.edgeconv2_sp(edge_feature1_sp).view(B, self.dim_per_head, self.num_head, N, K)
        attention2_sp = self.atten2_sp(gauss_hop_sp * edge_feature2_sp)
        edge_feature2_sp = (attention2_sp * edge_feature2_sp).view(B, -1, N, K)

        g_sp = edge_feature2_sp.mean(dim=-1, keepdim=False)

        g+=g_sp
        return g, hop_logits,hop_logits_sp  #torch.Size([12, 64, 125])  torch.Size([12, 7, 125, 125])


class SSDP_v2(nn.Module):
    def __init__(self, args, distance_numclass, in_channels, out_channels):
        super(SSDP_v2, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distance_numclass = distance_numclass

        self.edgeconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.hopconv = nn.Sequential(nn.Conv2d(out_channels, int(out_channels / 2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(out_channels / 2)),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Conv2d(
                                         int(out_channels / 2), distance_numclass, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(distance_numclass),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.hopmlp = nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.edgeconv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2))
        # Graph attention
        self.num_head = 4
        assert out_channels % self.num_head == 0
        self.dim_per_head = out_channels // self.num_head
        self.atten1 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                              kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
        self.atten2 = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(self.dim_per_head, 1,
                                              kernel_size=1, bias=False),
                                    nn.Softmax(dim=-1))
        # ---------------------------------------------------------------------------------------------------------------------------
        self.edgeconv1_sp = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  ##
                                          nn.BatchNorm2d(out_channels),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.hopconv_sp = nn.Sequential(nn.Conv2d(out_channels, int(out_channels / 2), kernel_size=1, bias=False),
                                        nn.BatchNorm2d(int(out_channels / 2)),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Conv2d(
                                            int(out_channels / 2), distance_numclass, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(distance_numclass),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.edgeconv2_sp = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(out_channels),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.atten1_sp = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv3d(self.dim_per_head, 1,
                                                 kernel_size=1, bias=False),
                                       nn.Softmax(dim=-1))
        self.atten2_sp = nn.Sequential(nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv3d(self.dim_per_head, 1,
                                                 kernel_size=1, bias=False),
                                       nn.Softmax(dim=-1))

    def forward(self, part_feature, part_distance, part_distance_sp):
        B, C_in, N = part_feature.shape
        K = N

        edge_feature0 = get_edge_feature(part_feature)  # (B, 2C, V, V) 通过计算每对部分之间的特征差异和保留目标部分的特征，生成描述部分之间关系的边特征

        edge_feature1 = self.edgeconv1(edge_feature0).view(B, self.dim_per_head, self.num_head, N, K)
        attention1 = self.atten1(edge_feature1)
        edge_feature1 = (attention1 * edge_feature1).view(B, -1, N, K)
        # ----------------------
        edge_feature1_sp = self.edgeconv1_sp(edge_feature0).view(B, self.dim_per_head, self.num_head, N, K)
        attention1_sp = self.atten1_sp(edge_feature1_sp)
        edge_feature1_sp = (attention1_sp * edge_feature1_sp).view(B, -1, N, K)
        # --------------------------

        # Hop prediction
        hop_logits = self.hopconv(edge_feature1)  # torch.Size([12, 7, 125, 125])
        hop = hop_logits.max(dim=1)[1]  # torch.Size([12, 125, 125])
        gauss_hop = gauss(hop, self.args.sigma2).view(B, 1, 1, N, K)  # torch.Size([12, 1, 1, 125, 125])

        gauss_hop_ture = gauss(part_distance, self.args.sigma2).view(B, 1,1, N, K)

        # -------------------------------- HOP_sp prediction
        hop_logits_sp = self.hopconv_sp(edge_feature1_sp)
        hop_sp = hop_logits_sp.max(dim=1)[1]
        gauss_hop_sp = gauss(hop_sp, self.args.sigma2).view(B, 1, 1, N, K)


        gauss_hop_sp_true = gauss(part_distance_sp, self.args.sigma2).view(B, 1, 1, N, K)

        # ---------------------------------
        edge_feature2 = self.edgeconv2(edge_feature1).view(B, self.dim_per_head, self.num_head, N, K)
        attention2 = self.atten2(gauss_hop_ture * edge_feature2)
        edge_feature2 = (attention2 * edge_feature2).view(B, -1, N, K)

        g = edge_feature2.mean(dim=-1, keepdim=False)  # (B, 64, 125) torch.Size([12, 64, 125])
        # -----------------------------------------------------------------------------------------------------------
        edge_feature2_sp = self.edgeconv2_sp(edge_feature1_sp).view(B, self.dim_per_head, self.num_head, N, K)
        attention2_sp = self.atten2_sp(gauss_hop_sp_true * edge_feature2_sp)
        edge_feature2_sp = (attention2_sp * edge_feature2_sp).view(B, -1, N, K)

        g_sp = edge_feature2_sp.mean(dim=-1, keepdim=False)

        g += g_sp
        return g, hop_logits, hop_logits_sp

class S4DRNetv1_v2(nn.Module):
    def __init__(self, args):
        super(S4DRNetv1_v2, self).__init__()
        self.args = args
        self.k = args.k
        distance_numclass = args.split_num + 2
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)


        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp1 = SSDP_v1(args, distance_numclass, 64*2, 64)

        # Layer 2
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp2 = SSDP_v1(args, distance_numclass, 64*2, 64)
        
        # Layer 3
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp3 = SSDP_v1(args, distance_numclass, 64*2, 64)

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 9, kernel_size=1, bias=False)
        
        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                                nn.BatchNorm1d(64),
                                                nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x, p2v_indices, part_rand_idx): #x torch.Size([12, 9, 4096]) p2v_indices torch.Size([12, 4096])

        batch_size, C, N = x.shape #输入x [b,c,n]  [b,9,4096]

        part_center, p2v_mask = get_part_feature( #(B, 3, 125)  (B, 1, N, 125)
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num**3, dim9=True)

        part_center_embedding = self.conv_pcenteremb(part_center)
        #  将体素块聚合后的空间结构    升维   （聚合后体素块的空间特征)

        # Layer 1
        x = get_graph_feature(x, k=self.k, dim9=True)# (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k) 适用于点云数据增强模块 x torch.Size([12, 18, 4096, 20])
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)  对点的特征提取
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature1 = part_center_embedding + part_feature1 #全局特征与局部特征的融合
        g1, hop1_logits ,hop1_logits_sp = self.ssdp1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)  torch.Size([12, 64, 4096])
        x1 += part2point1#torch.Size([12, 64, 4096])
        
        # Layer 2
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature2 = part_center_embedding + part_feature2
        g2, hop2_logits, hop2_logits_sp = self.ssdp2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2#torch.Size([12, 64, 4096])

        # Layer 3
        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num**3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3
        g3, hop3_logits ,hop3_logits_sp= self.ssdp3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3#torch.Size([12, 64, 4096])

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, N)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x, (hop1_logits, hop2_logits, hop3_logits) ,(hop1_logits_sp, hop2_logits_sp, hop3_logits_sp)


class S4DRNetv2_v2(nn.Module):
    def __init__(self, args):
        super(S4DRNetv2_v2, self).__init__()
        self.args = args
        self.k = args.k
        distance_numclass = args.split_num + 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)

        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp1 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        # Layer 2
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp2 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        # Layer 3
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp3 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 9, kernel_size=1, bias=False)

        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(64),
                                             nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, p2v_indices, part_rand_idx,part_distance, part_distance_sp):  # x torch.Size([12, 9, 4096]) p2v_indices torch.Size([12, 4096])

        batch_size, C, N = x.shape  # 输入x [b,c,n]  [b,9,4096]

        part_center, p2v_mask = get_part_feature(  # (B, 3, 125)  (B, 1, N, 125)
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num ** 3, dim9=True)

        part_center_embedding = self.conv_pcenteremb(part_center)
        #  将体素块聚合后的空间结构    升维   （聚合后体素块的空间特征)

        # Layer 1
        x = get_graph_feature(x, k=self.k,
                              dim9=True)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k) 适用于点云数据增强模块 x torch.Size([12, 18, 4096, 20])
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)  对点的特征提取
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature1 = part_center_embedding + part_feature1  # 全局特征与局部特征的融合
        g1, hop1_logits, hop1_logits_sp = self.ssdp1(part_feature1,part_distance, part_distance_sp)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)  torch.Size([12, 64, 4096])
        x1 += part2point1  # torch.Size([12, 64, 4096])

        # Layer 2
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature2 = part_center_embedding + part_feature2
        g2, hop2_logits, hop2_logits_sp = self.ssdp2(part_feature2,part_distance, part_distance_sp)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2  # torch.Size([12, 64, 4096])

        # Layer 3
        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3
        g3, hop3_logits, hop3_logits_sp = self.ssdp3(part_feature3,part_distance, part_distance_sp)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3  # torch.Size([12, 64, 4096])

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, N)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x, (hop1_logits, hop2_logits, hop3_logits), (hop1_logits_sp, hop2_logits_sp, hop3_logits_sp)


class S4DRNetv1_v1(nn.Module):
    def __init__(self, args):
        super(S4DRNetv1_v1, self).__init__()
        self.args = args
        self.k = args.k
        distance_numclass = args.split_num + 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)

        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp1 = SSDP_v1(args, distance_numclass, 64 * 2, 64)

        # Layer 2
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp2 = SSDP_v1(args, distance_numclass, 64 * 2, 64)

        # Layer 3
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp3 = SSDP_v1(args, distance_numclass, 64 * 2, 64)

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 8, kernel_size=1, bias=False)

        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(64),
                                             nn.LeakyReLU(negative_slope=0.2))

    def freeze_layers_1(self):
        # 冻结层的参数
        for param in self.conv6.parameters():
            param.requires_grad = False
        for param in self.conv7.parameters():
            param.requires_grad = False
        for param in self.conv8.parameters():
            param.requires_grad = False
        for param in self.conv9.parameters():
            param.requires_grad = False

    def freeze_layers_2(self):
        """
        Freeze all layers except conv6, conv7, conv8, and conv9.
        """
        for name, param in self.named_parameters():
            # Check if the layer is one of the layers that should not be frozen
            if 'conv6' not in name and 'conv7' not in name and 'conv8' not in name and 'conv9' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Make sure conv6, conv7, conv8, conv9 are trainable


    def forward(self, x, p2v_indices, part_rand_idx):  # x torch.Size([12, 9, 4096]) p2v_indices torch.Size([12, 4096])

        batch_size, C, N = x.shape  # 输入x [b,c,n]  [b,9,4096]

        part_center, p2v_mask = get_part_feature(  # (B, 3, 125)  (B, 1, N, 125)
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num ** 3, dim9=True)

        part_center_embedding = self.conv_pcenteremb(part_center)
        #  将体素块聚合后的空间结构    升维   （聚合后体素块的空间特征)

        # Layer 1
        x = get_graph_feature(x, k=self.k,
                              dim9=True)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k) 适用于点云数据增强模块 x torch.Size([12, 18, 4096, 20])
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)  对点的特征提取
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature1 = part_center_embedding + part_feature1  # 全局特征与局部特征的融合
        g1, hop1_logits, hop1_logits_sp = self.ssdp1(part_feature1)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)  torch.Size([12, 64, 4096])
        x1 += part2point1  # torch.Size([12, 64, 4096])

        # Layer 2
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature2 = part_center_embedding + part_feature2
        g2, hop2_logits, hop2_logits_sp = self.ssdp2(part_feature2)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2  # torch.Size([12, 64, 4096])

        # Layer 3
        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3
        g3, hop3_logits, hop3_logits_sp = self.ssdp3(part_feature3)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3  # torch.Size([12, 64, 4096])

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, N)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x, (hop1_logits, hop2_logits, hop3_logits), (hop1_logits_sp, hop2_logits_sp, hop3_logits_sp)


class S4DRNetv2_v1(nn.Module):
    def __init__(self, args):
        super(S4DRNetv2_v1, self).__init__()
        self.args = args
        self.k = args.k
        distance_numclass = args.split_num + 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)

        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp1 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        # Layer 2
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp2 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        # Layer 3
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.ssdp3 = SSDP_v2(args, distance_numclass, 64 * 2, 64)

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 8, kernel_size=1, bias=False)

        self.conv_pcenteremb = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(64),
                                             nn.LeakyReLU(negative_slope=0.2))

    def freeze_layers_1(self):
        # 冻结层的参数
        for param in self.conv6.parameters():
            param.requires_grad = False
        for param in self.conv7.parameters():
            param.requires_grad = False
        for param in self.conv8.parameters():
            param.requires_grad = False
        for param in self.conv9.parameters():
            param.requires_grad = False

    def freeze_layers_2(self):
        """
        Freeze all layers except conv6, conv7, conv8, and conv9.
        """
        for name, param in self.named_parameters():
            # Check if the layer is one of the layers that should not be frozen
            if 'conv6' not in name and 'conv7' not in name and 'conv8' not in name and 'conv9' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Make sure conv6, conv7, conv8, conv9 are trainable


    def forward(self, x, p2v_indices, part_rand_idx,part_distance, part_distance_sp):  # x torch.Size([12, 9, 4096]) p2v_indices torch.Size([12, 4096])

        batch_size, C, N = x.shape  # 输入x [b,c,n]  [b,9,4096]

        part_center, p2v_mask = get_part_feature(  # (B, 3, 125)  (B, 1, N, 125)
            x, p2v_indices, part_rand_idx, part_num=self.args.split_num ** 3, dim9=True)

        part_center_embedding = self.conv_pcenteremb(part_center)
        #  将体素块聚合后的空间结构    升维   （聚合后体素块的空间特征)

        # Layer 1
        x = get_graph_feature(x, k=self.k,
                              dim9=True)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k) 适用于点云数据增强模块 x torch.Size([12, 18, 4096, 20])
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)  对点的特征提取
        part_feature1, _ = get_part_feature(
            x1, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature1 = part_center_embedding + part_feature1  # 全局特征与局部特征的融合
        g1, hop1_logits, hop1_logits_sp = self.ssdp1(part_feature1,part_distance, part_distance_sp)
        part2point1 = get_pointfeature_from_part(g1, p2v_mask, N)  # (B,C,N)  torch.Size([12, 64, 4096])
        x1 += part2point1  # torch.Size([12, 64, 4096])

        # Layer 2
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature2, _ = get_part_feature(
            x2, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature2 = part_center_embedding + part_feature2
        g2, hop2_logits, hop2_logits_sp = self.ssdp2(part_feature2,part_distance, part_distance_sp)
        part2point2 = get_pointfeature_from_part(g2, p2v_mask, N)  # (B,C,N)
        x2 += part2point2  # torch.Size([12, 64, 4096])

        # Layer 3
        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        part_feature3, _ = get_part_feature(
            x3, p2v_indices, part_rand_idx, p2v_mask=p2v_mask, part_num=self.args.split_num ** 3)  # (B, C_128, 27)
        part_feature3 = part_center_embedding + part_feature3
        g3, hop3_logits, hop3_logits_sp = self.ssdp3(part_feature3,part_distance, part_distance_sp)
        part2point3 = get_pointfeature_from_part(g3, p2v_mask, N)  # (B,C,N)
        x3 += part2point3  # torch.Size([12, 64, 4096])

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, N)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x, (hop1_logits, hop2_logits, hop3_logits), (hop1_logits_sp, hop2_logits_sp, hop3_logits_sp)