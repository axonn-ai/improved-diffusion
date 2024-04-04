import pandas as pd
import numpy as np

def tflops_per_gpu(global_batch_size,
                   seq_length,
                   num_layers,
                   hidden_size,
                   num_gpus,
                   vocab_size=51200,
                   gpu="a100"
                  ):
    teraflop_in_batch = 96*global_batch_size*seq_length*num_layers*(hidden_size**2)*(1+seq_length/(6*hidden_size)+(vocab_size)/(16*num_layers*hidden_size))/(1e12)
    tflops_per_gpu =  teraflop_in_batch/num_gpus
    return tflops_per_gpu

def model_v1(B, S, K, H, Gd, Gr, Gc, Gdata):
    dp_comm = 2 * (Gdata-1) / Gdata * K * H * H / (Gc*Gr*Gd)
    depth_tensor_comm = 2 * (Gd-1) / Gd * K * H * H / (Gc*Gr)
    row_tensor_comm = 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * H/Gc ) 
    col_tensor_comm = 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * K * H/Gr )
    return dp_comm + depth_tensor_comm + row_tensor_comm + col_tensor_comm


def get_bw(ip, my, machine="perlmutter", version="v2"):
    if version == "v1":
        return 1

    if machine == "perlmutter":
        if ip==1:
            if my==2:
                return 76
            elif my==4:
                return 225
            elif my>=8:
                return 80
        elif ip == 2:
            if my == 2:
                return 76
            elif my >= 4:
                return 40
        elif ip >= 4:
            return 20
    elif machine == "frontier":
        if ip==1:
            if my==2:
                return 129
            elif my==4:
                return 52
            elif my==8:
                return 135
            else:
                return 80 / 2 # 34.031
        elif ip == 2:
            if my == 2:
                return 50
            elif my == 4:
                return 72
            else:
                return 40 / 2  # 20.62
        elif ip == 4:
            if my == 2:
                return 36 
            else:
                return 20 / 2 # 10.18
        elif ip >=8: 
            return 10 / 2 # 


def model_v2(B, S, IC, OC, Gd, Gr, Gc, Gdata, kernel=3, machine="perlmutter", transpose=False):
    """
    B - batch size
    S - number of pixels in image
    IC - input channels
    OC - output channels
    GD, Gr, Gc - tensor parallel dimensions
    Gdata - degree of data parallelism
    kernel - conv kernel dim
    """
    dp_comm = 4 * (Gdata-1) / Gdata * kernel * kernel * IC * OC / (Gc*Gr*Gd) * 2 /1024/1024/1024
    depth_tensor_comm = 3/2 * 2 * (Gd-1) / Gd * (kernel * kernel * IC * OC) / (Gc*Gr) * 2 /1024/1024/1024
    if not transpose:
        row_tensor_comm = 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * IC/Gc ) * 2 /1024/1024/1024
        col_tensor_comm = 2 * 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * OC/Gr ) * 2 /1024/1024/1024
    else:
        col_tensor_comm = 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * IC/Gr ) * 2 /1024/1024/1024
        row_tensor_comm = 2 * 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * OC/Gc ) * 2 /1024/1024/1024


    col_time = row_time = depth_time = data_time = 0 
    

    ip=1
    if Gc > 1:
        col_bw = get_bw(ip, Gc, machine)
        if col_bw is None:
            return None
        col_time = col_tensor_comm / col_bw 
    ip*= Gc 
    if Gr > 1:
        row_bw = get_bw(ip, Gr, machine)
        if row_bw is None:
            return None
        row_time = row_tensor_comm / row_bw 

    ip *= Gr 
    
    if Gd > 1:
        depth_bw = get_bw(ip, Gd, machine)
        if depth_bw is None:
            return None
        depth_time = depth_tensor_comm / depth_bw 

    ip *= Gd

    if Gdata > 1: 
        data_bw = get_bw(ip, Gdata, machine)
        if data_bw is None:
            return None
        data_time = dp_comm / data_bw 


    return col_time + row_time + depth_time + data_time

def model_resblock(B, S, IC, OC, Gd, Gr, Gc, Gdata, kernel=3, machine="perlmutter"):
    cost = 0
    cost += model_v2(B, S, IC, OC, Gd, Gr, Gc, Gdata, kernel=kernel, machine=machine)
    cost += model_v2(B, S, OC, OC, Gd, Gr, Gc, Gdata, kernel=kernel, machine=machine, transpose=True)
    if IC != OC:
        cost += model_v2(B, S, IC, OC, Gd, Gr, Gc, Gdata, kernel=kernel, machine=machine)
    return cost

def get_configs_for_unet(
        global_batch_size_in_samples,
        sequence_length,
        channels, 
        GPUs,
        minimum_degree_of_tensor_parallelism,
        model_version="v2",
        topk=5,
        no_dp=False,
        machine="perlmutter",
        limit=None
):
    S=sequence_length
    K=3
    B=global_batch_size_in_samples
    C=channels
    G=GPUs
    min_tp=minimum_degree_of_tensor_parallelism


    range = []
    i=0
    while 2**i <=G:
        range.append(2**i)
        i+=1

    data = {}
    ct = 0
    for Gc in range:
        for Gr in range:
            for Gd in range:
                for Gdata in range:
                    if Gc*Gr*Gd*Gdata == G and Gc*Gr*Gd>=min_tp and B%(Gdata*Gd)==0 and (not no_dp or Gdata==1):
                        #data[(Gc,Gr,Gd,Gdata)] =
                        ct += 1
                        # the 4 fc layers of a transformer,
                        # I swap Gc and Gr for "transposed layers" 
                        if limit is not None:
                            if Gc>limit or Gr>limit or Gd>limit or Gdata>limit:
                                continue
                        #model_v2(B, S, IC, OC, Gd, Gr, Gc, Gdata, kernel=3, machine="perlmutter")
                        # a = model_v2(B, S, 3, H, Gd, Gr, Gc, Gdata, machine) 
                        # b = model_v2(B, S, 1, H, Gd, Gc, Gr, Gdata, machine) 
                        # c = model_v2(B, S, 4, H, Gd, Gr, Gc, Gdata, machine) 
                        # d = model_v2(B, S, 4, H, Gd, Gc, Gr, Gdata, machine)
                        cost = 0
                        dim = S
                        for level in np.arange(4):
                            if level == 0:
                                cost += 3 * model_resblock(B, dim, 
                                                     C, C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                            elif level == 1:
                                cost += model_resblock(B, dim, # downsampled image 
                                                     C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                                cost += 2*model_resblock(B, dim, # downsampled image 
                                                     2*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                            else:
                                cost += 3*model_resblock(B, dim, # downsampled image 
                                                     2*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                            dim //= 4

                        # MIDDLE BLOCK
                        cost += 2 * model_resblock(B, dim, # downsampled image 
                                                     2*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)

                        for level in np.arange(4,-1,-1):
                            if level in [4, 3]:
                                cost += 3 * model_resblock(B, dim, 
                                                     4*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                            elif level == 2:
                                cost += 2 * model_resblock(B, dim, 
                                                     4*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                                cost +=  model_resblock(B, dim, 
                                                     3*C, 2*C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                                
                            else:
                                cost += 3*model_resblock(B, dim, # downsampled image 
                                                     2*C, C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                                cost += model_resblock(B, dim, # downsampled image 
                                                     3*C, C, 
                                                     Gd, Gr, Gc, Gdata, 
                                                     kernel=3, machine=machine)
                                
                            dim *= 4


                        data[(Gc,Gr,Gd,Gdata)] = cost
                        
    sorted_configs = sorted(data.items(), key=lambda x:x[1])

    # "Tried" to model computation, but this underestimates the time I think
    
    keys = "Gr","Gc","Gd","Gdata", "Comm-Time(s)"#,"Total-Time(s)"
    data = []
    for (Gc, Gr, Gd, Gdata), comm_time in sorted_configs[:topk]:
        data.append([Gr, Gc, Gd, Gdata, comm_time]), #total_time])

    df = pd.DataFrame(data, columns=keys)
    return df
