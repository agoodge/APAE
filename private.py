def import_data(dataset):
    
    if dataset == 'SWaT':

        normal_data_path = "Data/SWaT_Dataset_Normal_v0.csv"
        normal_data = pd.read_csv(normal_data_path, delimiter=",")
        normal_data = normal_data.drop(["Timestamp", "AIT201"], axis=1)

        attack_data_path = "Data/SWaT_Dataset_Attack_v0.csv"
        attack_data = pd.read_csv(attack_data_path, delimiter=",")
        attack_data = attack_data.drop(["Timestamp","AIT201"], axis=1)

    elif dataset == 'WADI':

        normal_data_path = "Data/WADI_14days.csv"
        normal_data = pd.read_csv(normal_data_path, delimiter=",")
        normal_data = normal_data.drop(["Date", "Time"], axis=1)

        attack_data_path = "Data/WADI_attackdata_labelled.csv"
        attack_data = pd.read_csv(attack_data_path, delimiter=",")
        attack_data = attack_data.drop(["Date", "Time"], axis=1)

    #fill NA values
    normal_data = np.asarray(normal_data.fillna(method='ffill'))
    #normalize
    scaler = MinMaxScaler()
    normal_data = scaler.fit_transform(normal_data)
    # create sliding window-based samples
    normal_data = split_data(normal_data, var.seq_len)
    # reshape
    normal_data = normal_data.reshape(-1,normal_data.shape[1]*normal_data.shape[2])

    # fill NA values
    attack_data = np.asarray(attack_data.fillna(method = 'ffill'))
    #remove label from data
    attack_x = attack_data[:,:-1]
    #normalize
    attack_x = scaler.transform(attack_x)
    #create sliding window based samples
    attack_x = split_data(attack_x, var.seq_len)
    # labels
    attack_y = attack_data[-len(attack_x):,-1].astype(int)
    #reshape
    attack_x = attack_x.reshape(-1,attack_x.shape[1]*attack_x.shape[2])

    # tensors
    normal_data = torch.as_tensor(normal_data, dtype = torch.float)
    attack_x = torch.as_tensor(attack_x, dtype = torch.float)

    train_loader, val_loader = create_train_set(normal_data)
        
    return train_loader, val_loader, attack_x, attack_y

def get_model_size(dataset):
    assert dataset in ['SWaT', 'WADI']
    if dataset == 'SWaT':
        enc_hidden = [500, 200, 100, 50]
    if dataset == 'WADI':
        enc_hidden = [1220,500, 200, 100]
    dec_hidden = enc_hidden[::-1]

    return enc_hidden, dec_hidden