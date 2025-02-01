from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, UCRAnomalyloader, SMDSegLoader_Original, \
    MSLSegLoader_Original, PSMSegLoader_Original, SMAPSegLoader_Original, SWATSegLoader_Original
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'PSM_Original': PSMSegLoader_Original,
    'MSL': MSLSegLoader,
    'MSL_Original': MSLSegLoader_Original,
    'SMAP': SMAPSegLoader,
    'SMAP_Original': SMAPSegLoader_Original,
    'SMD': SMDSegLoader,
    'SMD_Original': SMDSegLoader_Original,
    'SWAT': SWATSegLoader,
    'SWAT_Original': SWATSegLoader_Original,
    'UEA': UEAloader,
    'UCRA': UCRAnomalyloader,
}

class DatasetCatalog:
    DATASETS = {
        'ETTh1': {
            "root_path": "/storage/dataset/DSET924f39a246e2bcba76feef284556"
        },
        'ETTh2': {
            "root_path": "/storage/dataset/DSETecc2e54a4c80a793255c932e7b72"
        },
        'ETTm1': {
            "root_path": "/storage/dataset/DSET778fcf74414d8e186fd05350ebee"
        },
        'ETTm2': {
            "root_path": "/storage/dataset/DSETe9eb0a5a4b40876add2dbd3acb6a"
        },
        'ECL': {
            "root_path": "/storage/dataset/DSET73e1e542467986886113370b39d1"
        },
        'Traffic': {
            "root_path": "/storage/dataset/DSET69dd739245f59853a74d98d2cc4c"
        },
        'Weather': {
            "root_path": "/storage/dataset/DSETb990ae96465d9eff1bfff43e5eca"
        },
        'ILI': {
            "root_path": "/storage/dataset/DSET8a91cb7146f58a081f7fe7561dea"
        },
        'Exchange': {
            "root_path": "/storage/dataset/DSETffd84b7f4e4e81ad73db993d91e8"
        },
        'm4': {
            "root_path": "/storage/dataset/DSET14960c3e4f4f8455ea397c95d6fc/m4"
        },
        'MSL': {
            "root_path": "/storage/dataset/DSETdd12154a46d6a95bf08e5b859e79/anomaly_detection/MSL"
        },
        'PSM': {
            "root_path": "/storage/dataset/DSETdd12154a46d6a95bf08e5b859e79/anomaly_detection/PSM"
        },
        'SMAP': {
            "root_path": "/storage/dataset/DSETdd12154a46d6a95bf08e5b859e79/anomaly_detection/SMAP"
        },
        'SMD': {
            "root_path": "/storage/dataset/DSETdd12154a46d6a95bf08e5b859e79/anomaly_detection/SMD"
        },
        'SWaT': {
            "root_path": "/storage/dataset/DSETdd12154a46d6a95bf08e5b859e79/anomaly_detection/SWaT"
        },
        'EthanolConcentration': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/EthanolConcentration"
        },
        'FaceDetection': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/FaceDetection"
        },
        'Handwriting': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/Handwriting"
        },
        'Heartbeat': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/Heartbeat"
        },
        'JapaneseVowels': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/JapaneseVowels"
        },
        'PEMS-SF': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/PEMS-SF"
        },
        'SelfRegulationSCP1': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/SelfRegulationSCP1"
        },
        'SelfRegulationSCP2': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/SelfRegulationSCP2"
        },
        'SpokenArabicDigits': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/SpokenArabicDigits"
        },
        'UWaveGestureLibrary': {
            "root_path": "/storage/dataset/DSET91c90c4342f58692e53c088e8e81/classification/UWaveGestureLibrary"
        }
    }
    
    @staticmethod
    def get(name):
        if name in DatasetCatalog.DATASETS:
            return DatasetCatalog.DATASETS[name]
        
        raise RuntimeError("Dataset not available: {}".format(name))

def data_provider(args, flag):
    Data = data_dict[args.data]
    if args.use_anylearn:
        Data_ATTRS = DatasetCatalog.get(args.data)
        root_path = Data_ATTRS['root_path']
        args.root_path = root_path
    
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = 1 if (flag=='test' or flag=='TEST') else args.batch_size
    freq = args.freq

    # if args.task_name == 'anomaly_detection':
    if 'anomaly_detection' in args.task_name:
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            step=args.stride,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification' or args.task_name == 'classification_ablation':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
