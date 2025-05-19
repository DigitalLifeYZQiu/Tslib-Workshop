from anylearn.config import init_sdk
from anylearn.interfaces.resource import SyncResourceUploader
init_sdk('http://anylearn.nelbds.cn', 'DigitalLifeYZQiu', 'Qyz20020318!')

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from anylearn.applications.quickstart import quick_train

train_task, algo, dset, project = quick_train(
    project_name="协鑫项目",
    algorithm_force_update=True,
    algorithm_local_dir='./',
    algorithm_cloud_name="tslib-workshop",
    algorithm_entrypoint="bash scriptsTimeEval/Theta_STPrice.sh",
    algorithm_output="./anylearn_output/",
    quota_group_request={
            'name': 'CNNC',
            'RTX-3090-unique': 1,
            'CPU': 20,
            'Memory': 25,
        },
    image_name="QUICKSTART_PYTORCH2.1.2_CUDA11.7_PYTHON3.11",
    dataset_id="DSET5753f2134495b68d38f3740f9368",
    )

print(train_task)