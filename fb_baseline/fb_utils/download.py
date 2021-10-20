import os
import tarfile
import gdown


DATASETS_ID = {
    'handwritten': '1ptSVo4VEE-ed8-8NMg-DOhhBc9uGlFgK',
    'detection': '1kVFvoz6jDXckEtBQRg3pkQB_qVbsAAMG',
    'vqa': '1YK9vryTt8Dv4ftz-JqHohTG4aCM_6W4A',
    'c2c': '1YnrXdckRIiykCkz61cnW_b1x4angrbJl',
    'saved_models': '1Zd8isAp0lqgU95Xtn01R9df251NmY46X',
}


def extract_archive(archive_path, extract_path):
    if archive_path.endswith('tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            tar_file.extractall(path=extract_path)


def download_and_extract(data_dir, dataset_name):
    archive_path = os.path.join(data_dir, f'{dataset_name}.tar.gz')
    if not os.path.exists(archive_path):
        gdown.download(f'https://drive.google.com/uc?id={DATASETS_ID[dataset_name]}', archive_path, quiet=False)

    if not os.path.exists(os.path.join(data_dir, dataset_name)):
        extract_archive(archive_path, data_dir)
