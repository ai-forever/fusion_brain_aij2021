import os
import tarfile
import gdown


DATASETS_ID = {
    'ravdess_16k': '1uGDJYlFYaQg3cPiu-oicAXJqATMJ-wXv',
    'covid_tweets': '1_l27fVZbgVSGHku7f2cg_YVrKdJI5xuc',
    'handwritten': '1ptSVo4VEE-ed8-8NMg-DOhhBc9uGlFgK',
    'saved_models': '1Zd8isAp0lqgU95Xtn01R9df251NmY46X',
}


def extract_archive(archive_path, extract_path):
    if archive_path.endswith('tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, path=extract_path)


def download_and_extract(data_dir, dataset_name):
    archive_path = os.path.join(data_dir, f'{dataset_name}.tar.gz')
    if not os.path.exists(archive_path):
        gdown.download(f'https://drive.google.com/uc?id={DATASETS_ID[dataset_name]}', archive_path, quiet=False)

    if not os.path.exists(os.path.join(data_dir, dataset_name)):
        extract_archive(archive_path, data_dir)
